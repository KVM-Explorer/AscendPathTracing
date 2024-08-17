#include "allocator.h"
#include "rt_helper.h"
#include <cstdint>
using namespace AscendC;
#include <cstdlib>
using namespace std;

constexpr int32_t TOTAL_NUM = WIDTH * HEIGHT * SAMPLES * 4;
constexpr int32_t USE_CORE_NUM = 8;                                       // device core num
constexpr int32_t BLOCK_LENGTH = TOTAL_NUM / USE_CORE_NUM;                // 每个block处理的数据量(非字节数)
constexpr int32_t TILING_NUM = 1;                                         // custom config
constexpr int32_t BUFFER_NUM = 2;                                         // fix double buffer -> pipeline
constexpr int32_t TILING_LENGTH = BLOCK_LENGTH / TILING_NUM / BUFFER_NUM; // 真正每次处理的数据数量(非字节数)

class KernelRender {

  public:
    __aicore__ inline KernelRender() {}
    __aicore__ inline void Init(int w, int h, int s, GM_ADDR r, GM_ADDR spheres, GM_ADDR output) {

        width = w;
        height = h;
        samples = s;

        int32_t block_offset = BLOCK_LENGTH * GetBlockIdx(); // 每个Core计算一部分竖条纹

        InitRaySoA(inputRays, r, block_offset, BLOCK_LENGTH);
        InitColorSoA(resultColor, output, block_offset, BLOCK_LENGTH);
        inputSpheres.SetGlobalBuffer((__gm__ Float *)spheres, SPHERE_NUM * SPHERE_MEMBER_NUM);

        pipe.InitBuffer(rayQueue, BUFFER_NUM, TILING_LENGTH * sizeof(Float) * 6);        // ray xyz dxdydz = 6
        pipe.InitBuffer(colorQueue, BUFFER_NUM, TILING_LENGTH * sizeof(Float) * 3);      // color xyz = 3
        pipe.InitBuffer(sphereQueue, 1, SPHERE_NUM * sizeof(Float) * SPHERE_MEMBER_NUM); // num * bytes size * member num

        pipe.InitBuffer(sphereBuf, SPHERE_NUM * sizeof(Float) * SPHERE_MEMBER_NUM); // num * bytes size * member num

        pipe.InitBuffer(tmpBuf, TILING_LENGTH * sizeof(Float) * (3 + SPHERE_NUM + 8 + 20));
        pipe.InitBuffer(tmpIndexBuf, SPHERE_NUM * sizeof(uint32_t));
    }

    __aicore__ inline void Process() {
        // if (GetBlockIdx() == 0) {
        //     printf("core %ld\n", GetBlockIdx());
        //     auto ch = getchar();
        // }

        DataFormatCheck();
        InitAllocator();
        constexpr int loop_count = TILING_NUM * BUFFER_NUM;

        UploadSpheres();
        for (int i = 0; i < loop_count; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

    __aicore__ inline void Release() {
        
        // free local tensor
    }

  private:
    __aicore__ inline void DataFormatCheck() {
        ASSERT(TOTAL_NUM % USE_CORE_NUM == 0);  // ,"Total num must be divisible by use core num"
        ASSERT(BLOCK_LENGTH % TILING_NUM == 0); // ,"Block length must be divisible by tiling num"
        ASSERT(TILING_LENGTH == 64);            // ,"Tiling length must be 64"
    }

    __aicore__ inline void InitAllocator() {
        LocalTensor<Float> tmpBuffer = tmpBuf.Get<Float>();
        allocator.Init(tmpBuffer, TILING_LENGTH * (SPHERE_NUM + 20));
    }

    // upload sphere data to device memory
    __aicore__ inline void UploadSpheres() {
        sphereData = sphereBuf.Get<Float>();
        DataCopy(sphereData, inputSpheres, SPHERE_NUM * 10);
    }

    // system mem -> device memory
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<Float> ray = rayQueue.AllocTensor<Float>();

        // offset
        int32_t r_x = TILING_LENGTH * 0;
        int32_t r_y = TILING_LENGTH * 1;
        int32_t r_z = TILING_LENGTH * 2;
        int32_t r_dx = TILING_LENGTH * 3;
        int32_t r_dy = TILING_LENGTH * 4;
        int32_t r_dz = TILING_LENGTH * 5;

        DataCopy(ray[r_x], inputRays.ox[progress * TILING_LENGTH], TILING_LENGTH);
        DataCopy(ray[r_y], inputRays.oy[progress * TILING_LENGTH], TILING_LENGTH);
        DataCopy(ray[r_z], inputRays.oz[progress * TILING_LENGTH], TILING_LENGTH);
        DataCopy(ray[r_dx], inputRays.dx[progress * TILING_LENGTH], TILING_LENGTH);
        DataCopy(ray[r_dy], inputRays.dy[progress * TILING_LENGTH], TILING_LENGTH);
        DataCopy(ray[r_dz], inputRays.dz[progress * TILING_LENGTH], TILING_LENGTH);

        rayQueue.EnQue(ray);
    }

    // read device memory & compute & output to device queue & all samples
    __aicore__ inline void Compute(int32_t progress) {
        // printf("compute %d\n", progress);

        LocalTensor<Float> ray = rayQueue.DeQue<Float>();           // xxx |yyy|zzz| dxdxdx |dydydy|dzdzdz
        LocalTensor<Float> color = colorQueue.AllocTensor<Float>(); // xxx |yyy|zzz

        // VecLocalSoA ret;
        // ret.Init(tmpBuffer[tmp_addr], TILING_LENGTH);

        auto stage1val = AllocDecorator( allocator.Alloc(TILING_LENGTH * SPHERE_NUM));

        RayLocalSoA rays;
        rays.Init(ray, TILING_LENGTH);

        VecLocalSoA colors;
        colors.Init(color, TILING_LENGTH);

        SphereLocalSoA spheres;
        spheres.Init(sphereData);

        // Step1: compute ray-sphere intersection
        for (int i = 0; i < SPHERE_NUM; i++) {
            int offset = i * TILING_LENGTH;
            // clang-format off
            auto cur_sphere =Sphere{
                .r2 = spheres.r2.GetValue(i), 
                .x = spheres.x.GetValue(i), 
                .y = spheres.y.GetValue(i),
                .z = spheres.z.GetValue(i)};
            // clang-format on
            auto dst = stage1val.Get()[offset];
            SphereHitInfo(dst, allocator, cur_sphere, rays, TILING_LENGTH);
        }

        // Step2: compute color | Force Format to 256Bytes RayGroup
        uint64_t uint64Mask = (1ULL << 63);
        for (int i = 0; i < TILING_LENGTH; i++) {
            const uint64_t ray_mask[] = {(uint64Mask >> i), 0};
            auto minResult = AllocDecorator(allocator.Alloc(TILING_LENGTH));
            auto workspace = AllocDecorator(allocator.Alloc(TILING_LENGTH * 8)); // FIXME: 计算精准计算最小值需要的临时空间

            ReduceMin<Float>(minResult.Get(), stage1val.Get(), workspace.Get(), ray_mask, SPHERE_NUM, 8, true); // tmp1 = min(stage1val)
            auto val = minResult.Get().GetValue(0);

            auto index = minResult.Get().ReinterpretCast<uint32_t>().GetValue(1);
            index = index / TILING_LENGTH;

            // hit sphere
            if (val > 0) {

                colors.x.SetValue(i, spheres.colorX.GetValue(index));
                colors.y.SetValue(i, spheres.colorY.GetValue(index));
                colors.z.SetValue(i, spheres.colorZ.GetValue(index));
            } else {

                Float cx, cy, cz;
                Float ex, ey, ez;
                ex = spheres.emissionX.GetValue(index);
                ey = spheres.emissionX.GetValue(index);
                ez = spheres.emissionX.GetValue(index);

                colors.x.SetValue(i, 0);
                colors.y.SetValue(i, 0);
                colors.z.SetValue(i, 0);
            }
        }

        // for (int i = 0; i < TILING_LENGTH; i++) {
        //     colors.x.SetValue(i, rays.dx.GetValue(i));
        //     colors.y.SetValue(i, rays.dy.GetValue(i));
        //     colors.z.SetValue(i, rays.dz.GetValue(i));
        // }

        rayQueue.FreeTensor(ray);
        colorQueue.EnQue(color);
    }

    // write device queue to system mem
    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<Float> color = colorQueue.DeQue<Float>();

        // offset
        int32_t c_x = TILING_LENGTH * 0;
        int32_t c_y = TILING_LENGTH * 1;
        int32_t c_z = TILING_LENGTH * 2;

        DataCopy(resultColor.x[progress * TILING_LENGTH], color[c_x], TILING_LENGTH);
        DataCopy(resultColor.y[progress * TILING_LENGTH], color[c_y], TILING_LENGTH);
        DataCopy(resultColor.z[progress * TILING_LENGTH], color[c_z], TILING_LENGTH);

        colorQueue.FreeTensor(color);
    }

  private:
    int width;
    int height;
    int samples;

    // tmp memory allocator
    Allocator allocator;

    // global
    RaySoA inputRays;
    GlobalTensor<Float> inputSpheres;
    VecSoA resultColor;

    // Local
    LocalTensor<Float> sphereData;

    AscendC::TBuf<QuePosition::VECIN> sphereBuf;
    AscendC::TBuf<QuePosition::VECCALC> tmpBuf;
    AscendC::TBuf<QuePosition::VECCALC> tmpIndexBuf;

    // matmul::Matmul<aType,bType,biasType,cType> mm;
    TQue<QuePosition::VECIN, BUFFER_NUM> rayQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> colorQueue;
    TQue<QuePosition::VECIN, 1> sphereQueue;

    TPipe pipe;
};

// cpu kernel function
extern "C" __global__ __aicore__ void render(GM_ADDR rays, GM_ADDR spheres, GM_ADDR colors) {
    KernelRender op;

    op.Init(WIDTH, HEIGHT, SAMPLES, rays, spheres, colors);
    op.Process();
    op.Release();
}

// npu kernel function
#ifndef __CCE_KT_TEST__
// call of kernel function
void render_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *rays, uint8_t *spheres, uint8_t *colors) {
    render<<<blockDim, l2ctrl, stream>>>(rays, spheres, colors);
}
#endif
