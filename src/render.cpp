#include "allocator.h"
#include "rt_helper.h"
#include <cstdint>
using namespace AscendC;
#include <cstdlib>
using namespace std;

constexpr int32_t TOTAL_NUM = WIDTH * HEIGHT * SAMPLES * 4;
constexpr int32_t USE_CORE_NUM = 8;                                        // device core num
constexpr int32_t BLOCK_LENGTH = TOTAL_NUM / USE_CORE_NUM;                 // 每个block处理的数据量(非字节数)
constexpr int32_t BUFFER_NUM = 2;                                          // fix double buffer -> pipeline
constexpr int32_t TILING_NUM = BLOCK_LENGTH / (GENERIC_SIZE * BUFFER_NUM); // custom config

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
        inputSpheres.SetGlobalBuffer((__gm__ Float *)spheres, 512 / sizeof(Float));

        pipe.InitBuffer(rayQueue, BUFFER_NUM, GENERIC_SIZE * sizeof(Float) * 6);   // ray xyz dxdydz = 6
        pipe.InitBuffer(colorQueue, BUFFER_NUM, GENERIC_SIZE * sizeof(Float) * 3); // color xyz = 3
        pipe.InitBuffer(sphereQueue, 1, 512);                                      // num * bytes size * member num

        pipe.InitBuffer(sphereBuf, Round256(SPHERE_NUM * sizeof(Float) * SPHERE_MEMBER_NUM)); // num * bytes size * member num

        pipe.InitBuffer(tmpBuf, GENERIC_SIZE * sizeof(Float) * (GENERIC_SIZE));
        pipe.InitBuffer(tmpIndexBuf, SPHERE_NUM * sizeof(uint32_t));
    }

    __aicore__ inline void Process() {
#ifdef __CCE_KT_TEST__
        // if (GetBlockIdx() == 0) {
        //     printf("core %ld\n", GetBlockIdx());
        //     auto ch = getchar();
        // }
#endif
        DataFormatCheck();

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
        ASSERT(TOTAL_NUM % USE_CORE_NUM == 0);    // ,"Total num must be divisible by use core num"
        ASSERT(BLOCK_LENGTH % TILING_NUM == 0);   // ,"Block length must be divisible by tiling num"
        ASSERT(GENERIC_SIZE == 64);               // ,"Tiling length must be 64"
        ASSERT(BLOCK_LENGTH % GENERIC_SIZE == 0); // ,"Block length must be divisible by tiling length"
    }

    // upload sphere data to device memory
    __aicore__ inline void UploadSpheres() {
        sphereData = sphereBuf.Get<Float>();
        DataCopy(sphereData, inputSpheres, 512 / sizeof(Float));
    }

    // system mem -> device memory
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<Float> ray = rayQueue.AllocTensor<Float>();

        // offset
        int32_t r_x = GENERIC_SIZE * 0;
        int32_t r_y = GENERIC_SIZE * 1;
        int32_t r_z = GENERIC_SIZE * 2;
        int32_t r_dx = GENERIC_SIZE * 3;
        int32_t r_dy = GENERIC_SIZE * 4;
        int32_t r_dz = GENERIC_SIZE * 5;

        DataCopy(ray[r_x], inputRays.ox[progress * GENERIC_SIZE], GENERIC_SIZE);
        DataCopy(ray[r_y], inputRays.oy[progress * GENERIC_SIZE], GENERIC_SIZE);
        DataCopy(ray[r_z], inputRays.oz[progress * GENERIC_SIZE], GENERIC_SIZE);
        DataCopy(ray[r_dx], inputRays.dx[progress * GENERIC_SIZE], GENERIC_SIZE);
        DataCopy(ray[r_dy], inputRays.dy[progress * GENERIC_SIZE], GENERIC_SIZE);
        DataCopy(ray[r_dz], inputRays.dz[progress * GENERIC_SIZE], GENERIC_SIZE);

        rayQueue.EnQue(ray);
    }

    // read device memory & compute & output to device queue & all samples
    __aicore__ inline void Compute(int32_t progress) {
        // printf("compute %d\n", progress);

        LocalTensor<Float> ray = rayQueue.DeQue<Float>();           // xxx |yyy|zzz| dxdxdx |dydydy|dzdzdz
        LocalTensor<Float> color = colorQueue.AllocTensor<Float>(); // xxx |yyy|zzz

        Allocator allocator;
        LocalTensor<Float> tmpBuffer = tmpBuf.Get<Float>();
        allocator.Init(tmpBuffer, GENERIC_SIZE * (GENERIC_SIZE));

        auto retBuffer = AllocDecorator(allocator.Alloc(GENERIC_SIZE * 3));
        VecLocalSoA ret;
        ret.Init(retBuffer.Get(), GENERIC_SIZE);
        Duplicate(ret.x, Float(1.0), GENERIC_SIZE);
        Duplicate(ret.y, Float(1.0), GENERIC_SIZE);
        Duplicate(ret.z, Float(1.0), GENERIC_SIZE);

        auto retMask = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
        Duplicate(retMask.Get().ReinterpretCast<uint16_t>(), uint16_t(UINT16_MAX), GENERIC_SIZE * sizeof(Float) / sizeof(uint16_t));

        RayLocalSoA rays;
        rays.Init(ray, GENERIC_SIZE);

        VecLocalSoA colors;
        colors.Init(color, GENERIC_SIZE);
        Duplicate(colors.x, Float(0), GENERIC_SIZE);
        Duplicate(colors.y, Float(0), GENERIC_SIZE);
        Duplicate(colors.z, Float(0), GENERIC_SIZE);

        SphereLocalSoA spheres;
        spheres.Init(sphereData);

        // DEBUG({
        //     printf("retBuffer\n");
        //     CPUDumpTensor("retBuffer", retBuffer.Get(), GENERIC_SIZE * 3);
        // })

        // static int32_t cnt = 0;
        int bound = 0;
        while (bound < 5) {
            // Step1: compute ray-sphere intersection
            auto hitMinT = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
            auto hitIndex = AllocDecorator(allocator.Alloc(GENERIC_SIZE));

            ComputeHitInfo(hitMinT.Get(), hitIndex.Get(), rays, spheres, allocator, bound);

            // DEBUG(if (cnt == 0 && bound == 1) {
            //     printf("bound: %d\n", bound);
            //     CPUDumpTensorU("ReduceMin Index int32_T", hitIndex.Get().ReinterpretCast<int32_t>(), GENERIC_SIZE);
            // })

            // Step4: update ray info, compute hitPos-rayPos,hitPos-SpherePos-rayDir
            GenerateNewRays(rays, hitIndex.Get(), hitMinT.Get(), spheres, allocator, bound);

            // DEBUG({
            //     if (cnt == 0)
            //     {
            //         CPUDumpTensor("New ray x", rays.ox, GENERIC_SIZE);
            //         CPUDumpTensor("New ray y", rays.oy, GENERIC_SIZE);
            //         CPUDumpTensor("New ray z", rays.oz, GENERIC_SIZE);
            //         CPUDumpTensor("New ray dx", rays.dx, GENERIC_SIZE);
            //         CPUDumpTensor("New ray dy", rays.dy, GENERIC_SIZE);
            //         CPUDumpTensor("New ray dz", rays.dz, GENERIC_SIZE);
            //     }
            // })

            // Step5: compute diffuse color & mask
            AccumulateIntervalColor(ret, retMask.Get(), hitIndex.Get(), spheres, allocator, bound);

            // DEBUG({
            //     printf("process %d depth %d\n",progress,bound);
            //     CPUDumpTensor("Color x", ret.x, GENERIC_SIZE);
            //     CPUDumpTensor("Color y", ret.y, GENERIC_SIZE);
            //     CPUDumpTensor("Color z", ret.z, GENERIC_SIZE);
            // })


            bound++;
        }

        // Step6: compute final color

        // Step7: write to device queue

        Muls(colors.x, ret.x, Float(12), GENERIC_SIZE);
        Muls(colors.y, ret.y, Float(12), GENERIC_SIZE);
        Muls(colors.z, ret.z, Float(12), GENERIC_SIZE);

        // DEBUG({
        //     CPUDumpTensor("Color x", colors.x, GENERIC_SIZE);
        //     CPUDumpTensor("Color y", colors.y, GENERIC_SIZE);
        //     CPUDumpTensor("Color z", colors.z, GENERIC_SIZE);
        // })

        rayQueue.FreeTensor(ray);
        colorQueue.EnQue(color);
        // cnt++;
        // DEBUG({ printf("===========cnt: %d===================\n", cnt); })
    }

    // write device queue to system mem
    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<Float> color = colorQueue.DeQue<Float>();

        // offset
        int32_t c_x = GENERIC_SIZE * 0;
        int32_t c_y = GENERIC_SIZE * 1;
        int32_t c_z = GENERIC_SIZE * 2;

        DataCopy(resultColor.x[progress * GENERIC_SIZE], color[c_x], GENERIC_SIZE);
        DataCopy(resultColor.y[progress * GENERIC_SIZE], color[c_y], GENERIC_SIZE);
        DataCopy(resultColor.z[progress * GENERIC_SIZE], color[c_z], GENERIC_SIZE);

        colorQueue.FreeTensor(color);
    }

  private:
    int width;
    int height;
    int samples;

    // tmp memory allocator

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
