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

        pipe.InitBuffer(tmpBuf, TILING_LENGTH * sizeof(Float) * (3 + SPHERE_NUM + 8));
        pipe.InitBuffer(maskBuf, 128);
        pipe.InitBuffer(tmpIndexBuf, SPHERE_NUM * sizeof(uint32_t));

        // REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm); // 初始化
    }

    __aicore__ inline void Process() {
        DataFormatCheck();

        constexpr int loop_count = TILING_NUM * BUFFER_NUM;

        UploadSpheres();
        GenerateIndices();
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

    // upload sphere data to device memory
    __aicore__ inline void UploadSpheres() {
        sphereData = sphereBuf.Get<Float>();
        DataCopy(sphereData, inputSpheres, SPHERE_NUM * 10);
    }

    __aicore__ inline void GenerateIndices() {
        indexData = tmpIndexBuf.Get<uint32_t>();
        uint32_t cur = 0;
        for (int i = 0; i < SPHERE_NUM; i++) {
            indexData.SetValue(i, cur);
            cur += TILING_LENGTH;
        }
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

        LocalTensor<Float> ray = rayQueue.DeQue<Float>();           // xxx |yyy|zzz| dxdxdx |dydydy|dzdzdz
        LocalTensor<Float> color = colorQueue.AllocTensor<Float>(); // xxx |yyy|zzz

        LocalTensor<Float> tmpBuffer = tmpBuf.Get<Float>();

        int32_t tmp_addr = 0;
        VecLocalSoA ret;
        ret.Init(tmpBuffer[tmp_addr], TILING_LENGTH);
        tmp_addr += TILING_LENGTH * 3; // ray count * 3
        LocalTensor<Float> stage1val = tmpBuffer[tmp_addr];
        tmp_addr += TILING_LENGTH * SPHERE_NUM; // ray count * SPHERE_NUM

        LocalTensor<Float> sharedTmpBuffer = tmpBuffer[tmp_addr]; // ray count  * 8

        LocalTensor<uint8_t> maskeBase = maskBuf.Get<uint8_t>();
        LocalTensor<uint8_t> retMask = maskeBase[0]; // addr: 0  half128bit，float64bit-> 8Bytes
        LocalTensor<uint8_t> mask2 = maskeBase[32];  // addr: TILE_LENGTH/8 -> 8Bytes FIXME: 不满足32Bytes对齐
        LocalTensor<uint8_t> mask3 = maskeBase[64];
        LocalTensor<uint8_t> stage1mask = maskeBase[96];

        // count
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
            auto dst = stage1val[offset];
            SphereHitInfo(dst, sharedTmpBuffer, cur_sphere, rays, TILING_LENGTH);
        }

        // if (get_block_idx() == 0) {
        //     for (int i = 0; i < SPHERE_NUM; i++) {
        //         printf("Core: %ld sphere:%d\n", GetBlockIdx(), i);
        //         for (int j = 0; j < TILING_LENGTH; j++) {
        //             printf("%5.2f, ", stage1val.GetValue(i * TILING_LENGTH + j));
        //         }
        //         printf("\n============\n");
        //     }
        // }

        // Reset val(<0) to INF
        // CompareScalar(stage1val,stage1val,Float(0),CMPMODE::GT,count*SPHERE_NUM); // mask = t > 0
        // Select(stage1val,mask1,stage1val,Float(1e20),SELMODE::VSEL_CMPMASK_SPR,SPHERE_NUM*count); // t = mask1 ? t : INF
        // for(int i=0;i<SPHERE_NUM;i++){
        //     printf("Core: %ld sphere:%d\n",GetBlockIdx(),i);
        //     for(int j=0;j<count;j++){
        //         printf("%5.2f, ",stage1val.GetValue(i*count+j));
        //     }
        //     printf("============\n");
        // }

        // Step2: compute color | Force Format to 256Bytes RayGroup
        uint64_t uint64Mask = (1ULL << 63);
        for (int i = 0; i < TILING_LENGTH; i++) {
            const uint64_t ray_mask[] = {(uint64Mask >> i), 0};
            LocalTensor<Float> tmp1 = tmpBuffer[tmp_addr];
            LocalTensor<Float> tmp2 = tmpBuffer[tmp_addr + TILING_LENGTH];
            // worklocal space:

            ReduceMin<Float>(tmp1, stage1val, tmp2, ray_mask, SPHERE_NUM, 8, true); // tmp1 = min(stage1val)
            auto val = tmp1.GetValue(0);

            auto index = tmp1.ReinterpretCast<uint32_t>().GetValue(1);
            index = index / TILING_LENGTH;

            // if (GetBlockIdx() == 0) {
            //     printf("id:%d, val:%f, index:%d\n", i, val, index);
            // }

            // hit sphere
            if (val > 0) {

                colors.x.SetValue(i,spheres.colorX.GetValue(index));
                colors.y.SetValue(i,spheres.colorY.GetValue(index));
                colors.z.SetValue(i,spheres.colorZ.GetValue(index));

                // color 0.1 0.5 0.7
                // colors.x.SetValue(i, Float(0.0));
                // colors.y.SetValue(i, Float(1.0));
                // colors.z.SetValue(i, Float(0.0));
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

    // global
    RaySoA inputRays;
    GlobalTensor<Float> inputSpheres;
    VecSoA resultColor;

    // Local
    LocalTensor<Float> sphereData;
    LocalTensor<uint32_t> indexData;

    AscendC::TBuf<QuePosition::VECIN> sphereBuf;
    AscendC::TBuf<QuePosition::VECCALC> tmpBuf;
    AscendC::TBuf<QuePosition::VECCALC> tmpIndexBuf;
    AscendC::TBuf<QuePosition::VECCALC> maskBuf;

    // matmul::Matmul<aType,bType,biasType,cType> mm;
    TQue<QuePosition::VECIN, BUFFER_NUM> rayQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> colorQueue;
    TQue<QuePosition::VECIN, 1> sphereQueue;

    TPipe pipe;
};

extern "C" __global__ __aicore__ void render(GM_ADDR rays, GM_ADDR spheres, GM_ADDR colors) {
    KernelRender op;

    op.Init(WIDTH, HEIGHT, SAMPLES, rays, spheres, colors);
    op.Process();
    op.Release();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void render_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *rays, uint8_t *spheres, uint8_t *colors) {
    render<<<blockDim, l2ctrl, stream>>>(rays, spheres, colors);
}
#endif
