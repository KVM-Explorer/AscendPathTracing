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

        int32_t block_offset = BLOCK_LENGTH * GetBlockIdx();

        InitRaySoA(inputRays, r, block_offset, BLOCK_LENGTH);
        InitColorSoA(resultColor, output, block_offset, BLOCK_LENGTH);
        inputSpheres.SetGlobalBuffer((__gm__ Float *)spheres, SPHERE_NUM * SPHERE_MEMBER_NUM);

        pipe.InitBuffer(rayQueue, BUFFER_NUM, TILING_LENGTH * sizeof(Float) * 6);        // ray xyz dxdydz = 6
        pipe.InitBuffer(colorQueue, BUFFER_NUM, TILING_LENGTH * sizeof(Float) * 3);      // color xyz = 3
        pipe.InitBuffer(sphereQueue, 1, SPHERE_NUM * sizeof(Float) * SPHERE_MEMBER_NUM); // num * bytes size * member num

        pipe.InitBuffer(sphereBuf, SPHERE_NUM * sizeof(Float) * SPHERE_MEMBER_NUM); // num * bytes size * member num

        pipe.InitBuffer(tmpBuf, TILING_LENGTH * sizeof(Float) * (8 + SPHERE_NUM));
        pipe.InitBuffer(maskBuf, 128);
        pipe.InitBuffer(tmpIndexBuf,SPHERE_NUM * sizeof(uint32_t));

        // REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm); // 初始化
    }

    __aicore__ inline void Process() {

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
    // upload sphere data to device memory
    __aicore__ inline void UploadSpheres() {
        sphereData = sphereBuf.Get<Float>();
        DataCopy(sphereData, inputSpheres, SPHERE_NUM * 10);
    }

    __aicore__ inline void GenerateIndices() {
        indexData = tmpIndexBuf.Get<uint32_t>();
        uint32_t cur = 0;
        for(int i=0;i<SPHERE_NUM;i++){
            indexData.SetValue(i,cur);
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

        LocalTensor<Float> ocX = tmpBuffer[0];
        LocalTensor<Float> ocY = tmpBuffer[TILING_LENGTH * 1];
        LocalTensor<Float> ocZ = tmpBuffer[TILING_LENGTH * 2];
        LocalTensor<Float> tmp1 = tmpBuffer[TILING_LENGTH * 3];
        LocalTensor<Float> tmp2 = tmpBuffer[TILING_LENGTH * 4];
        LocalTensor<Float> tmp3 = tmpBuffer[TILING_LENGTH * 5];
        LocalTensor<Float> b = tmpBuffer[TILING_LENGTH * 6];
        LocalTensor<Float> c = tmpBuffer[TILING_LENGTH * 7];
        LocalTensor<Float> stage1val = tmpBuffer[TILING_LENGTH * 8];

        LocalTensor<uint8_t> maskeBase = maskBuf.Get<uint8_t>();
        LocalTensor<uint8_t> mask1 = maskeBase[0];  // addr: 0  half128bit，float64bit-> 8Bytes
        LocalTensor<uint8_t> mask2 = maskeBase[32]; // addr: TILE_LENGTH/8 -> 8Bytes FIXME: 不满足32Bytes对齐
        LocalTensor<uint8_t> mask3 = maskeBase[64];
        LocalTensor<uint8_t> stage1mask = maskeBase[96];


        // count
        int32_t count = TILING_LENGTH;
        RayLocalSoA rays;
        rays.ox = ray[TILING_LENGTH * 0];
        rays.oy = ray[TILING_LENGTH * 1];
        rays.oz = ray[TILING_LENGTH * 2];
        rays.dx = ray[TILING_LENGTH * 3];
        rays.dy = ray[TILING_LENGTH * 4];
        rays.dz = ray[TILING_LENGTH * 5];

        VecLocalSoA colors;
        colors.x = color[TILING_LENGTH * 0];
        colors.y = color[TILING_LENGTH * 1];
        colors.z = color[TILING_LENGTH * 2];

        SphereLocalSoA spheres;
        int sphere_offset = 0;
        spheres.r2 = sphereData[sphere_offset + SPHERE_NUM * 0];
        spheres.x = sphereData[sphere_offset + SPHERE_NUM * 1];
        spheres.y = sphereData[sphere_offset + SPHERE_NUM * 2];
        spheres.z = sphereData[sphere_offset + SPHERE_NUM * 3];
        spheres.emissionX = sphereData[sphere_offset + SPHERE_NUM * 4];
        spheres.emissionY = sphereData[sphere_offset + SPHERE_NUM * 5];
        spheres.emissionZ = sphereData[sphere_offset + SPHERE_NUM * 6];
        spheres.colorX = sphereData[sphere_offset + SPHERE_NUM * 7];
        spheres.colorY = sphereData[sphere_offset + SPHERE_NUM * 8];
        spheres.colorZ = sphereData[sphere_offset + SPHERE_NUM * 9];

        // Step1: compute ray-sphere intersection
        for (int i = 0; i < SPHERE_NUM; i++) {
            int offset = i * count;

            Adds(ocX, rays.ox, -spheres.x.GetValue(i), count); // ocX = rays.ox - spheres.x
            Adds(ocY, rays.oy, -spheres.y.GetValue(i), count); // ocY = rays.oy - spheres.y
            Adds(ocZ, rays.oz, -spheres.z.GetValue(i), count); // ocZ = rays.oz - spheres.z

            Muls(tmp1, ocX, rays.ox.GetValue(i), count); // tmp1 = ocX * rays.ox
            Muls(tmp2, ocY, rays.oy.GetValue(i), count); // tmp2 = ocY * rays.oy
            Muls(tmp3, ocZ, rays.oz.GetValue(i), count); // tmp3 = ocZ * rays.oz

            Add(b, tmp1, tmp2, count); // b = tmp1 + tmp2 + tmp3
            Add(b, b, tmp3, count);

            Mul(tmp1, ocX, ocX, count); // tmp1 = ocX * ocX
            Mul(tmp2, ocY, ocY, count); // tmp2 = ocY * ocY
            Mul(tmp3, ocZ, ocZ, count); // tmp3 = ocZ * ocZ

            Add(c, tmp1, tmp2, count); // c = tmp1 + tmp2 + tmp3 - r^2 //TODO: 使用AddMul优化
            Add(c, c, tmp3, count);
            Adds(c, c, -spheres.r2.GetValue(i), count);

            // disc = b^2 - c
            Mul(tmp1, b, b, count); // tmp1 = b * b

            // mask1 = c > 0, 每个mask 64位正好对应Float 的256Bytes
            CompareScalar(mask1, c, Float(0), CMPMODE::GT, count);

            auto repeat = TILING_LENGTH * sizeof(Float) / 256; // 256 Bytes为单位读取

            {
                auto &discrSq = tmp2;
                auto &t0 = tmp3;
                auto &t1 = tmp1;

                Sqrt(discrSq, tmp1, count); // tmp2 = sqrt(tmp1)

                Sub(t0, b, discrSq, count); // t0 = b - discrSq
                Add(t1, b, discrSq, count); // t1 = b + discrSq

                // mask2 FIXME: mask 也需要32Bytes 对齐
                CompareScalar(mask2, t0, Float(0), CMPMODE::GT, count);                            // mask2 = t0 > 0
                Select(stage1val[offset], mask2, t0, t1, SELMODE::VSEL_TENSOR_TENSOR_MODE, count); // t = mask2 ? t0 : t1  
                // Tips: t still may < 0

                // mask3
                // mask3 = t > 0 && t in mask1
                // CompareScalar(mask3,stage1val,Float(0),CMPMODE::GT,count); // mask3 = t > 0
                // And(stage1mask.ReinterpretCast<uint16_t>(),mask3.ReinterpretCast<uint16_t>(),mask1.ReinterpretCast<uint16_t>(),count/8/ sizeof(uint16_t)); // retMask = mask3 & mask1
            }
        }


        // Step2: compute color
        for (int i = 0; i < count; i++) {
            Gather(tmp1, stage1val, indexData, i, SPHERE_NUM); 
            CompareScalar(mask1, tmp1, Float(0), CMPMODE::GT, SPHERE_NUM); // mask = t > 0 ，FIXME: 数据量太少没有256Byte对齐 需要保证calc * sizeof(element) % 256 == 0
            if(mask3.GetValue(0)==0){ // 8个sphere没有交点
                colors.x.SetValue(i, Float(0));
                colors.y.SetValue(i, Float(0));
                colors.z.SetValue(i, Float(0));
                continue;
            }else{
                colors.x.SetValue(i, Float(0.1));
                colors.y.SetValue(i, Float(0.5));
                colors.z.SetValue(i, Float(0.7));
            }            
        }

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
