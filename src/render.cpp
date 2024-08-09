#include "rt_helper.h"
#include <cstdint>
using namespace AscendC;
#include <cstdlib>
// #include <lib/matmul_intf.h>
using namespace std;

constexpr int32_t TOTAL_NUM = WIDTH * HEIGHT * SAMPLES * 4;
constexpr int32_t USE_CORE_NUM = 8;                                       // device core num
constexpr int32_t BLOCK_LENGTH = TOTAL_NUM / USE_CORE_NUM;                // 每个block处理的数据量(非字节数)
constexpr int32_t TILING_NUM = 8;                                         // custom config
constexpr int32_t BUFFER_NUM = 2;                                         // fix double buffer -> pipeline
constexpr int32_t TILING_LENGTH = BLOCK_LENGTH / TILING_NUM / BUFFER_NUM; // 真正每次处理的数据数量(非字节数)

// using aType =  matmul::MatmulType<TPosition::GM, CubeFormat::ND, Float>;
// using bType = matmul::MatmulType<TPosition::GM,CubeFormat::ND, Float>;
// using cType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, Float>;
// using biasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, Float>;

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

        pipe.InitBuffer(tmpBuf, SPHERE_NUM * sizeof(Float) * 8);
        pipe.InitBuffer(maskBuf, SPHERE_NUM * sizeof(uint8_t) * 1);

        // REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm); // 初始化
    }

    __aicore__ inline void Process() {

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

        LocalTensor<Float> ray = rayQueue.DeQue<Float>();           // xxx |yyy|zzz| dxdxdx |dydydy|dzdzdz
        LocalTensor<Float> color = colorQueue.AllocTensor<Float>(); // xxx |yyy|zzz

        LocalTensor<Float> tmpBuffer = tmpBuf.Get<Float>();

        LocalTensor<Float> ocX = tmpBuffer[0];
        LocalTensor<Float> ocY = tmpBuffer[SPHERE_NUM * 1];
        LocalTensor<Float> ocZ = tmpBuffer[SPHERE_NUM * 2];
        LocalTensor<Float> tmp1 = tmpBuffer[SPHERE_NUM * 3];
        LocalTensor<Float> tmp2 = tmpBuffer[SPHERE_NUM * 4];
        LocalTensor<Float> tmp3 = tmpBuffer[SPHERE_NUM * 5];
        LocalTensor<Float> b = tmpBuffer[SPHERE_NUM * 6];
        LocalTensor<Float> c = tmpBuffer[SPHERE_NUM * 7];
        // LocalTensor<Float> discrPos = tmpBuf.Get<Float>();
        LocalTensor<uint8_t> mask1 = maskBuf.Get<uint8_t>();// TODO: mask buffer

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

        // generate color
        // Add(colors.x, rays.ox, rays.dx, count);
        // Add(colors.y, rays.oy, rays.dy, count);
        // Add(colors.z, rays.oz, rays.dz, count);

        // Mins(colors.x, colors.x, Float(1.0f) , count);
        // Mins(colors.y, colors.y,Float(1.0f), count);
        // Mins(colors.z, colors.z, Float(1.0f), count);

        for (int i = 0; i < count; i++) {
            Adds(ocX, spheres.x, rays.ox.GetValue(i), SPHERE_NUM);
            Adds(ocY, spheres.y, rays.oy.GetValue(i), SPHERE_NUM);
            Adds(ocZ, spheres.z, rays.oz.GetValue(i), SPHERE_NUM);

            Muls(tmp1, ocX, rays.ox.GetValue(i), SPHERE_NUM);
            Muls(tmp2, ocY, rays.oy.GetValue(i), SPHERE_NUM);
            Muls(tmp3, ocZ, rays.oz.GetValue(i), SPHERE_NUM);

            Add(b,tmp1,tmp2,SPHERE_NUM);
            Add(b,b,tmp3,SPHERE_NUM);

            Mul(tmp1,ocX,ocX,SPHERE_NUM);
            Mul(tmp2,ocY,ocY,SPHERE_NUM);
            Mul(tmp3,ocZ,ocZ,SPHERE_NUM);

            Add(c,tmp1,tmp2,SPHERE_NUM);
            Add(c,c,tmp3,SPHERE_NUM);

            // disc = b^2 - c
            Mul(tmp1,b,b,SPHERE_NUM);
            Sub(tmp2,tmp1,c,SPHERE_NUM);

            // if disc < 0, no intersection
            CompareScalar(mask1,tmp1,Float(0),CMPMODE::GT,SPHERE_NUM);
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

    AscendC::TBuf<QuePosition::VECIN> sphereBuf;
    AscendC::TBuf<QuePosition::VECCALC> tmpBuf;
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
