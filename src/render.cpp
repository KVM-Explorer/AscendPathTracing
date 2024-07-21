#include "rt_helper.h"
#include <cstdint>
using namespace AscendC;
#include <cstdlib>
using namespace std;

constexpr int32_t TOTAL_NUM = WIDTH * HEIGHT * SAMPLES * 4;
constexpr int32_t USE_CORE_NUM = 8;
constexpr int32_t BLOCK_LENGTH = TOTAL_NUM / USE_CORE_NUM;
constexpr int32_t TILING_NUM = 8;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILING_LENGTH = BLOCK_LENGTH / TILING_NUM / BUFFER_NUM;

class KernelRender {

  public:
    __aicore__ inline KernelRender() {}
    __aicore__ inline void Init(int w, int h, int s, GM_ADDR r, GM_ADDR output) {

        width = w;
        height = h;
        samples = s;

        constexpr int32_t block_length = TOTAL_NUM / USE_CORE_NUM;
        int32_t block_offset = block_length * GetBlockIdx();

        InitRaySoA(inputRays, r, block_offset, block_length);
        InitColorSoA(resultColor, output, block_offset, block_length);

        pipe.InitBuffer(rayQueue, BUFFER_NUM, TILING_LENGTH * sizeof(Float) * 6); // ray xyz dxdydz = 6
        pipe.InitBuffer(rayQueue, BUFFER_NUM, TILING_LENGTH * sizeof(Float) * 3); // color xyz = 3
    }

    __aicore__ inline void Process() {

        constexpr int loop_count = TILING_NUM * BUFFER_NUM;
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
    // system mem -> device mem
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

    // read device mem & compute & output to device queue & all samples
    __aicore__ inline void Compute(int32_t progress) {

        LocalTensor<Float> ray = rayQueue.DeQue<Float>();           // xxx |yyy|zzz| dxdxdx |dydydy|dzdzdz
        LocalTensor<Float> color = colorQueue.AllocTensor<Float>(); // xxx |yyy|zzz

        // count
        int32_t count = TILING_LENGTH;

        LocalTensor<Float> rayX = ray[TILING_LENGTH * 0];
        LocalTensor<Float> rayY = ray[TILING_LENGTH * 1];
        LocalTensor<Float> rayZ = ray[TILING_LENGTH * 2];
        LocalTensor<Float> rayDX = ray[TILING_LENGTH * 3];
        LocalTensor<Float> rayDY = ray[TILING_LENGTH * 4];
        LocalTensor<Float> rayDZ = ray[TILING_LENGTH * 5];

        LocalTensor<Float> colorX = color[TILING_LENGTH * 0];
        LocalTensor<Float> colorY = color[TILING_LENGTH * 1];
        LocalTensor<Float> colorZ = color[TILING_LENGTH * 2];

        // generate color
        Add(colorX, rayX, rayDX, count);
        Add(colorY, rayY, rayDY, count);
        Add(colorZ, rayZ, rayDZ, count);

        Mins(colorX, colorX, 1.0f, count);
        Mins(colorY, colorY, 1.0f, count);
        Mins(colorZ, colorZ, 1.0f, count);

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
    VecSoA resultColor;

    // local
    RayLocalSoA rays;
    VecLocalSoA colors;

    // 输入队列
    TQue<QuePosition::VECIN, BUFFER_NUM> rayQueue;
    // 输出队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> colorQueue;

    TPipe pipe;
};

extern "C" __global__ __aicore__ void render(GM_ADDR rays, GM_ADDR colors) {
    KernelRender op;

    op.Init(WIDTH, HEIGHT, SAMPLES, rays, colors);
    op.Process();
    op.Release();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void render_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *rays, uint8_t *colors) {
    render<<<blockDim, l2ctrl, stream>>>(rays, colors);
}
#endif
