#include "rt_helper.h"
#include <cstdint>
using namespace AscendC;
#include <cstdlib>
using namespace std;

constexpr int32_t TOTAL_NUM = WIDTH * HEIGHT * SAMPLES * 4;
constexpr int32_t USE_CORE_NUM = 8;                                       // device core num
constexpr int32_t BLOCK_LENGTH = TOTAL_NUM / USE_CORE_NUM;                // 每个block处理的数据量(非字节数)
constexpr int32_t TILING_NUM = 8;                                         // custom config
constexpr int32_t BUFFER_NUM = 2;                                         // fix double buffer -> pipeline
constexpr int32_t TILING_LENGTH = BLOCK_LENGTH / TILING_NUM / BUFFER_NUM; // 真正每次处理的数据数量(非字节数)

class KernelRender {

  public:
    __aicore__ inline KernelRender() {}
    __aicore__ inline void Init(int w, int h, int s, GM_ADDR r, GM_ADDR output) {

        width = w;
        height = h;
        samples = s;

        InitRaySoA(r);
        InitColorSoA(output);

        pipe.InitBuffer(rayQueue, BUFFER_NUM, TILING_LENGTH * sizeof(Float) * 6);   // ray xyz dxdydz = 6
        pipe.InitBuffer(colorQueue, BUFFER_NUM, TILING_LENGTH * sizeof(Float) * 3); // color xyz = 3

        // printf("rayQueue Length: %d\n", TILING_LENGTH * 6);
    }

    __aicore__ inline void InitRaySoA(GM_ADDR r) {
        int32_t block_offset = BLOCK_LENGTH * GetBlockIdx();

        int32_t ray_count = WIDTH * HEIGHT * SAMPLES * 4;
        int32_t ray_offset = ray_count;

        inputRays_ox.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 0 + block_offset, BLOCK_LENGTH);
        inputRays_oy.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 1 + block_offset, BLOCK_LENGTH);
        inputRays_oz.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 2 + block_offset, BLOCK_LENGTH);
        inputRays_dx.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 3 + block_offset, BLOCK_LENGTH);
        inputRays_dy.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 4 + block_offset, BLOCK_LENGTH);
        inputRays_dz.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 5 + block_offset, BLOCK_LENGTH);
    }

    __aicore__ inline void InitColorSoA(GM_ADDR output) {
        int32_t block_offset = BLOCK_LENGTH * GetBlockIdx();
        int32_t color_count = WIDTH * HEIGHT * SAMPLES * 4;
        int32_t color_offset = color_count;

        resultColor_x.SetGlobalBuffer((__gm__ Float *)output + color_offset * 0 + block_offset, BLOCK_LENGTH);
        resultColor_y.SetGlobalBuffer((__gm__ Float *)output + color_offset * 1 + block_offset, BLOCK_LENGTH);
        resultColor_z.SetGlobalBuffer((__gm__ Float *)output + color_offset * 2 + block_offset, BLOCK_LENGTH);
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
    // system mem -> device memory
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<Float> ray = rayQueue.AllocTensor<Float>();

        // printf("ray length: %d\n", ray.GetLength());

        // offset
        int32_t r_x = TILING_LENGTH * 0;
        int32_t r_y = TILING_LENGTH * 1;
        int32_t r_z = TILING_LENGTH * 2;
        int32_t r_dx = TILING_LENGTH * 3;
        int32_t r_dy = TILING_LENGTH * 4;
        int32_t r_dz = TILING_LENGTH * 5;

        // printf("r_x: %d\n", r_x);
        // printf("r_y: %d\n", r_y);
        // printf("r_z: %d\n", r_z);
        // printf("r_dx: %d\n", r_dx);
        // printf("r_dy: %d\n", r_dy);
        // printf("r_dz: %d\n", r_dz);

        // printf("progress: %d\n", progress);

        // Print the address of the destination buffer
        // printf("Dst Addr: ray[r_x]: %ld\n", ray[r_x].GetLocalBufferAddr());
        // printf("Dst Addr: ray[r_y]: %ld\n", ray[r_y].GetLocalBufferAddr());
        // printf("Dst Addr: ray[r_z]: %ld\n", ray[r_z].GetLocalBufferAddr());
        // printf("Dst Addr: ray[r_dx]: %ld\n", ray[r_dx].GetLocalBufferAddr());
        // printf("Dst Addr: ray[r_dy]: %ld\n", ray[r_dy].GetLocalBufferAddr());
        // printf("Dst Addr: ray[r_dz]: %ld\n", ray[r_dz].GetLocalBufferAddr());

        // Print the src address
        // printf("current progress index: %d\n", progress * TILING_LENGTH);
        // printf("Src Addr: inputRays.ox: %p\n", inputRays.ox[progress * TILING_LENGTH].GetPhyAddr());
        // printf("Src Addr: inputRays.oy: %p\n", inputRays.oy[progress * TILING_LENGTH].GetPhyAddr());
        // printf("Src Addr: inputRays.oz: %p\n", inputRays.oz[progress * TILING_LENGTH].GetPhyAddr());
        // printf("Src Addr: inputRays.dx: %p\n", inputRays.dx[progress * TILING_LENGTH].GetPhyAddr());
        // printf("Src Addr: inputRays.dy: %p\n", inputRays.dy[progress * TILING_LENGTH].GetPhyAddr());
        // printf("Src Addr: inputRays.dz: %p\n", inputRays.dz[progress * TILING_LENGTH].GetPhyAddr());

        DataCopy(ray[r_x], inputRays_ox[progress * TILING_LENGTH], TILING_LENGTH);
        DataCopy(ray[r_y], inputRays_oy[progress * TILING_LENGTH], TILING_LENGTH);
        DataCopy(ray[r_z], inputRays_oz[progress * TILING_LENGTH], TILING_LENGTH);
        DataCopy(ray[r_dx], inputRays_dx[progress * TILING_LENGTH], TILING_LENGTH);
        DataCopy(ray[r_dy], inputRays_dy[progress * TILING_LENGTH], TILING_LENGTH);
        DataCopy(ray[r_dz], inputRays_dz[progress * TILING_LENGTH], TILING_LENGTH);

        rayQueue.EnQue(ray);
    }

    // read device memory & compute & output to device queue & all samples
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

        DataCopy(resultColor_x[progress * TILING_LENGTH], color[c_x], TILING_LENGTH);
        DataCopy(resultColor_y[progress * TILING_LENGTH], color[c_y], TILING_LENGTH);
        DataCopy(resultColor_z[progress * TILING_LENGTH], color[c_z], TILING_LENGTH);

        colorQueue.FreeTensor(color);
    }

  private:
    int width;
    int height;
    int samples;

    // global
    // [[deprecated("Can't construct a struct contain LocalTensor or Global Tensor")]]
    // RaySoA inputRays;
    GlobalTensor<Float> inputRays_ox, inputRays_oy, inputRays_oz, inputRays_dx, inputRays_dy, inputRays_dz;
    // [[deprecated("Can't construct a struct contain LocalTensor or Global Tensor")]]
    // VecSoA resultColor;
    GlobalTensor<Float> resultColor_x, resultColor_y, resultColor_z;

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

// 配置指向全局内存的指针
