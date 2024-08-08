#pragma once
#include "common.h"
#include "kernel_operator.h"

using Float = float;

struct RaySoA {
    AscendC::GlobalTensor<Float> ox, oy, oz;
    AscendC::GlobalTensor<Float> dx, dy, dz;
    __aicore__ inline RaySoA() {}
};

struct RayLocalSoA {
    AscendC::LocalTensor<Float> ox, oy, oz;
    AscendC::LocalTensor<Float> dx, dy, dz;
    __aicore__ inline RayLocalSoA() {}
};

struct SphereLocalSoA {
    AscendC::LocalTensor<Float> r2,x, y, z;
    AscendC::LocalTensor<Float> emissionX, emissionY, emissionZ;
    AscendC::LocalTensor<Float> colorX, colorY, colorZ;
    __aicore__ inline SphereLocalSoA() {}
};

struct VecSoA {
    AscendC::GlobalTensor<Float> x, y, z;
    __aicore__ inline VecSoA() {}
};

struct VecLocalSoA {
    AscendC::LocalTensor<Float> x, y, z;
    __aicore__ inline VecLocalSoA() {}
};


// 配置指向全局内存的指针
__aicore__ inline void InitRaySoA(RaySoA &ray, GM_ADDR r, int block_offset, int block_length) {
    int32_t ray_count = WIDTH * HEIGHT * SAMPLES * 4;
    int32_t ray_offset = ray_count;

    ray.ox.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 0 + block_offset, block_length);
    ray.oy.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 1 + block_offset, block_length);
    ray.oz.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 2 + block_offset, block_length);
    ray.dx.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 3 + block_offset, block_length);
    ray.dy.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 4 + block_offset, block_length);
    ray.dz.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 5 + block_offset, block_length);
}

__aicore__ inline void InitColorSoA(VecSoA &color, GM_ADDR output, int block_offset, int block_length) {
    int32_t color_count = WIDTH * HEIGHT * SAMPLES * 4;
    int32_t color_offset = color_count;

    color.x.SetGlobalBuffer((__gm__ Float *)output + color_offset * 0 + block_offset, block_length);
    color.y.SetGlobalBuffer((__gm__ Float *)output + color_offset * 1 + block_offset, block_length);
    color.z.SetGlobalBuffer((__gm__ Float *)output + color_offset * 2 + block_offset, block_length);
}
