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

struct SphereSoA {
    AscendC::GlobalTensor<Float> x, y, z, r2;
    AscendC::GlobalTensor<Float> emissionX, emissionY, emissionZ;
    AscendC::GlobalTensor<Float> colorX, colorY, colorZ;
    AscendC::GlobalTensor<int> refl;
    __aicore__ inline SphereSoA() {}
};

struct SphereLocalSoA {
    AscendC::LocalTensor<Float> x, y, z, r2;
    AscendC::LocalTensor<Float> emissionX, emissionY, emissionZ;
    AscendC::LocalTensor<Float> colorX, colorY, colorZ;
    AscendC::LocalTensor<int> refl;
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

constexpr Float EPSILON = 1e-4;

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

// Padding Empty Data
__aicore__ inline void InitSphereSoA(SphereSoA &spheres, GM_ADDR s, int block_offset, int block_length) {
    int32_t sphere_count = SPHERE_NUM;
    int32_t sphere_offset = sphere_count;

    spheres.x.SetGlobalBuffer((__gm__ Float *)s + sphere_offset * 0 + block_offset, block_length);
    spheres.y.SetGlobalBuffer((__gm__ Float *)s + sphere_offset * 1 + block_offset, block_length);
    spheres.z.SetGlobalBuffer((__gm__ Float *)s + sphere_offset * 2 + block_offset, block_length);
    spheres.r2.SetGlobalBuffer((__gm__ Float *)s + sphere_offset * 3 + block_offset, block_length);

    spheres.emissionX.SetGlobalBuffer((__gm__ Float *)s + sphere_offset * 4 + block_offset, block_length);
    spheres.emissionY.SetGlobalBuffer((__gm__ Float *)s + sphere_offset * 5 + block_offset + block_length, block_length);
    spheres.emissionZ.SetGlobalBuffer((__gm__ Float *)s + sphere_offset * 6 + block_offset + block_length, block_length);

    spheres.colorX.SetGlobalBuffer((__gm__ Float *)s + sphere_offset * 7 + block_offset, block_length);
    spheres.colorY.SetGlobalBuffer((__gm__ Float *)s + sphere_offset * 8 + block_offset, block_length);
    spheres.colorZ.SetGlobalBuffer((__gm__ Float *)s + sphere_offset * 9 + block_offset, block_length);

    spheres.refl.SetGlobalBuffer((__gm__ int *)s + sphere_offset * 10 + block_offset, block_length);
}