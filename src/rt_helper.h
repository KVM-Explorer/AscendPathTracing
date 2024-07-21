#pragma once
#include "common.h"
#include "kernel_operator.h"

using Float = float;

struct Sphere {
    Float x, y, z;
    Float r;
    Float emission[3];
    Float color[3];
    int refl;
};

struct SphereSoA {
    Float *x, *y, *z, *r2;
    Float *emission, *color;
    int *refl;
};

struct RaySoA {
    AscendC::GlobalTensor<Float> ox, oy, oz;
    AscendC::GlobalTensor<Float> dx, dy, dz;
};

struct RayLocalSoA {
    AscendC::LocalTensor<Float> ox, oy, oz;
    AscendC::LocalTensor<Float> dx, dy, dz;
};



struct VecSoA {
    AscendC::GlobalTensor<Float> x, y, z;
};

struct VecLocalSoA {
    AscendC::LocalTensor<Float> x, y, z;
};

Sphere spheres[] = {
    {50, 81.6, 81.6, 1e5, {0, 0, 0}, {0.75, 0.25, 0.25}, 0},  // Left
    {50, 81.6, 81.6, 1e5, {0, 0, 0}, {0.25, 0.25, 0.75}, 0},  // Right
    {50, 1e5, 81.6, 1e5, {0, 0, 0}, {0.75, 0.75, 0.75}, 0},   // Bottom
    {50, 81.6, 1e5, 1e5, {0, 0, 0}, {0.75, 0.75, 0.75}, 0},   // Top
    {1e5, 81.6, 81.6, 1e5, {0, 0, 0}, {0.75, 0.75, 0.75}, 0}, // Back
    {50, 81.6, 1e5, 1e5, {0, 0, 0}, {0.75, 0.75, 0.75}, 0},   // Front
    {27, 16.5, 47, 16.5, {0, 0, 0}, {1, 1, 1}, 1},            // Mirror
    {73, 16.5, 78, 16.5, {0, 0, 0}, {1, 1, 1}, 1},            // Glass
};

constexpr int SpheresCount = sizeof(spheres) / sizeof(Sphere);

constexpr Float EPSILON = 1e-4;

inline SphereSoA convertSoA(const Sphere *spheres, int count) {
    Float *x = new Float[count];
    Float *y = new Float[count];
    Float *z = new Float[count];
    Float *r2 = new Float[count];
    Float *emission = new Float[count * 3];
    Float *color = new Float[count * 3];
    int *refl = new int[count];

    for (int i = 0; i < count; i++) {
        x[i] = spheres[i].x;
        y[i] = spheres[i].y;
        z[i] = spheres[i].z;
        r2[i] = spheres[i].r * spheres[i].r;
        emission[i * 3] = spheres[i].emission[0];
        emission[i * 3 + 1] = spheres[i].emission[1];
        emission[i * 3 + 2] = spheres[i].emission[2];
        color[i * 3] = spheres[i].color[0];
        color[i * 3 + 1] = spheres[i].color[1];
        color[i * 3 + 2] = spheres[i].color[2];
        refl[i] = spheres[i].refl;
    }

    return {x, y, z, r2, emission, color, refl};
}

void InitRaySoA(RaySoA &ray, GM_ADDR r, int block_offset, int block_length) {
    int32_t ray_count = WIDTH * HEIGHT * SAMPLES * 4;
    int32_t ray_offset = ray_count * sizeof(Float);

    ray.ox.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 0 + block_offset, block_length);
    ray.oy.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 1 + block_offset, block_length);
    ray.oz.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 2 + block_offset, block_length);
    ray.dx.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 3 + block_offset, block_length);
    ray.dy.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 4 + block_offset, block_length);
    ray.dz.SetGlobalBuffer((__gm__ Float *)r + ray_offset * 5 + block_offset, block_length);
}

void InitColorSoA(VecSoA &color, GM_ADDR output, int block_offset, int block_length) {
    int32_t color_count = WIDTH * HEIGHT * SAMPLES * 4;
    int32_t color_offset = color_count * sizeof(Float);

    color.x.SetGlobalBuffer((__gm__ Float *)output + color_offset * 0 + block_offset, block_length);
    color.y.SetGlobalBuffer((__gm__ Float *)output + color_offset * 1 + block_offset, block_length);
    color.z.SetGlobalBuffer((__gm__ Float *)output + color_offset * 2 + block_offset, block_length);
}
