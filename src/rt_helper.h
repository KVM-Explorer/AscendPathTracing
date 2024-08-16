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
    inline void Init(AscendC::LocalTensor<Float> data, int offset) {
        ox = data[offset * 0];
        oy = data[offset * 1];
        oz = data[offset * 2];
        dx = data[offset * 3];
        dy = data[offset * 4];
        dz = data[offset * 5];
    }
};

struct SphereLocalSoA {
    AscendC::LocalTensor<Float> r2, x, y, z;
    AscendC::LocalTensor<Float> emissionX, emissionY, emissionZ;
    AscendC::LocalTensor<Float> colorX, colorY, colorZ;
    __aicore__ inline SphereLocalSoA() {}
    inline void Init(AscendC::LocalTensor<Float> data) {
        r2 = data[SPHERE_NUM * 0];
        x = data[SPHERE_NUM * 1];
        y = data[SPHERE_NUM * 2];
        z = data[SPHERE_NUM * 3];
        emissionX = data[SPHERE_NUM * 4];
        emissionY = data[SPHERE_NUM * 5];
        emissionZ = data[SPHERE_NUM * 6];
        colorX = data[SPHERE_NUM * 7];
        colorY = data[SPHERE_NUM * 8];
        colorZ = data[SPHERE_NUM * 9];
    }
};

struct Sphere {
    Float r2, x, y, z;
};

struct VecSoA {
    AscendC::GlobalTensor<Float> x, y, z;
    __aicore__ inline VecSoA() {}
};

struct VecLocalSoA {
    AscendC::LocalTensor<Float> x, y, z;
    __aicore__ inline VecLocalSoA() {}
    inline void Init(AscendC::LocalTensor<Float> data, int offset) {
        x = data[offset * 0];
        y = data[offset * 1];
        z = data[offset * 2];
    }
};

template <typename T> inline void CPUDumpTensor(const char *name, const AscendC::LocalTensor<T> &tensor, int count, bool ismask = false) {
    printf("%s: \n\t", name);
    auto format_str = "%5.3f ";
    if (ismask)
        format_str = "%08b ";
    for (int i = 0; i < count; i++) {
        if (i % 8 == 0 && i != 0)
            printf("\n\t");
        printf(format_str, tensor.GetValue(i));
    }
    printf("\n");
}

// inline void CPUDumpTensor(const char *name, const AscendC::LocalTensor<Float> &tensor, int count) {
//     printf("%s: \n\t", name);
//     auto format_str = "%5.3f ";
//     for (int i = 0; i < count; i++) {
//         if(i%8==0&&i!=0) printf("\n\t");
//         printf(format_str, tensor.GetValue(i));
//     }
//     printf("\n");
// }

// inline void CPUDumpMask(const char *name, const AscendC::LocalTensor<uint8_t> &tensor, int count) {
//     printf("%s: \n\t", name);
//     auto format_str = "%b\n";
//     for (int i = 0; i < count; i++) {
//         if(i%8==0&&i!=0) printf("\n\t");
//         printf(format_str, tensor.GetValue(i));
//     }
//     printf("\n");
// }

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

/*
 * @brief 计算光线与球体的交点
 * @param dst 输出的交点对应的t值，即ray到交点的距离
 * @param sharedTmpBuffer 临时缓冲区（N倍的ray count| TILE_LENGTH）
 * @param spheres 球体信息
 * @param rays 光线信息
 * @param count 光线数量
 */
__aicore__ inline void SphereHitInfo(AscendC::LocalTensor<Float> &dst, AscendC::LocalTensor<Float> &sharedTmpBuffer, Sphere &sphere,
                                     RayLocalSoA &rays, int count) {
    using AscendC::LocalTensor;
    using namespace AscendC;
    LocalTensor<Float> ocX = sharedTmpBuffer[count * 0], ocY = sharedTmpBuffer[count * 1], ocZ = sharedTmpBuffer[count * 2];
    LocalTensor<Float> b = sharedTmpBuffer[count * 3], c = sharedTmpBuffer[count * 4];
    LocalTensor<Float> t0 = sharedTmpBuffer[count * 5], t1 = sharedTmpBuffer[count * 6];
    LocalTensor<uint8_t> disc_bitmask = sharedTmpBuffer[count * 7].ReinterpretCast<uint8_t>();

    Adds(ocX, rays.ox, -sphere.x, count); // ocX = spheres.x - rays.ox
    Muls(ocX, ocX, Float(-1), count);
    Adds(ocY, rays.oy, -sphere.y, count); // ocY = spheres.y - rays.oy
    Muls(ocY, ocY, Float(-1), count);
    Adds(ocZ, rays.oz, -sphere.z, count); // ocZ = spheres.z - rays.oz
    Muls(ocZ, ocZ, Float(-1), count);

    // b = ocX * rays.ox + ocY * rays.oy + ocZ * rays.oz
    Duplicate(b, Float(0), count);     // b = 0
    MulAddDst(b, ocX, rays.dx, count); // b += ocX * rays.ox
    MulAddDst(b, ocY, rays.dy, count); // b += ocY * rays.oy
    MulAddDst(b, ocZ, rays.dz, count); // b += ocZ * rays.oz

    // c =  ocX * ocX + ocY * ocY + ocZ * ocZ - sphere.r2
    Duplicate(c, Float(0), count);
    MulAddDst(c, ocX, ocX, count); // c += ocX * ocX
    MulAddDst(c, ocY, ocY, count); // c += ocY * ocY
    MulAddDst(c, ocZ, ocZ, count); // c += ocZ * ocZ
    Adds(c, c, -sphere.r2, count); // c = dot(oc, oc) - sphere.r2

    auto &disc = ocX;
    // disc = b^2 - c
    Mul(disc, b, b, count);    // disc = b * b
    Sub(disc, disc, c, count); // disc = disc - c

    // CompareScalar(disc_bitmask, disc, Float(0), CMPMODE::GT, count); // 存在交点
    // uint64_t bitmask = disc_bitmask.ReinterpretCast<uint64_t>().GetValue(0);

    // static int idx = 0;
    // if (GetBlockIdx() == 0 && idx == 0) {
    //     // CPUDumpTensor("dst", dst, count);
    //     CPUDumpTensor("bitmask", disc_bitmask, count / 8, true);
    //     idx++;
    // }

    auto &discrSq = ocY;
    Sqrt(discrSq, disc, count); // tmp2 = sqrt(tmp1) | 对于负数sqrt会返回nan

    Sub(t0, b, discrSq, count); // t0 = b - discrSq | nan
    Add(t1, b, discrSq, count); // t1 = b + discrSq | nan

    auto &t_bitmask = ocZ;
    CompareScalar(t_bitmask, t0, Float(0), CMPMODE::GT, count);                                         // mask2 = t0 > 0
    Select(dst, t_bitmask.ReinterpretCast<uint8_t>(), t0, t1, SELMODE::VSEL_TENSOR_TENSOR_MODE, count); // t = mask2 ? t0 : t1

    // set number less than 0 | nan to INF
    auto &select_bitmask = ocX;
    CompareScalar(select_bitmask, dst, Float(0), CMPMODE::GT, count);

    Select(dst, select_bitmask.ReinterpretCast<uint8_t>(), dst, Float(1e20), SELMODE::VSEL_TENSOR_SCALAR_MODE, count); // t = mask3 ? t : INF

    
}