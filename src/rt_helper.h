#pragma once
#include "allocator.h"
#include "common.h"
#include "kernel_operator.h"


/*
    * @brief 获取规约操作的临时缓冲区长度
    * @param data_num 数据数量个数
    * @param index 是否需要索引
    * @return 临时缓冲区长度
    * @note UnifiedBuffer 满足datablock 32B对齐，同时无论是否需要索引，都需要分配内存
                          单次规约操作的最大字节数是256B, 8个datablock，生成一个val和index
                          不需要index的情况下，处理一次操作的数据并对齐到32B
                          否则在需要index可能存在递归规约的情况，此时需要保留全部的内存，并且每轮规约的数据需要对齐到32B

*/
template <typename T> inline std::uint32_t GetWorkspaceLength(int data_num, bool index = false) {
    // TODO:
    return -1;
}
struct RaySoA {
    AscendC::GlobalTensor<Float> ox, oy, oz;
    AscendC::GlobalTensor<Float> dx, dy, dz;
    __aicore__ inline RaySoA() {}
};

struct RayLocalSoA {
    AscendC::LocalTensor<Float> ox, oy, oz;
    AscendC::LocalTensor<Float> dx, dy, dz;
    __aicore__ inline RayLocalSoA() {}
    __aicore__ inline void Init(AscendC::LocalTensor<Float> data, int offset) {
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
    __aicore__ inline void Init(AscendC::LocalTensor<Float> data) {
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
    __aicore__ inline void Init(AscendC::LocalTensor<Float> data, int offset) {
        x = data[offset * 0];
        y = data[offset * 1];
        z = data[offset * 2];
    }
};

#ifdef __CCE_KT_TEST__
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
#endif

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
__aicore__ inline void SphereHitInfo(AscendC::LocalTensor<Float> &dst, Allocator &allocator, Sphere &sphere, RayLocalSoA &rays, int count) {
    using AscendC::LocalTensor;
    using namespace AscendC;

    auto ocX =  AllocDecorator( allocator.Alloc(count));
    auto ocY = AllocDecorator( allocator.Alloc(count));
    auto ocZ = AllocDecorator( allocator.Alloc(count));

    Adds(ocX.Get(), rays.ox, -sphere.x, count); // ocX = spheres.x - rays.ox
    Muls(ocX.Get(), ocX.Get(), Float(-1), count);
    Adds(ocY.Get(), rays.oy, -sphere.y, count); // ocY = spheres.y - rays.oy
    Muls(ocY.Get(), ocY.Get(), Float(-1), count);
    Adds(ocZ.Get(), rays.oz, -sphere.z, count); // ocZ = spheres.z - rays.oz
    Muls(ocZ.Get(), ocZ.Get(), Float(-1), count);

    // b = ocX * rays.ox + ocY * rays.oy + ocZ * rays.oz
    auto b = AllocDecorator( allocator.Alloc(count));
    Duplicate(b.Get(), Float(0), count);            // b = 0
    MulAddDst(b.Get(), ocX.Get(), rays.dx, count); // b += ocX * rays.ox
    MulAddDst(b.Get(), ocY.Get(), rays.dy, count); // b += ocY * rays.oy
    MulAddDst(b.Get(), ocZ.Get(), rays.dz, count); // b += ocZ * rays.oz

    // c =  ocX * ocX + ocY * ocY + ocZ * ocZ - sphere.r2
    auto c = AllocDecorator( allocator.Alloc(count));
    Duplicate(c.Get(), Float(0), count);
    MulAddDst(c.Get(), ocX.Get(), ocX.Get(), count); // c += ocX * ocX
    MulAddDst(c.Get(), ocY.Get(), ocY.Get(), count); // c += ocY * ocY
    MulAddDst(c.Get(), ocZ.Get(), ocZ.Get(), count); // c += ocZ * ocZ
    Adds(c.Get(), c.Get(), -sphere.r2, count);        // c = dot(oc, oc) - sphere.r2

    // disc = b^2 - c
    auto disc = AllocDecorator( allocator.Alloc(count));
    Mul(disc.Get(), b.Get(), b.Get(), count);    // disc = b * b
    Sub(disc.Get(), disc.Get(), c.Get(), count); // disc = disc - c

    auto discrSq = AllocDecorator( allocator.Alloc(count));
    Sqrt(discrSq.Get(), disc.Get(), count); // tmp2 = sqrt(tmp1) | 对于负数sqrt会返回nan

    auto t0 = AllocDecorator( allocator.Alloc(count));
    auto t1 = AllocDecorator( allocator.Alloc(count));

    Sub(t0.Get(), b.Get(), discrSq.Get(), count); // t0 = b - discrSq | nan
    Add(t1.Get(), b.Get(), discrSq.Get(), count); // t1 = b + discrSq | nan

    auto t_mask = AllocDecorator( allocator.Alloc(count));
    CompareScalar(t_mask.Get(), t0.Get(), Float(0), CMPMODE::GT, count);                                                // mask2 = t0 > 0
    Select(dst, t_mask.Get().ReinterpretCast<uint8_t>(), t0.Get(), t1.Get(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count); // t = mask2 ? t0 : t1

    // set number less than 0 | nan to INF
    auto select_mask = AllocDecorator( allocator.Alloc(count));
    CompareScalar(select_mask.Get(), dst, Float(0), CMPMODE::GT, count);

    // if (get_block_idx() == 0)
    //     printf("DEBUG:: select_mask %d\n", select_mask.GetId());
    Select(dst, select_mask.Get().ReinterpretCast<uint8_t>(), dst, Float(1e20), SELMODE::VSEL_TENSOR_SCALAR_MODE, count); // t = mask3 ? t : INF
}