#pragma once
#include "allocator.h"
#include "common.h"
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
template <typename T> inline void CPUDumpTensorU(const char *name, const AscendC::LocalTensor<T> &tensor, int count, bool ismask = false) {
    printf("%s: \n\t", name);
    auto format_str = "%d ";
    if (ismask)
        format_str = "%08b ";
    for (int i = 0; i < count; i++) {
        if (i % 8 == 0 && i != 0)
            printf("\n\t");
        printf(format_str, tensor.GetValue(i));
    }
    printf("\n");
}

#define DEBUG(content)                                                                                                                               \
    if (AscendC::GetBlockIdx() == 0) {                                                                                                               \
        content                                                                                                                                      \
    }

#endif

template <typename T> inline std::uint32_t GetWorkspaceLength(int data_num, bool index = false) {
    // TODO: 请根据实际情况修改
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
__aicore__ inline void SphereHitInfo(AscendC::LocalTensor<Float> &dst, Allocator &allocator, Sphere &sphere, RayLocalSoA &rays) {
    using AscendC::LocalTensor;
    using namespace AscendC;

    auto ocX = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto ocY = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto ocZ = AllocDecorator(allocator.Alloc(GENERIC_SIZE));

    Adds(ocX.Get(), rays.ox, -sphere.x, GENERIC_SIZE); // ocX = spheres.x - rays.ox
    Muls(ocX.Get(), ocX.Get(), Float(-1), GENERIC_SIZE);
    Adds(ocY.Get(), rays.oy, -sphere.y, GENERIC_SIZE); // ocY = spheres.y - rays.oy
    Muls(ocY.Get(), ocY.Get(), Float(-1), GENERIC_SIZE);
    Adds(ocZ.Get(), rays.oz, -sphere.z, GENERIC_SIZE); // ocZ = spheres.z - rays.oz
    Muls(ocZ.Get(), ocZ.Get(), Float(-1), GENERIC_SIZE);

    // b = ocX * rays.ox + ocY * rays.oy + ocZ * rays.oz
    auto b = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    Duplicate(b.Get(), Float(0), GENERIC_SIZE);           // b = 0
    MulAddDst(b.Get(), ocX.Get(), rays.dx, GENERIC_SIZE); // b += ocX * rays.ox
    MulAddDst(b.Get(), ocY.Get(), rays.dy, GENERIC_SIZE); // b += ocY * rays.oy
    MulAddDst(b.Get(), ocZ.Get(), rays.dz, GENERIC_SIZE); // b += ocZ * rays.oz

    // c =  ocX * ocX + ocY * ocY + ocZ * ocZ - sphere.r2
    auto c = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    Duplicate(c.Get(), Float(0), GENERIC_SIZE);
    MulAddDst(c.Get(), ocX.Get(), ocX.Get(), GENERIC_SIZE); // c += ocX * ocX
    MulAddDst(c.Get(), ocY.Get(), ocY.Get(), GENERIC_SIZE); // c += ocY * ocY
    MulAddDst(c.Get(), ocZ.Get(), ocZ.Get(), GENERIC_SIZE); // c += ocZ * ocZ
    Adds(c.Get(), c.Get(), -sphere.r2, GENERIC_SIZE);       // c = dot(oc, oc) - sphere.r2

    // disc = b^2 - c
    auto disc = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    Mul(disc.Get(), b.Get(), b.Get(), GENERIC_SIZE);    // disc = b * b
    Sub(disc.Get(), disc.Get(), c.Get(), GENERIC_SIZE); // disc = disc - c

    auto discrSq = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    Sqrt(discrSq.Get(), disc.Get(), GENERIC_SIZE); // tmp2 = sqrt(tmp1) | 对于负数sqrt会返回nan

    auto t0 = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto t1 = AllocDecorator(allocator.Alloc(GENERIC_SIZE));

    Sub(t0.Get(), b.Get(), discrSq.Get(), GENERIC_SIZE); // t0 = b - discrSq | nan
    Add(t1.Get(), b.Get(), discrSq.Get(), GENERIC_SIZE); // t1 = b + discrSq | nan

    // dst = t0 > 0 ? t0 : t1
    auto t_mask = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    CompareScalar(t_mask.Get(), t0.Get(), Float(0), CMPMODE::GT, GENERIC_SIZE);                                               // mask2 = t0 > 0
    Select(dst, t_mask.Get().ReinterpretCast<uint8_t>(), t0.Get(), t1.Get(), SELMODE::VSEL_TENSOR_TENSOR_MODE, GENERIC_SIZE); // t = mask2 ? t0 : t1

    // set number less than 0 | nan to INF

    auto select_mask = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    CompareScalar(select_mask.Get(), dst, Float(0), CMPMODE::GT, GENERIC_SIZE);

    Select(dst, select_mask.Get().ReinterpretCast<uint8_t>(), dst, Float(1e20), SELMODE::VSEL_TENSOR_SCALAR_MODE,
           GENERIC_SIZE); // t = mask3 ? t : INF

    // DEBUG(
    //     printf("Debug::SphereHitInfo\n");
    //     CPUDumpTensor("dst", dst, GENERIC_SIZE);
    // )
}

__aicore__ inline void Transpose(AscendC::LocalTensor<Float> &dst, AscendC::LocalTensor<Float> &src, Allocator &allocator) {
    auto indices = AllocDecorator(allocator.Alloc(SPHERE_NUM));
    // ArithProgression(indices.Get().ReinterpretCast<int32_t>(), int32_t(0), int32_t(GENERIC_SIZE), SPHERE_NUM); // half/float/int16_t/int32_t
    for (int i = 0; i < SPHERE_NUM; i++) {
        indices.Get().ReinterpretCast<uint32_t>().SetValue(i, uint32_t(i * GENERIC_SIZE * sizeof(Float)));
    }

    for (int i = 0; i < GENERIC_SIZE; i++) {
        int offset = i * SPHERE_NUM;
        // srcoffset type: uint32_t
        Gather(dst[offset], src, indices.Get().ReinterpretCast<uint32_t>(), i * sizeof(Float), SPHERE_NUM);
    }

    // BUG: Ascend Copy API Bitmask 存在模板实例化的bug
}

__aicore__ inline void ReduceMinInfo(AscendC::LocalTensor<Float> &minIndex, AscendC::LocalTensor<Float> &minVal, AscendC::LocalTensor<Float> &src,
                                     Allocator &allocator) {
    using namespace AscendC;
    constexpr auto reduceRepeat = GENERIC_SIZE * SPHERE_NUM * sizeof(Float) / 256;
    // 64*8个datablock，每个datablock 32B，每个datablock 8个float 规约后变为-> 64个datablock | (对于AiCore每次处理8个datablock 256B)
    BlockReduceMin(minVal, src, reduceRepeat, 256 / sizeof(Float), 1, 1, 8);

    // Duplicate 8 datablock to 16datablock
    auto minResultPad = AllocDecorator(allocator.Alloc(GENERIC_SIZE * SPHERE_NUM));
    constexpr auto padRepeat = GENERIC_SIZE * SPHERE_NUM * sizeof(Float) / 256;
    Brcb(minResultPad.Get(), minVal, padRepeat, {1, 8});

    // TODO: Tips: 小端在前，大端在后(x86 CPU显示的数据正好和实际位置相反),实际8个bit中的位次从右往左
    auto rawIndex = AllocDecorator(allocator.Alloc(GENERIC_SIZE * SPHERE_NUM / 8));

    Compare(rawIndex.Get().ReinterpretCast<uint8_t>(), src, minResultPad.Get(), CMPMODE::EQ, GENERIC_SIZE * SPHERE_NUM);

    // DEBUG({
    //     printf("Debug::RecudeInfo rawIndex\n");
    //     // CPUDumpTensor("src", src, GENERIC_SIZE * SPHERE_NUM);
    //     // CPUDumpTensor("minResultPad", minResultPad.Get(), GENERIC_SIZE * SPHERE_NUM);
    //     CPUDumpTensor("rawIndex", rawIndex.Get().ReinterpretCast<uint8_t>(), GENERIC_SIZE * SPHERE_NUM / 8, true);
    // })

    // Collect 1bit index in each datablock

    auto sphereIndex = AllocDecorator(allocator.Alloc(GENERIC_SIZE * SPHERE_NUM));
    // 构造Sphere索引
    for (int i = 0; i < GENERIC_SIZE; i++) {
        auto offset = i * SPHERE_NUM;
        for (int j = 0; j < SPHERE_NUM; j++) {
            sphereIndex.Get().ReinterpretCast<uint32_t>().SetValue(offset + j, uint32_t(j));
        }
    }

    DEBUG({
        printf("Debug::RecudeInfo sphereIndex\n");
        CPUDumpTensorU("sphereIndex", sphereIndex.Get().ReinterpretCast<uint32_t>(), GENERIC_SIZE * SPHERE_NUM);
    })

    auto gatherMaskParam = GatherMaskParams{0, 8, 0, 8};
    uint64_t gatherCount = 0;
    GatherMask(minIndex.ReinterpretCast<uint32_t>(), sphereIndex.Get().ReinterpretCast<uint32_t>(), rawIndex.Get().ReinterpretCast<uint32_t>(), false,
               0, gatherMaskParam, gatherCount);

    DEBUG({
        printf("Debug::RecudeInfo minIndex\n");
        CPUDumpTensorU("minIndex", minIndex.ReinterpretCast<uint32_t>(), GENERIC_SIZE);
    })

    // printf("Gather GENERIC_SIZE %ld\n", gatherCount);
}

__aicore__ inline void GenerateNewRays(RayLocalSoA &rays, AscendC::LocalTensor<Float> &hitIndex, AscendC::LocalTensor<Float> &hitMinT,
                                       SphereLocalSoA &spheres, Allocator &allocator) {
    /// hitPos = rayPos + rayDir * hitMinT

    using namespace AscendC;
    auto hitPosX = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto hitPosY = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto hitPosZ = AllocDecorator(allocator.Alloc(GENERIC_SIZE));

    Mul(hitPosX.Get(), rays.dx, hitMinT, GENERIC_SIZE);
    Add(hitPosX.Get(), rays.ox, hitPosX.Get(), GENERIC_SIZE);
    Mul(hitPosY.Get(), rays.dy, hitMinT, GENERIC_SIZE);
    Add(hitPosY.Get(), rays.oy, hitPosY.Get(), GENERIC_SIZE);
    Mul(hitPosZ.Get(), rays.dz, hitMinT, GENERIC_SIZE);
    Add(hitPosZ.Get(), rays.oz, hitPosZ.Get(), GENERIC_SIZE);

    // gather sphere xyz by hitIndex
    auto sphereX = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto sphereY = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto sphereZ = AllocDecorator(allocator.Alloc(GENERIC_SIZE));

    // Muls 不支持uint32_t类型的dst和src | 支持 half/float/int16_t/int32_t
    // auto srcOffsetLocal = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    // Muls(srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), hitIndex.ReinterpretCast<uint32_t>(), uint32_t(sizeof(Float)), GENERIC_SIZE);

    // 暂不支持uint32_t 转化为int32_t
    // auto srcOffsetLocal = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    // Cast<int32_t>(srcOffsetLocal.Get().ReinterpretCast<int32_t>(), hitIndex.ReinterpretCast<uint32_t>(),RoundMode::CAST_NONE, GENERIC_SIZE);
    // Muls(srcOffsetLocal.Get().ReinterpretCast<int32_t>(), srcOffsetLocal.Get().ReinterpretCast<int32_t>(), int32_t(sizeof(Float)), GENERIC_SIZE);
    // Cast<uint32_t>(srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), srcOffsetLocal.Get().ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,
    // GENERIC_SIZE);

    auto srcOffsetLocal = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    Muls(srcOffsetLocal.Get().ReinterpretCast<int32_t>(), hitIndex.ReinterpretCast<int32_t>(), int32_t(sizeof(Float)), GENERIC_SIZE);

    DEBUG({
        printf("Debug::GenerateNewRays srcOffsetLocal\n");
        CPUDumpTensorU("srcOffsetLocal", srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), GENERIC_SIZE);
    })

    Duplicate(sphereX.Get(), Float(0), GENERIC_SIZE);
    Duplicate(sphereY.Get(), Float(0), GENERIC_SIZE);
    Duplicate(sphereZ.Get(), Float(0), GENERIC_SIZE);

    Gather(sphereZ.Get(), spheres.z, srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), 0, GENERIC_SIZE);
    Gather(sphereX.Get(), spheres.x, srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), 0, GENERIC_SIZE);
    Gather(sphereY.Get(), spheres.y, srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), 0, GENERIC_SIZE);

    DEBUG({
        printf("Debug::GenerateNewRays\n");
        CPUDumpTensorU("src local offset", srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), GENERIC_SIZE);
        CPUDumpTensor("sphere z raw", spheres.z, 8);
        CPUDumpTensor("sphereZ", sphereZ.Get(), GENERIC_SIZE);
    })

    auto normalX = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto normalY = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto normalZ = AllocDecorator(allocator.Alloc(GENERIC_SIZE));

    // normal = hitPos - spherePos
    Sub(normalX.Get(), hitPosX.Get(), sphereX.Get(), GENERIC_SIZE);
    Sub(normalY.Get(), hitPosY.Get(), sphereY.Get(), GENERIC_SIZE);
    Sub(normalZ.Get(), hitPosZ.Get(), sphereZ.Get(), GENERIC_SIZE);

    // normalize normal
    auto normalLen = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto normalLenSq = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    Duplicate(normalLenSq.Get(), Float(0), GENERIC_SIZE);

    MulAddDst(normalLen.Get(), normalX.Get(), normalX.Get(), GENERIC_SIZE);
    MulAddDst(normalLen.Get(), normalY.Get(), normalY.Get(), GENERIC_SIZE);
    MulAddDst(normalLen.Get(), normalZ.Get(), normalZ.Get(), GENERIC_SIZE);
    Sqrt(normalLen.Get(), normalLen.Get(), GENERIC_SIZE);

    auto normalizeX = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto normalizeY = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto normalizeZ = AllocDecorator(allocator.Alloc(GENERIC_SIZE));

    Div(normalizeX.Get(), normalX.Get(), normalLen.Get(), GENERIC_SIZE);
    Div(normalizeY.Get(), normalY.Get(), normalLen.Get(), GENERIC_SIZE);
    Div(normalizeZ.Get(), normalZ.Get(), normalLen.Get(), GENERIC_SIZE);

    // compute new ray direction
    // rayDir = rayDir - 2 * dot(rayDir, normal) * normal
    // Q: 基于这个公式计算所得的rayDir是否需要归一化？
    // A: 不需要，因为这个公式是一个反射公式，反射后的方向是一个单位向量
    // Q: normal 是否需要归一化？
    // A: 需要，因为这个公式是一个反射公式，反射后的方向是一个单位向量

    auto dotValue = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    Duplicate(dotValue.Get(), Float(0), GENERIC_SIZE);
    MulAddDst(dotValue.Get(), rays.dx, normalizeX.Get(), GENERIC_SIZE);
    MulAddDst(dotValue.Get(), rays.dy, normalizeY.Get(), GENERIC_SIZE);
    MulAddDst(dotValue.Get(), rays.dz, normalizeZ.Get(), GENERIC_SIZE);
    Muls(dotValue.Get(), dotValue.Get(), Float(2), GENERIC_SIZE);
    Mul(normalX.Get(), normalizeX.Get(), dotValue.Get(), GENERIC_SIZE);
    Mul(normalY.Get(), normalizeY.Get(), dotValue.Get(), GENERIC_SIZE);
    Mul(normalZ.Get(), normalizeZ.Get(), dotValue.Get(), GENERIC_SIZE);
    Sub(rays.dx, rays.dx, normalX.Get(), GENERIC_SIZE);
    Sub(rays.dy, rays.dy, normalY.Get(), GENERIC_SIZE);
    Sub(rays.dz, rays.dz, normalZ.Get(), GENERIC_SIZE);

    // update ray | 64 elements
    Muls(rays.ox, hitPosX.Get(), Float(1), GENERIC_SIZE);
    Muls(rays.oy, hitPosY.Get(), Float(1), GENERIC_SIZE);
    Muls(rays.oz, hitPosZ.Get(), Float(1), GENERIC_SIZE);
}

__aicore__ inline void AccumulateIntervalColor(VecLocalSoA &ret, AscendC::LocalTensor<Float> &retMask, AscendC::LocalTensor<Float> &hitIndex,
                                               SphereLocalSoA &spheres, Allocator &allocator) {
    using namespace AscendC;

    // 确定终止ray的mask
    auto TerminateMask = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    CompareScalar(TerminateMask.Get(), hitIndex, Float(7), CMPMODE::EQ, GENERIC_SIZE); // FIXME: 更新数据类型

    // 合并mask
    // And(retMask, retMask, TerminateMask.Get(), GENERIC_SIZE/8); // TODO: 合并mask有bug

    // 累加颜色
    auto diffuseX = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto diffuseY = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto diffuseZ = AllocDecorator(allocator.Alloc(GENERIC_SIZE));

    DEBUG({
        auto diffuse = diffuseX.Get().GetSize();
        printf("Debug::AccumulateIntervalColor\n");
        printf("diffuseX size %d\n", diffuse);
    })

    auto srcOffsetLocal = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    Muls(srcOffsetLocal.Get().ReinterpretCast<int32_t>(), hitIndex.ReinterpretCast<int32_t>(), int32_t(sizeof(Float)), GENERIC_SIZE);

    Duplicate(diffuseX.Get(), Float(-1), GENERIC_SIZE);

    Gather(diffuseX.Get(), spheres.colorX, srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), 0, GENERIC_SIZE);
    Gather(diffuseY.Get(), spheres.colorY, srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), 0, GENERIC_SIZE);
    Gather(diffuseZ.Get(), spheres.colorZ, srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), 0, GENERIC_SIZE);

    DEBUG({
        printf("Debug::AccumulateIntervalColor\n");
        auto srcOffsetSize = srcOffsetLocal.Get().GetSize();
        printf("srcOffsetSize %d\n", srcOffsetSize);
        CPUDumpTensor("sphere ColorX", spheres.colorX, SPHERE_NUM);
        CPUDumpTensorU("srcOffsetLocal", srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), GENERIC_SIZE);
        CPUDumpTensor("diffuseX", diffuseX.Get(), GENERIC_SIZE);
    })

    Mul(ret.x, ret.x, diffuseX.Get(), GENERIC_SIZE); // FIXME: 禁止使用存在Emission的Sphere更新颜色
    Mul(ret.y, ret.y, diffuseY.Get(), GENERIC_SIZE);
    Mul(ret.z, ret.z, diffuseZ.Get(), GENERIC_SIZE);
}