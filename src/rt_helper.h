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

/*
 * @brief 计算存储空间并上取整到256B
 */
__aicore__ inline uint32_t Round256(uint32_t length) { return (length + 255) & (~255); }

/*
 * @brief 计算存储空间并上取整到32B
 */
__aicore__ inline uint32_t Round32(uint32_t length) { return (length + 31) & (~31); }

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
    AscendC::LocalTensor<Float> base;
    AscendC::LocalTensor<Float> r2, x, y, z;
    AscendC::LocalTensor<Float> emissionX, emissionY, emissionZ;
    AscendC::LocalTensor<Float> colorX, colorY, colorZ;
    __aicore__ inline SphereLocalSoA() {}
    __aicore__ inline void Init(AscendC::LocalTensor<Float> data) {
        base = data;
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
__aicore__ inline void SphereHitInfo(AscendC::LocalTensor<Float> &dst, Allocator &allocator, Sphere &sphere, RayLocalSoA &rays, int idx, int depth) {
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

    // DEBUG(
    //     if (depth == 1 && idx == 0) {
    //         printf("Debug::SphereHitInfo Depth: %d\n", depth);
    //         // CPUDumpTensor("ocX", ocX.Get(), GENERIC_SIZE);
    //         // CPUDumpTensor("ocY", ocY.Get(), GENERIC_SIZE);
    //         // CPUDumpTensor("ocZ", ocZ.Get(), GENERIC_SIZE);
    //         CPUDumpTensor("rays.dx", rays.dx, GENERIC_SIZE);
    //         CPUDumpTensor("rays.dy", rays.dy, GENERIC_SIZE);
    //         CPUDumpTensor("rays.dz", rays.dz, GENERIC_SIZE);
    //         CPUDumpTensor("b", b.Get(), GENERIC_SIZE);
    //     }
    // )

    // c =  ocX * ocX + ocY * ocY + ocZ * ocZ - sphere.r2
    auto c = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    Duplicate(c.Get(), Float(0), GENERIC_SIZE);
    MulAddDst(c.Get(), ocX.Get(), ocX.Get(), GENERIC_SIZE); // c += ocX * ocX
    MulAddDst(c.Get(), ocY.Get(), ocY.Get(), GENERIC_SIZE); // c += ocY * ocY
    MulAddDst(c.Get(), ocZ.Get(), ocZ.Get(), GENERIC_SIZE); // c += ocZ * ocZ
    Adds(c.Get(), c.Get(), -sphere.r2, GENERIC_SIZE);       // c = dot(oc, oc) - sphere.r2

    // DEBUG(if (depth == 1 && idx == 0) {
    //     printf("Debug::SphereHitInfo Depth: %d\n", depth);
    //     CPUDumpTensor("b", b.Get(), GENERIC_SIZE);
    //     CPUDumpTensor("c", c.Get(), GENERIC_SIZE);
    // })

    // disc = b^2 - c
    auto disc = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    Mul(disc.Get(), b.Get(), b.Get(), GENERIC_SIZE);    // disc = b * b
    Sub(disc.Get(), disc.Get(), c.Get(), GENERIC_SIZE); // disc = disc - c

    // DEBUG(
    //     if(depth==1&&idx==0){
    //         printf("Debug::SphereHitInfo Depth: %d\n",depth);
    //         CPUDumpTensor("disc", disc.Get(), GENERIC_SIZE);
    //     }
    // )

    auto discrSq = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    Sqrt(discrSq.Get(), disc.Get(), GENERIC_SIZE); // tmp2 = sqrt(tmp1) | 对于负数sqrt会返回nan

    auto t0 = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto t1 = AllocDecorator(allocator.Alloc(GENERIC_SIZE));

    Sub(t0.Get(), b.Get(), discrSq.Get(), GENERIC_SIZE); // t0 = b - discrSq | nan
    Add(t1.Get(), b.Get(), discrSq.Get(), GENERIC_SIZE); // t1 = b + discrSq | nan

    // DEBUG({
    //     if (depth == 1 && idx == 0) {
    //         printf("Debug::SphereHitInfo Depth: %d\n", depth);
    //         CPUDumpTensor("t0", t0.Get(), GENERIC_SIZE);
    //         CPUDumpTensor("t1", t1.Get(), GENERIC_SIZE);
    //     }
    // })

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
    auto indices = AllocDecorator(allocator.Alloc(SPHERE_NUM * GENERIC_SIZE));
    // ArithProgression(indices.Get().ReinterpretCast<int32_t>(), int32_t(0), int32_t(GENERIC_SIZE), SPHERE_NUM); // half/float/int16_t/int32_t

    for (int i = 0; i < GENERIC_SIZE * SPHERE_NUM; i++) {
        int32_t u = i / SPHERE_NUM; // row
        int32_t v = i % SPHERE_NUM; // col
        int32_t pos = v * GENERIC_SIZE + u;

        indices.Get().ReinterpretCast<uint32_t>().SetValue(i, pos * sizeof(Float));
        // DEBUG({
        //     if (i % 64 == 0) {
        //         printf("\n");
        //     }
        //     printf("%d ", pos);
        // })
    }

    AscendC::Gather(dst, src, indices.Get().ReinterpretCast<uint32_t>(), 0, GENERIC_SIZE * SPHERE_NUM);

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

    // DEBUG({
    //     printf("Debug::RecudeInfo sphereIndex\n");
    //     CPUDumpTensorU("sphereIndex", sphereIndex.Get().ReinterpretCast<uint32_t>(), GENERIC_SIZE * SPHERE_NUM);
    // })

    auto gatherMaskParam = GatherMaskParams{0, 8, 0, 8};
    uint64_t gatherCount = 0;
    GatherMask(minIndex.ReinterpretCast<uint32_t>(), sphereIndex.Get().ReinterpretCast<uint32_t>(), rawIndex.Get().ReinterpretCast<uint32_t>(), false,
               0, gatherMaskParam, gatherCount);

    // DEBUG({
    //     printf("Debug::RecudeInfo minIndex\n");
    //     CPUDumpTensorU("minIndex", minIndex.ReinterpretCast<uint32_t>(), GENERIC_SIZE);
    // })

    // printf("Gather GENERIC_SIZE %ld\n", gatherCount);
}

__aicore__ inline void ComputeHitInfo(AscendC::LocalTensor<Float> &minT, AscendC::LocalTensor<Float> &minIdx, RayLocalSoA &rays,
                                      SphereLocalSoA &spheres, Allocator &allocator, int depth) {
    using namespace AscendC;
    auto hitInfo = AllocDecorator(allocator.Alloc(GENERIC_SIZE * SPHERE_NUM));
    for (int i = 0; i < SPHERE_NUM; i++) {
        int offset = i * GENERIC_SIZE;
        // clang-format off
            auto cur_sphere =Sphere{
                .r2 = spheres.r2.GetValue(i), 
                .x = spheres.x.GetValue(i), 
                .y = spheres.y.GetValue(i),
                .z = spheres.z.GetValue(i)};
        // clang-format on
        auto dst = hitInfo.Get()[offset];
        SphereHitInfo(dst, allocator, cur_sphere, rays, i, depth);
        // DEBUG({
        //     if(depth==1 && i==0){
        //         printf("Debug::ComputeHitInfo Depth: %d\n", depth);
        //         CPUDumpTensor("dst", dst, SPHERE_NUM);
        //     }
        // })
    }

    // DEBUG({
    //     if (depth == 1) {
    //         printf("Debug::ComputeHitInfo Depth: %d\n", depth);
    //         CPUDumpTensor("hitInfo", hitInfo.Get(), GENERIC_SIZE * SPHERE_NUM);
    //     }
    // })

    auto transpose = AllocDecorator(allocator.Alloc(GENERIC_SIZE * SPHERE_NUM));

    // Step2: ray num of hit sphere -> sphere num of hit ray
    Transpose(transpose.Get(), hitInfo.Get(), allocator);

    // DEBUG({
    //     if (depth == 1) {
    //         CPUDumpTensor("transpose", transpose.Get(), GENERIC_SIZE * SPHERE_NUM);
    //     }
    // })

    // Step3: Compare & Get Min Index 8 float -> 32B = 1 block
    ReduceMinInfo(minIdx, minT, transpose.Get(), allocator);
}

__aicore__ inline void GenerateNewRays(RayLocalSoA &rays, AscendC::LocalTensor<Float> &hitIndex, AscendC::LocalTensor<Float> &hitMinT,
                                       SphereLocalSoA &spheres, Allocator &allocator, int depth) {
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
    // auto tmpZ1 = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    // auto tmpZ2 = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    // auto tmpZ3 = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    // auto tmpZ4 = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    // auto tmpSphereZ = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    // auto testIndex = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    // auto smallIndex = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    // auto tmpZ5 = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    // auto tmpZ6 = AllocDecorator(allocator.Alloc(GENERIC_SIZE));

    // for (int i = 0; i < GENERIC_SIZE; i++) {
    //     testIndex.Get().ReinterpretCast<uint32_t>().SetValue(i, uint32_t(i % SPHERE_NUM * sizeof(Float)));
    // }

    // for (int i = 0; i < SPHERE_NUM; i++) {
    //     tmpSphereZ.Get().SetValue(i, spheres.z.GetValue(i));
    // }

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

    Duplicate(sphereX.Get(), Float(-1), GENERIC_SIZE);
    Duplicate(sphereY.Get(), Float(-1), GENERIC_SIZE);
    Duplicate(sphereZ.Get(), Float(-1), GENERIC_SIZE);

    Gather(sphereX.Get(), spheres.x, srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), 0, GENERIC_SIZE);
    Gather(sphereY.Get(), spheres.y, srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), 0, GENERIC_SIZE);
    Gather(sphereZ.Get(), spheres.z, srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), 0, GENERIC_SIZE);

    // DEBUG({
    //     // CPUDumpTensorU("Index Raw", hitIndex.ReinterpretCast<int32_t>(), GENERIC_SIZE, true);
    //     // CPUDumpTensorU("offset", srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), GENERIC_SIZE, false);
    //     // CPUDumpTensorU("testIndex", testIndex.Get().ReinterpretCast<uint32_t>(), GENERIC_SIZE, false);
    //     // CPUDumpTensor("sphereX", sphereX.Get(), GENERIC_SIZE);
    //     // CPUDumpTensor("sphereY", sphereY.Get(), GENERIC_SIZE);
    //     // CPUDumpTensor("STD sphereZ", spheres.z, SPHERE_NUM);
    //     // CPUDumpTensor("tmpSphereZ", tmpSphereZ.Get(), SPHERE_NUM);
    //     CPUDumpTensor("cur output sphereZ", sphereZ.Get(), GENERIC_SIZE); // Excepted

    //     // CPUDumpTensor("cur output sphereZ", sphereZ.Get(), GENERIC_SIZE);
    //     // printf("targetIndex\n\t");
    //     // for (int j = 0; j < GENERIC_SIZE; j++) {
    //     //     auto val = hitIndex.ReinterpretCast<int32_t>().GetValue(j);
    //     //     printf("%d ", val);
    //     //     if (j % 8 == 7) {
    //     //         printf("\n\t");
    //     //     }
    //     // }
    //     // printf("\n");
    //     // printf("sphereZ by minVal index Directly(Error)\n\t");
    //     // for (int j = 0; j < GENERIC_SIZE; j++) {
    //     //     auto val = spheres.z.GetValue(hitIndex.ReinterpretCast<int32_t>().GetValue(j));
    //     //     printf("%5.3f ", val);
    //     //     if (j % 8 == 7) {
    //     //         printf("\n\t");
    //     //     }
    //     // }
    //     // printf("\n");
    //     // printf("sphereZ one by one(right)\n\t");
    //     // for (int j = 0; j < GENERIC_SIZE; j++) {
    //     //     auto val = spheres.z.GetValue(j % SPHERE_NUM); // correct
    //     //     printf("%5.3f ", val);
    //     //     if (j % 8 == 7) {
    //     //         printf("\n\t");
    //     //     }
    //     // }
    //     // printf("\n");
    //     // printf("sphereZ by convert index(Error)\n\t");

    //     // for (int j = 0; j < GENERIC_SIZE; j++) {
    //     //     auto val = spheres.z.GetValue(testIndex.Get().ReinterpretCast<uint32_t>().GetValue(j));
    //     //     printf("%5.3f ", val);
    //     //     if (j % 8 == 7) {
    //     //         printf("\n\t");
    //     //     }
    //     // }
    //     // printf("\n");
    // })
    // FIXME: 直接使用spheres.z会因为数据未知的原因导致无法读取部分数据(目前只知道和spheres.z有关),使用基于原始基地址的Base也无法解决问题

    // DEBUG({
    //     printf("Debug::GenerateNewRays\n");
    //     CPUDumpTensorU("src local offset", srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), GENERIC_SIZE);
    //     // printf("sphere base addr %ld\n", (int64_t)spheres.base.GetPhyAddr());
    //     // printf("sphere x addr %ld\n", (int64_t)spheres.x.GetPhyAddr());
    //     // printf("sphere y addr %ld\n", (int64_t)spheres.y.GetPhyAddr());
    //     // printf("sphere z addr %ld\n", (int64_t)spheres.z.GetPhyAddr());
    //     CPUDumpTensor("sphere z raw", spheres.z, 8);
    //     CPUDumpTensor("sphereX", sphereX.Get(), GENERIC_SIZE);
    //     CPUDumpTensor("sphereY", sphereY.Get(), GENERIC_SIZE);
    //     CPUDumpTensor("sphereZ", sphereZ.Get(), GENERIC_SIZE);
    // })

    auto normalX = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto normalY = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto normalZ = AllocDecorator(allocator.Alloc(GENERIC_SIZE));

    // normal = hitPos - spherePos
    Sub(normalX.Get(), hitPosX.Get(), sphereX.Get(), GENERIC_SIZE);
    Sub(normalY.Get(), hitPosY.Get(), sphereY.Get(), GENERIC_SIZE);
    Sub(normalZ.Get(), hitPosZ.Get(), sphereZ.Get(), GENERIC_SIZE);

    // normalize normal
    auto normalLen = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    Duplicate(normalLen.Get(), Float(0), GENERIC_SIZE);

    MulAddDst(normalLen.Get(), normalX.Get(), normalX.Get(), GENERIC_SIZE);
    MulAddDst(normalLen.Get(), normalY.Get(), normalY.Get(), GENERIC_SIZE);
    MulAddDst(normalLen.Get(), normalZ.Get(), normalZ.Get(), GENERIC_SIZE);

    // DEBUG({
    //     if(depth==0){
    //         printf("Debug::GenerateNewRays %d\n",depth);
    //         CPUDumpTensor("normalLen", normalLen.Get(), GENERIC_SIZE);
    //     }
    // })

    Sqrt(normalLen.Get(), normalLen.Get(), GENERIC_SIZE);

    auto normalizeX = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto normalizeY = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto normalizeZ = AllocDecorator(allocator.Alloc(GENERIC_SIZE));

    Div(normalizeX.Get(), normalX.Get(), normalLen.Get(), GENERIC_SIZE);
    Div(normalizeY.Get(), normalY.Get(), normalLen.Get(), GENERIC_SIZE);
    Div(normalizeZ.Get(), normalZ.Get(), normalLen.Get(), GENERIC_SIZE);

    // DEBUG({
    //     if(depth==0){
    //         printf("Debug::GenerateNewRays %d\n",depth);
    //         CPUDumpTensor("normalX", normalX.Get(), GENERIC_SIZE);
    //         CPUDumpTensor("normalY", normalY.Get(), GENERIC_SIZE);
    //         CPUDumpTensor("normalZ", normalZ.Get(), GENERIC_SIZE);

    //         CPUDumpTensor("normalLen", normalLen.Get(), GENERIC_SIZE);
    //         CPUDumpTensor("normalizeX", normalizeX.Get(), GENERIC_SIZE);
    //         CPUDumpTensor("normalizeY", normalizeY.Get(), GENERIC_SIZE);
    //         CPUDumpTensor("normalizeZ", normalizeZ.Get(), GENERIC_SIZE);
    //     }
    // })

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

    // 提取DiffuseColor
    auto diffuseX = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto diffuseY = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    auto diffuseZ = AllocDecorator(allocator.Alloc(GENERIC_SIZE));

    Duplicate(diffuseX.Get(), Float(-1), GENERIC_SIZE);
    Duplicate(diffuseY.Get(), Float(-1), GENERIC_SIZE);
    Duplicate(diffuseZ.Get(), Float(-1), GENERIC_SIZE);

    auto srcOffsetLocal = AllocDecorator(allocator.Alloc(GENERIC_SIZE));
    Muls(srcOffsetLocal.Get().ReinterpretCast<int32_t>(), hitIndex.ReinterpretCast<int32_t>(), int32_t(sizeof(Float)), GENERIC_SIZE);

    auto tmpDiffuseZ = AllocDecorator(allocator.Alloc(SPHERE_NUM));
    for (int i = 0; i < SPHERE_NUM; i++) {
        tmpDiffuseZ.Get().SetValue(i, spheres.colorZ.GetValue(i));
    }

    Gather(diffuseX.Get(), spheres.colorX, srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), 0, GENERIC_SIZE);
    Gather(diffuseY.Get(), spheres.colorY, srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), 0, GENERIC_SIZE);
    Gather(diffuseZ.Get(), tmpDiffuseZ.Get(), srcOffsetLocal.Get().ReinterpretCast<uint32_t>(), 0, GENERIC_SIZE);

    // DEBUG({
    //     printf("Debug::AccumulateIntervalColor\n");
    //     auto srcOffsetSize = srcOffsetLocal.Get().GetSize();
    //     CPUDumpTensor("diffuseX", diffuseX.Get(), GENERIC_SIZE);
    //     CPUDumpTensor("diffuseY", diffuseY.Get(), GENERIC_SIZE);
    //     CPUDumpTensor("diffuseZ", diffuseZ.Get(), GENERIC_SIZE);
    // })

    // 确定终止ray的mask
    // auto TerminateMask = AllocDecorator(allocator.Alloc(GENERIC_SIZE / 8));
    // CompareScalar(TerminateMask.Get().ReinterpretCast<uint8_t>(), hitIndex.ReinterpretCast<int32_t>(), int32_t(7), CMPMODE::EQ, GENERIC_SIZE); //
    // FIXME: 更新数据类型

    // // 合并mask
    // auto mask1 = TerminateMask.Get().ReinterpretCast<uint64_t>().GetValue(0);
    // auto mask2 =retMask.ReinterpretCast<uint64_t>().GetValue(0);
    // auto mask = mask1 & mask2;

    // DEBUG({
    //     printf("Debug::AccumulateIntervalColor\n");
    //     CPUDumpTensorU("TerminateMask", TerminateMask.Get().ReinterpretCast<uint8_t>(), GENERIC_SIZE / 8, true);
    //     CPUDumpTensorU("retMask", retMask.ReinterpretCast<uint8_t>(), GENERIC_SIZE / 8, true);
    //     printf("mask1 %ld mask2 %ld mask %ld\n", mask1, mask2, mask);
    // })

    // DEBUG({
    //     printf("Debug::AccumulateIntervalColor\n");
    //     CPUDumpTensor("ret all", ret.x, GENERIC_SIZE*3);
    //     CPUDumpTensor("ret.x", ret.x, GENERIC_SIZE);
    //     CPUDumpTensor("ret.y", ret.y, GENERIC_SIZE);
    //     CPUDumpTensor("ret.z", ret.z, GENERIC_SIZE);
    // })
  

    Mul(ret.x, diffuseX.Get(), ret.x, GENERIC_SIZE);
    Mul(ret.y, diffuseY.Get(), ret.y, GENERIC_SIZE);
    Mul(ret.z, diffuseZ.Get(), ret.z, GENERIC_SIZE);

    // Mul(testX.Get(),ret.x,diffuseX.Get(),GENERIC_SIZE);
    // Mul(testY.Get(),ret.y,diffuseY.Get(),GENERIC_SIZE);
    // Mul(testZ.Get(),ret.z,diffuseZ.Get(),GENERIC_SIZE);
    // DEBUG({
    //     printf("Debug::AccumulateIntervalColor\n");
    //     CPUDumpTensor("ret.x", ret.x, GENERIC_SIZE);
    //     CPUDumpTensor("ret.y", ret.y, GENERIC_SIZE);
    //     CPUDumpTensor("ret.z", ret.z, GENERIC_SIZE);
    //     // CPUDumpTensor("testX", testX.Get(), GENERIC_SIZE);
    //     // CPUDumpTensor("testY", testY.Get(), GENERIC_SIZE);
    //     // CPUDumpTensor("testZ", testZ.Get(), GENERIC_SIZE);

    // })
}