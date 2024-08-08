
#include "common.h"
#include "data_utils.h"



#ifndef __CCE_KT_TEST__
#include <acl/acl.h>
extern void render_do(uint32_t coreDim, void *l2ctrl, void *stream,
                      uint8_t *rays, uint8_t *spheres,uint8_t *colors);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void render(GM_ADDR rays,GM_ADDR spheres, GM_ADDR colors);
#endif

int main() {

    uint32_t blockDim = 8;
    uint32_t elementNums = WIDTH * HEIGHT * 4 * SAMPLES;


#ifdef __CCE_KT_TEST__

    size_t inputRayByteSize = elementNums * sizeof(uint32_t) * 6; // TODO: 更新为不同的数据类型Float，Half等
    size_t inputSphereByteSize = SPHERE_NUM * sizeof(uint32_t) * 10; // r^2 xyz color emssion
    size_t outputColorByteSize = elementNums * sizeof(uint32_t) * 3;

    uint8_t *rays = (uint8_t *)AscendC::GmAlloc(inputRayByteSize);
    uint8_t *spheres = (uint8_t *)AscendC::GmAlloc(inputSphereByteSize);
    uint8_t *colors = (uint8_t *)AscendC::GmAlloc(outputColorByteSize);

    // copy data_gen
    ReadFile("./input/rays.bin", inputRayByteSize, rays, inputRayByteSize);
    ReadFile("./input/spheres.bin", inputSphereByteSize, spheres, inputSphereByteSize);
    
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(render, blockDim, rays,spheres, colors);

    // copy result
    WriteFile("./output/color.bin", colors, outputColorByteSize);

    AscendC::GmFree((void *)rays);
    AscendC::GmFree((void *)spheres);
    AscendC::GmFree((void *)colors);

#else
    size_t inputRayByteSize = elementNums * sizeof(uint32_t) * 6; // rays
    size_t inputSphereByteSize = SPHERE_NUM * sizeof(uint32_t) * 10; // r^2 xyz color emssion
    size_t outputColorByteSize = elementNums * sizeof(uint32_t) * 3;

    aclrtContext context;
    int32_t deviceId = 0;

    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *rayHost,*sphereHost, *colorHost;
    uint8_t *rayDevice,*sphereDevice, *colorDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&rayHost), inputRayByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&sphereHost), inputSphereByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&colorHost), outputColorByteSize));

    CHECK_ACL(aclrtMalloc((void **)&rayDevice, inputRayByteSize,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    CHCEK_ACL(aclrtMalloc((void **)&sphereDevice, inputSphereByteSize,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&colorDevice, outputColorByteSize,
                          ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./input/rays.bin", inputRayByteSize, rayHost, inputRayByteSize);
    CHECK_ACL(aclrtMemcpy(rayDevice, inputRayByteSize, rayHost, inputRayByteSize,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    ReadFile("./input/spheres.bin", inputSphereByteSize, rayHost, inputSphereByteSize);
    CHECK_ACL(aclrtMemcpy(rayDevice, inputSphereByteSize, rayHost, inputSphereByteSize,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    render_do(blockDim, nullptr, stream, rayDevice, colorDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(colorHost, outputColorByteSize, colorDevice,
                          outputColorByteSize, ACL_MEMCPY_DEVICE_TO_HOST));

    WriteFile("./output/color.bin", colorHost, outputColorByteSize);

    CHECK_ACL(aclrtFree(rayDevice));
    CHECK_ACL(aclrtFree(sphereDevice));
    CHECK_ACL(aclrtFree(colorDevice));

    CHECK_ACL(aclrtFreeHost(rayHost));
    CHECK_ACL(aclrtFreeHost(sphereHost));
    CHECK_ACL(aclrtFreeHost(colorHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

#endif
}