# AscendPathTracing

## Intro

Path Tracing based on huawei ascend npu and written by c++ and ascend-c

## Tips&Lessons

NPU版本的算子无法支持常见系统库函数，CCE编译器缺少对应实现

## Demo运行

- RUN_MODE
  - cpu
  - npu
- SOC_VERSION
  - Ascend310P1
  - Ascend310B4 (Atlas 200 I DK)

```bash
  bash run.sh -r [RUN_MODE] -v  [SOC_VERSION] 
  bash run.sh -r cpu -v Ascend310P1
```

## 运行结果

![image](./demo.png)

## TODO

[-] 迁移到xmake,便于调试和运行
[x] 实现一个简单的MemoryPool和allocator负责管理分配临时内存

## 已知问题

Atlas 200I DK Compare API CPU模拟版本中没有解决nan和其他数值的比较的问题，按理应该返回false,实际场景如下比较ret>eps的结果，返回false则赋予1e20否则结果是ret
```bash
Debug::SphereHitInfo SphereId: 7, Depth: 0
select ret(t0,t1): 
          nan   nan   nan   nan   nan   nan   nan   nan 
          nan   nan   nan   nan   nan   nan   nan   nan 
          nan   nan   nan   nan   nan   nan   nan   nan 
          nan   nan   nan   nan   nan   nan   nan   nan 
          nan   nan   nan   nan   nan   nan   nan   nan 
          nan   nan   nan   nan   nan   nan   nan   nan 
          nan   nan   nan   nan   nan   nan 70.461 73.631 
        67.288 62.383 43.229 40.274 31.642 28.657 20.188 23.288 
compare mask: 
        00100111 10000111 00011100 01111000 00001100 01001111 11101101 00011000 
dst with 1e20: 
          nan   nan   nan 100000002004087734272.000 100000002004087734272.000   nan 100000002004087734272.000 100000002004087734272.000 
          nan   nan   nan 100000002004087734272.000 100000002004087734272.000 100000002004087734272.000 100000002004087734272.000   nan 
        100000002004087734272.000 100000002004087734272.000   nan   nan   nan 100000002004087734272.000 100000002004087734272.000 100000002004087734272.000 
        100000002004087734272.000 100000002004087734272.000 100000002004087734272.000   nan   nan   nan   nan 100000002004087734272.000 
        100000002004087734272.000 100000002004087734272.000   nan   nan 100000002004087734272.000 100000002004087734272.000 100000002004087734272.000 100000002004087734272.000 
          nan   nan   nan   nan 100000002004087734272.000 100000002004087734272.000   nan 100000002004087734272.000 
          nan 100000002004087734272.000   nan   nan 100000002004087734272.000   nan 70.461 73.631 
        100000002004087734272.000 100000002004087734272.000 100000002004087734272.000 40.274 31.642 100000002004087734272.000 100000002004087734272.000 100000002004087734272.000 
```