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