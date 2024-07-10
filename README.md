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
  - Ascend310P4 (Atlas 200 I DK)

```bash
  bash run.sh -r [RUN_MODE] -v  [SOC_VERSION] 
```
