#pragma once
#include <stdint.h>

const int32_t WIDTH = 16;  // min 16 继续小可能涉及数据对齐问题，无法通过验证
const int32_t HEIGHT = 16; // min 16
const int32_t SAMPLES = 1; // SAMPLES * 4 = total samples

const float PI = 3.1415926535897932385f;
constexpr float EPSILON = 1e-4;
const int32_t SPHERE_NUM = 8;
const int32_t SPHERE_MEMBER_NUM = 10;
using Float = float;

const int32_t GENERIC_SIZE = 64;

