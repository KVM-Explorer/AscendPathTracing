#include "common.h"
#include "kernel_operator.h"
// #include <cmath>
#include <cstdint>
using namespace AscendC;
#include <cstdlib>
using namespace std;

using FF = float;

__aicore__ static FF randomFF() { return 0.5; }

__aicore__ static FF FFabs(FF x) {
    if (x > FF(0))
        return x;
    return FF(0) - x;
}

__aicore__ float InvSqrt(float x) {
    float xhalf = 0.5f * x;
    int i = *(int *)&x;             // 将浮点数转换为整数
    i = 0x5f3759df - (i >> 1);      // 初始猜测的位运算
    x = *(float *)&i;               // 将整数转换回浮点数
    x = x * (1.5f - xhalf * x * x); // Newton's method
    return x;
}

__aicore__ float sqrt32_approx(float value) { return 1.0f / InvSqrt(value); }
__aicore__ inline FF FFsqrt(FF x) { return sqrt32_approx(x); }

inline FF clamp(FF x) { return x < FF(0) ? FF(0) : x > FF(1) ? FF(1) : x; }

struct Vec {
    // LocalTensor<FF> data; //  x y z w
    FF x, y, z, w;

    __aicore__ bool operator<(const Vec &b) const {
        return x < b.x && y < b.y && z < b.z;
    }

    __aicore__ Vec(FF x_ = 0, FF y_ = 0, FF z_ = 0, FF w_ = 0)
        : x(x_), y(y_), z(z_), w(w_) {}

    __aicore__ Vec operator+(const Vec &v) const {
        return Vec(x + v.x, y + v.y, z + v.z);
    };
    __aicore__ Vec operator-(const Vec &v) const {
        return Vec(x - v.x, y - v.y, z - v.z);
    };
    __aicore__ Vec operator*(FF b) const { return Vec(x * b, y * b, z * b); };
    __aicore__ Vec mult(const Vec &v) const {
        return Vec(x * v.x, y * v.y, z * v.z);
    }
    __aicore__ Vec &norm() {
        return *this = *this * (1 / sqrt(x * x + y * y + z * z));
    }
    __aicore__ FF dot(const Vec &v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    __aicore__ Vec cross(const Vec &v) const {
        return Vec(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }
};

__aicore__ static Vec randomVec() {
    return Vec(randomFF(), randomFF(), randomFF());
}

struct Ray {
    Vec o, d;
    __aicore__ Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
    __aicore__ Ray(const Ray &data) : o(data.o), d(data.d) {}
};

struct Sphere {
    FF rad;
    Vec p, e, c;
    MaterialType m;
    __aicore__ Sphere(FF rad_, Vec p_, Vec e_, Vec c_, MaterialType m_)
        : rad(rad_), p(p_), e(e_), c(c_), m(m_) {}
    __aicore__ FF intersect(const Ray &r) const {
        Vec op = p - r.o;
        FF t, epsilon = 1e-4, b = op.dot(r.d),
              det = b * b - op.dot(op) + rad * rad;
        if (det < FF(0))
            return 0;
        else
            det = FFsqrt(det);
        return (t = b - det) > epsilon ? t
                                       : ((t = b + det) > epsilon ? t : FF(0));
    }
};

int32_t num_spheres = 9;
// 不允许分配静态数组
// static Sphere spheres[] = {
//     Sphere(1e5, Vec(1e5 + 1, 40.8, 81.6), Vec(), Vec(.75, .25, .25),
//            DIFF), // Left
//     Sphere(1e5, Vec(-1e5 + 99, 40.8, 81.6), Vec(), Vec(.25, .25, .75),
//            DIFF), // Right
//     Sphere(1e5, Vec(50, 40.8, 1e5), Vec(), Vec(.75, .75, .75), DIFF),
//     Sphere(1e5, Vec(50, 40.8, -1e5 + 170), Vec(), Vec(), DIFF),
//     Sphere(1e5, Vec(50, 1e5, 81.6), Vec(), Vec(.75, .75, .75), DIFF),
//     Sphere(1e5, Vec(50, -1e5 + 81.6, 81.6), Vec(), Vec(.75, .75, .75), DIFF),
//     Sphere(16.5, Vec(27, 16.5, 47), Vec(), Vec(1, 1, 1) * .999,
//            SPEC), // Mirror
//     Sphere(16.5, Vec(73, 16.5, 78), Vec(), Vec(1, 1, 1) * .999,
//            REFR), // Glass
//     Sphere(600, Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12), Vec(),
//            DIFF)}; // Light

/*
__aicore__ inline bool intersect(const Ray &r, float &t, int &id) {
    float n = num_spheres, d, inf = t = 1e20;
    for (int i = int(n); i--;)
        if ((d = spheres[i].intersect(r)) && d < t) {
            t = d;
            id = i;
        }
    return t < inf;
}

__aicore__ inline Vec radiance(const Ray &input_r, int input_depth) {

    float t;
    int id = 0;
    Ray r = input_r;
    int depth = input_depth;
    Vec cl(0, 0, 0); // accumulated color
    Vec cf(1, 1, 1); // accumulated reflectance

    while (1) {
        if (!intersect(r, t, id))
            return cl;                   // if miss, return black
        const Sphere &obj = spheres[id]; // the hit object
        Vec x = r.o + r.d * t, n = (x - obj.p).norm(),
            nl = n.dot(r.d) < FF(0) ? n : n * -1, f = obj.c;
        float p = f.x > f.y && f.x > f.z ? f.x
                  : f.y > f.z            ? f.y
                                         : f.z; // max refl
        cl = cl + cf.mult(obj.e);
        if (++depth > 5)

            if (randomFF() < p) {
                f = f * (1 / p);
            } else {
                return cl; // R.R.
            }
        cf = cf.mult(f);
        if (obj.m == DIFF) { // Ideal DIFFUSE reflection
            float r1 = FF(2 * PI) * randomFF(), r2 = randomFF(),
                  r2s = FFsqrt(r2);
            Vec w = nl,
                u = ((FFabs(w.x) > .1F ? Vec(0, 1) : Vec(1)).cross(w)).norm(),
                v = w.cross(u);
            Vec d =
                (u * cos(r1) * r2s + v * sin(r1) * r2s + w * FFsqrt(1.0F - r2))
                    .norm();
            // return obj.e + f.mult(radiance(Ray(x,d),depth,Xi));
            r = Ray(x, d);
            continue;
        } else if (obj.m == SPEC) { // Ideal SPECULAR reflection
            // return obj.e +
            // f.mult(radiance(Ray(x,r.d-n*2*n.dot(r.d)),depth,Xi));
            r = Ray(x, r.d - n * 2 * n.dot(r.d));
            continue;
        }
        Ray reflRay(x, r.d - n * 2 * n.dot(r.d)); // Ideal dielectric REFRACTION
        bool into = n.dot(nl) > FF(0);            // Ray fromoutside going in?
        int nc = 1;
        float nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl),
              cos2t;
        if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) <
            0) { // Total internal reflection
            // return obj.e + f.mult(radiance(reflRay,depth,Xi));
            r = reflRay;
            continue;
        }
        Vec tdir =
            (r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + FFsqrt(cos2t))))
                .norm();
        float a = nt - nc, b = nt + nc, R0 = a * a / (b * b),
              c = 1 - (into ? -ddn : tdir.dot(n));
        float Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re,
              P = .25f + .5f * Re, RP = Re / P, TP = Tr / (1 - P);
        // return obj.e + f.mult(erand48(Xi)<P ?
        //                       radiance(reflRay,    depth,Xi)*RP:
        //                       radiance(Ray(x,tdir),depth,Xi)*TP);
        if (randomFF() < P) {
            cf = cf * RP;
            r = reflRay;
        } else {
            cf = cf * TP;
            r = Ray(x, tdir);
        }
        continue;
    }
}

*/

constexpr int32_t TOTAL_NUM = WIDTH * HEIGHT * SAMPLES * 4;
constexpr int32_t USE_CORE_NUM = 8;
constexpr int32_t TILING_NUM = 8;
constexpr int32_t BUFFER_NUM = 4;

class KernelRender {

  public:
    __aicore__ inline KernelRender() {}
    __aicore__ inline void Init(int w, int h, int s, GM_ADDR input,
                                GM_ADDR output) {

        width = w;
        height = h;
        samples = s;

        constexpr int32_t block_length = TOTAL_NUM / USE_CORE_NUM;

        int32_t block_offset = block_length * GetBlockIdx();
        // 数组式访问，而非指针式访问
        inputRays.SetGlobalBuffer((__gm__ Ray *)input + block_offset,
                                  block_length);

        resultColor.SetGlobalBuffer((__gm__ Vec *)output + block_offset,
                                    block_length);

        constexpr int32_t tiling_length = TOTAL_NUM / USE_CORE_NUM / TILING_NUM;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tiling_length * sizeof(Ray));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tiling_length * sizeof(Vec));
    }

    __aicore__ inline void Process() {

        int loop_count = TILING_NUM * BUFFER_NUM;
        // constexpr int32_t ONCE_NUM =
        //     TOTAL_NUM / USE_CORE_NUM / TILING_NUM / BUFFER_NUM;
        for (int i = 0; i < loop_count; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

    __aicore__ inline void Release() {
        // free local tensor
    }

  private:
    // system mem -> device mem
    __aicore__ inline void CopyIn(int32_t progress) {
        constexpr int32_t tiling_length =
            TOTAL_NUM / USE_CORE_NUM / TILING_NUM / BUFFER_NUM;

        // generate rays
        LocalTensor<Ray> cur_ray = inQueueX.AllocTensor<Ray>();
        // DataCopy(cur_ray, inputRays[progress * TILE_LENGTH], TILE_LENGTH);
        DataCopy(cur_ray, inputRays[progress * tiling_length], tiling_length);
        inQueueX.EnQue(cur_ray);
    }

    // read device mem & compute & output to device queue & all samples
    __aicore__ inline void Compute(int32_t progress) {

        LocalTensor<Vec> ret = outQueueY.AllocTensor<Vec>();
        LocalTensor<Ray> ray = inQueueX.DeQue<Ray>();

        constexpr int32_t data_num =
            TOTAL_NUM / USE_CORE_NUM / TILING_NUM / BUFFER_NUM;
        // const int32_t data_offset = data_num * progress;

        Vec color(0.6, 0.7, 0.5);
        for (int i = 0; i < data_num; i++) {
            // auto cur_ray = ray.GetValue(i);
            auto cur1_ray = *(reinterpret_cast<Ray *>(ray.GetPhyAddr(i)));
            // auto test_ray = reinterpret_cast<Ray
            // (ray.GetLocalBufferAddr(i)); ray.GetLocalBufferAddr(i);
            // ray.GetBufferHandle();

            // // color = radiance(cur_ray, 0);            

            // ret.SetValue(i, color);
            *(reinterpret_cast<Vec *>(ret.GetPhyAddr(i))) = color;
            // auto ret_ptr = reinterpret_cast<Vec
            // *>(ret.GetLocalBufferAddr(i)); *ret_ptr = color;
        }

        inQueueX.FreeTensor(ray);
        outQueueY.EnQue<Vec>(ret);
    }

    // write device queue to system mem
    __aicore__ inline void CopyOut(int32_t progress) {
        // deque output tensor from VECOUT queue
        LocalTensor<Vec> color = outQueueY.DeQue<Vec>();
        // copy progress_th tile from local tensor to global tensor
        constexpr int32_t tiling_length =
            TOTAL_NUM / USE_CORE_NUM / TILING_NUM / BUFFER_NUM;

        DataCopy(resultColor[progress * tiling_length], color, tiling_length);
        // free output tensor for reuse
        outQueueY.FreeTensor(color);
    }

  private:
    int width;
    int height;
    int samples;
    // GlobalTensor<Sphere> spheres;
    // LocalTensor<Sphere> localSpheres;
    GlobalTensor<Vec> resultColor;
    GlobalTensor<Ray> inputRays;
    // 输入队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    // 输出队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    TPipe pipe;
};

extern "C" __global__ __aicore__ void render(GM_ADDR rays, GM_ADDR colors) {
    KernelRender op;

    op.Init(WIDTH, HEIGHT, SAMPLES, rays, colors);
    op.Process();
    op.Release();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void render_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *rays,
               uint8_t *colors) {
    render<<<blockDim, l2ctrl, stream>>>(rays, colors);
}
#endif
