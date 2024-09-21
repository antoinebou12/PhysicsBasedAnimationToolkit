// Kernel.cuh
#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "Kernel.h"

__device__ inline float Kernel::poly6(const float3& pi, const float3& pj) const
{
    const float3 r = pi - pj;
    const float r2 = dot(r, r);
    if (r2 < h2)
    {
        const float x = h2 - r2;
        return kPoly * x * x * x;
    }
    return 0.0f;
}

__device__ inline float3 Kernel::spiky(const float3& pi, const float3& pj) const
{
    const float3 r = pi - pj;
    const float rl = length(r);

    if (rl > 1e-6f && rl < h)
    {
        const float x = h - rl;
        return -((kSpiky * x * x) / rl) * r;
    }

    return make_float3(0.0f, 0.0f, 0.0f);
}

__device__ inline float Kernel::scorr(const float3& pi, const float3& pj) const
{
    const float3 r = pi - pj;
    const float r2 = dot(r, r);
    const float x = h2 - r2;
    return kPoly * x * x * x;
}

#endif // KERNEL_CUH
