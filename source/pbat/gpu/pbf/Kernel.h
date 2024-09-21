// Kernel.h
#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>

class Kernel
{
public:
    explicit Kernel(float h);

    __device__ float poly6(const float3& pi, const float3& pj) const;
    __device__ float3 spiky(const float3& pi, const float3& pj) const;
    __device__ float scorr(const float3& pi, const float3& pj) const;

private:
    float h, h2, h3, h9, deltaqSq;
    float kPoly, kSpiky;
};

#endif // KERNEL_H
