#ifndef PBF_KERNELS_CUH
#define PBF_KERNELS_CUH

#include <cuda_runtime.h>
#include "Kernel.h"

namespace PbfKernels
{

__global__ void updatePredictedPositions(
    const float3* positions,
    const float3* velocities,
    float3* predictedPositions,
    const float3 externalForce,
    const float dt,
    const size_t numParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles)
    {
        predictedPositions[i] = positions[i] + velocities[i] * dt + dt * dt * externalForce;
    }
}

__global__ void computeDensityAndLambda(
    const float3* predictedPositions,
    float* densities,
    float* lambdas,
    const float rho0,
    const float eps,
    const size_t numParticles,
    const Kernel kernel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles)
    {
        float rho = 0.0f;
        // For simplicity, consider all particles as neighbors
        for (int j = 0; j < numParticles; ++j)
        {
            rho += kernel.poly6(predictedPositions[i], predictedPositions[j]);
        }
        densities[i] = rho;
        float C = rho / rho0 - 1.0f;

        float sumGradSquared = 0.0f;
        // Compute sum of gradient magnitudes squared
        for (int j = 0; j < numParticles; ++j)
        {
            float3 grad = kernel.spiky(predictedPositions[i], predictedPositions[j]) / rho0;
            sumGradSquared += dot(grad, grad);
        }
        lambdas[i] = -C / (sumGradSquared + eps);
    }
}

__global__ void computeDeltaP(
    const float3* predictedPositions,
    const float* lambdas,
    float3* deltaP,
    const float rho0,
    const float kCorr,
    const size_t numParticles,
    const Kernel kernel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles)
    {
        float3 delta = make_float3(0.0f, 0.0f, 0.0f);
        for (int j = 0; j < numParticles; ++j)
        {
            float scorr = -kCorr * pow(kernel.poly6(predictedPositions[i], predictedPositions[j]) / kernel.scorr(predictedPositions[i], predictedPositions[j]), 4);
            delta += (lambdas[i] + lambdas[j] + scorr) * kernel.spiky(predictedPositions[i], predictedPositions[j]);
        }
        deltaP[i] = delta / rho0;
        // Update predicted positions
        // Note: This can be combined with the deltaP computation to save memory
    }
}

__global__ void updatePositionsAndVelocities(
    float3* positions,
    float3* velocities,
    const float3* predictedPositions,
    const float dt,
    const size_t numParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles)
    {
        velocities[i] = (predictedPositions[i] - positions[i]) / dt;
        positions[i] = predictedPositions[i];
    }
}

__global__ void boxCollision(
    float3* predictedPositions,
    const float3 min,
    const float3 max,
    const size_t numParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles)
    {
        predictedPositions[i].x = fminf(fmaxf(predictedPositions[i].x, min.x), max.x);
        predictedPositions[i].y = fminf(fmaxf(predictedPositions[i].y, min.y), max.y);
        predictedPositions[i].z = fminf(fmaxf(predictedPositions[i].z, min.z), max.z);
    }
}

} // namespace PbfKernels

#endif // PBF_KERNELS_CUH
