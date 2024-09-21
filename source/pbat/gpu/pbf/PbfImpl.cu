#include "PbfImpl.cuh"
#include "PbfKernels.cuh"
#include "common/CUDAUtils.h"

Pbf::PbfImpl::PbfImpl(float radius_, float rho0_, float eps_, int maxIter_, float c_, float kCorr_)
    : radius(radius_), rho0(rho0_), eps(eps_), maxIter(maxIter_), c(c_), kCorr(kCorr_),
      hashGrid(make_float3(-2.0f, 0.0f, -1.0f), make_float3(2.0f, 2.0f, 1.0f), radius)
{
}

void Pbf::PbfImpl::setParticles(const std::vector<Particle>& particles)
{
    h_particles = particles;
    const size_t numParticles = h_particles.size();

    d_positions.resize(numParticles);
    d_velocities.resize(numParticles);
    d_predictedPositions.resize(numParticles);
    d_densities.resize(numParticles);
    d_lambdas.resize(numParticles);
    d_keys.resize(numParticles);

    thrust::host_vector<float3> h_positions(numParticles);
    thrust::host_vector<float3> h_velocities(numParticles);
    for (size_t i = 0; i < numParticles; ++i)
    {
        h_positions[i] = make_float3(h_particles[i].x.x(), h_particles[i].x.y(), h_particles[i].x.z());
        h_velocities[i] = make_float3(h_particles[i].v.x(), h_particles[i].v.y(), h_particles[i].v.z());
    }

    d_positions = h_positions;
    d_velocities = h_velocities;
}

const std::vector<Particle>& Pbf::PbfImpl::getParticles() const
{
    return h_particles;
}

void Pbf::PbfImpl::step(float dt)
{
    updatePredictedPositions(dt);
    updateHashGrid();
    buildNeighborhood();
    for (int iter = 0; iter < maxIter; ++iter)
    {
        computeDensityAndLambda();
        computeDeltaP();
        boxCollision();
    }
    updatePositionsAndVelocities(dt);
    applyViscosity();
    applyVorticityConfinement();
}

void Pbf::PbfImpl::updatePredictedPositions(float dt)
{
    const size_t numParticles = h_particles.size();
    const float3 externalForce = make_float3(0.0f, -9.8f, 0.0f);

    PbfKernels::updatePredictedPositions<<<(numParticles + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(d_positions.data()),
        thrust::raw_pointer_cast(d_velocities.data()),
        thrust::raw_pointer_cast(d_predictedPositions.data()),
        externalForce, dt, numParticles);
}

void Pbf::PbfImpl::updateHashGrid()
{
    // Implement hash grid update using CUDA if needed
}

void Pbf::PbfImpl::buildNeighborhood()
{
    // Implement neighborhood building using CUDA if needed
}

void Pbf::PbfImpl::computeDensityAndLambda()
{
    const size_t numParticles = h_particles.size();
    Kernel kernel(radius);

    PbfKernels::computeDensityAndLambda<<<(numParticles + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(d_predictedPositions.data()),
        thrust::raw_pointer_cast(d_densities.data()),
        thrust::raw_pointer_cast(d_lambdas.data()),
        rho0, eps, numParticles, kernel);
}

void Pbf::PbfImpl::computeDeltaP()
{
    const size_t numParticles = h_particles.size();
    Kernel kernel(radius);

    PbfKernels::computeDeltaP<<<(numParticles + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(d_predictedPositions.data()),
        thrust::raw_pointer_cast(d_lambdas.data()),
        thrust::raw_pointer_cast(d_predictedPositions.data()),
        rho0, kCorr, numParticles, kernel);
}

void Pbf::PbfImpl::updatePositionsAndVelocities(float dt)
{
    const size_t numParticles = h_particles.size();

    PbfKernels::updatePositionsAndVelocities<<<(numParticles + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(d_positions.data()),
        thrust::raw_pointer_cast(d_velocities.data()),
        thrust::raw_pointer_cast(d_predictedPositions.data()),
        dt, numParticles);
}

void Pbf::PbfImpl::applyViscosity()
{
    // Implement viscosity application using CUDA if needed
}

void Pbf::PbfImpl::applyVorticityConfinement()
{
    // Implement vorticity confinement using CUDA if needed
}

void Pbf::PbfImpl::boxCollision()
{
    const size_t numParticles = h_particles.size();
    const float3 min = make_float3(-2.0f, 0.0f, -1.0f);
    const float3 max = make_float3(2.0f, 2.0f, 1.0f);

    PbfKernels::boxCollision<<<(numParticles + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(d_predictedPositions.data()),
        min, max, numParticles);
}
