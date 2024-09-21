#include "Pbf.h"
#include "PbfImpl.cuh"

Pbf::Pbf(float radius, float rho0, float eps, int maxIter, float c, float kCorr)
    : mImpl(new PbfImpl(radius, rho0, eps, maxIter, c, kCorr))
{
}

void Pbf::setParticles(const std::vector<Particle>& particles)
{
    mImpl->setParticles(particles);
}

const std::vector<Particle>& Pbf::getParticles() const
{
    return mImpl->getParticles();
}

void Pbf::step(float dt)
{
    mImpl->step(dt);
}
