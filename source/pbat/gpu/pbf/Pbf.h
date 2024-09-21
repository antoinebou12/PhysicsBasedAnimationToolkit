#ifndef PBF_H
#define PBF_H

#include "Particle.h"
#include <vector>
#include <memory>

class Pbf
{
public:
    Pbf(float radius, float rho0, float eps, int maxIter, float c, float kCorr);

    void setParticles(const std::vector<Particle>& particles);
    const std::vector<Particle>& getParticles() const;

    void step(float dt);

private:
    class PbfImpl;
    std::unique_ptr<PbfImpl> mImpl;
};

#endif // PBF_H
