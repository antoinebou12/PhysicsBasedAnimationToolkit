// Pbf.h
#ifndef PBF_H
#define PBF_H

#include "Particle.h"
#include "Kernel.h"
#include "HashGrid.h"
#include <vector>
#include <memory>
#include <Eigen/Dense>

namespace pbat {
namespace pbf {

class Pbf
{
public:
    Pbf(float radius, float rho0, float eps, int maxIter, float c, float kCorr);

    void setParticles(const std::vector<Particle>& particles);
    std::vector<Particle>& getParticles();
    const std::vector<Particle>& getParticles() const;

    void step(float dt);

private:
    // Implementation class
    class PbfImpl
    {
    public:
        PbfImpl(float radius, float rho0, float eps, int maxIter, float c, float kCorr);

        void setParticles(const std::vector<Particle>& particles);
        std::vector<Particle>& getParticles();
        const std::vector<Particle>& getParticles() const;

        void step(float dt);

    private:
        void updatePredictedPositions(float dt);
        void updateHashGrid();
        void buildNeighborhood();
        void computeDensityAndLambda();
        void computeDeltaP();
        void updatePositionsAndVelocities(float dt);
        void applyViscosity();
        void applyVorticityConfinement();
        void boxCollision();

        float radius;
        float rho0;
        float eps;
        int maxIter;
        float c;
        float kCorr;

        HashGrid m_hashGrid;
        std::vector<Particle> m_particles;

        Eigen::Vector3f aabbMin;
        Eigen::Vector3f aabbMax;
    };

    std::unique_ptr<PbfImpl> mImpl;
};

} // namespace pbf
} // namespace pbat

#endif // PBF_H
