#ifndef PBF_IMPL_CUH
#define PBF_IMPL_CUH

#include "Particle.h"
#include "Kernel.h"
#include "HashGrid.h"
#include <vector>
#include <thrust/device_vector.h>

class Pbf::PbfImpl
{
public:
    PbfImpl(float radius, float rho0, float eps, int maxIter, float c, float kCorr);

    void setParticles(const std::vector<Particle>& particles);
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

    thrust::device_vector<float3> d_positions;
    thrust::device_vector<float3> d_velocities;
    thrust::device_vector<float3> d_predictedPositions;
    thrust::device_vector<float> d_densities;
    thrust::device_vector<float> d_lambdas;
    thrust::device_vector<int> d_keys;

    HashGrid hashGrid;

    std::vector<Particle> h_particles;
};

#endif // PBF_IMPL_CUH
