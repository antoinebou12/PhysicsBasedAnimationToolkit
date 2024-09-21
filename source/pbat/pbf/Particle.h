// Particle.h
#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>
#include <Eigen/Dense>

namespace pbat {
namespace pbf {

struct Particle
{
    Eigen::Vector3f x;      // Current position
    Eigen::Vector3f v;      // Velocity
    Eigen::Vector3f xstar;  // Predicted position
    Eigen::Vector3f vdiff;  // Velocity difference (for viscosity)
    Eigen::Vector3f deltaP; // Position correction
    Eigen::Vector3f omega;  // Vorticity
    float rho;              // Density
    float lambda;           // Lagrange multiplier
    int key;                // Hash grid key
    std::vector<Particle*> N; // Neighboring particles
};

} // namespace pbf
} // namespace pbat

#endif // PARTICLE_H
