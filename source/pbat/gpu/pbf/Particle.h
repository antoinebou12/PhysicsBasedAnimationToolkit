#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>
#include <Eigen/Dense>

struct Particle
{
    Eigen::Vector3f x;      // Current position
    Eigen::Vector3f v;      // Velocity
    Eigen::Vector3f xstar;  // Predicted position
    float rho;              // Density
    float lambda;           // Lagrange multiplier
    int key;                // Hash grid key
};

#endif // PARTICLE_H
