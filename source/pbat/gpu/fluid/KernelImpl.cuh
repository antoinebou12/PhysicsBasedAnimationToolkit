#ifndef KERNEL_IMPL_CUH
#define KERNEL_IMPL_CUH

#include <Eigen/Dense>

namespace SPH
{
    // Declarations of internal helper functions

    /**
     * @brief Poly6 kernel function implementation.
     * @param pi Position vector of particle i.
     * @param pj Position vector of particle j.
     * @param h2 Square of the smoothing length.
     * @param kPoly Precomputed kernel coefficient.
     * @return The scalar kernel value between particles i and j.
     */
    float poly6_kernel(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj, float h2, float kPoly);

    /**
     * @brief Spiky kernel gradient function implementation.
     * @param pi Position vector of particle i.
     * @param pj Position vector of particle j.
     * @param h Smoothing length.
     * @param kSpiky Precomputed kernel coefficient.
     * @return The gradient vector of the kernel between particles i and j.
     */
    Eigen::Vector3f spiky_kernel(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj, float h, float kSpiky);

} // namespace SPH

#endif // KERNEL_IMPL_CUH
