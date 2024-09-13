#ifndef KERNEL_H
#define KERNEL_H

#include <Eigen/Dense>

namespace SPH
{
    class Kernel
    {
    private:
        static constexpr float pi = 3.14159265358979323846f;

        float h, h2, h3, h9;
        float deltaqSq;
        float kPoly, kSpiky;

    public:
        /**
         * @brief Constructs a Kernel object with a specified smoothing length.
         * @param _h Smoothing length (kernel radius).
         */
        explicit Kernel(float _h = 0.0f);

        /**
         * @brief The poly6 kernel function.
         * @param pi Position vector of particle i.
         * @param pj Position vector of particle j.
         * @return The scalar kernel value between particles i and j.
         */
        float poly6(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj) const;

        /**
         * @brief The spiky kernel gradient function.
         *        Use this for gradient calculations, e.g., when computing lambda.
         * @param pi Position vector of particle i.
         * @param pj Position vector of particle j.
         * @return The gradient vector of the kernel between particles i and j.
         */
        Eigen::Vector3f spiky(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj) const;

        /**
         * @brief Computes the correction factor for density fluctuation correction in SPH simulations.
         * @return The scalar correction factor.
         */
        float scorr() const;
    };
} // namespace SPH

#endif // KERNEL_H
