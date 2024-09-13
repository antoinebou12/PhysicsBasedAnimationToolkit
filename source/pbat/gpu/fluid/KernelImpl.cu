#include "KernelImpl.cuh"
#include <algorithm> // For std::max

namespace SPH
{
    float poly6_kernel(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj, float h2, float kPoly)
    {
        Eigen::Vector3f r_vec = pi - pj;
        float r2 = r_vec.squaredNorm();
        if (r2 < h2)
        {
            float x = h2 - r2;
            return kPoly * x * x * x;
        }
        return 0.0f;
    }

    Eigen::Vector3f spiky_kernel(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj, float h, float kSpiky)
    {
        Eigen::Vector3f r_vec = pi - pj;
        float r = r_vec.norm();

        if (r < 1e-6f)
        {
            return Eigen::Vector3f::Zero();
        }

        if (r < h)
        {
            float x = h - r;
            float coeff = (kSpiky * x * x) / r;
            return -coeff * r_vec;
        }

        return Eigen::Vector3f::Zero();
    }
} // namespace SPH
