#include "Kernel.h"

namespace SPH
{
    Kernel::Kernel(float _h)
        : h(_h),
          h2(_h * _h),
          h3(_h * _h * _h),
          h9(h3 * h3 * h3),
          deltaqSq(0.1f * h2),
          kPoly(315.0f / (64.0f * pi * h9)),
          kSpiky(45.0f / (pi * h3 * h3))
    {
    }

    float Kernel::poly6(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj) const
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

    Eigen::Vector3f Kernel::spiky(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj) const
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

    float Kernel::scorr() const
    {
        float x = h2 - deltaqSq;
        return kPoly * x * x * x;
    }
} // namespace SPH
