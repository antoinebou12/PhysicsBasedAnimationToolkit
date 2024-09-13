#include "Kernel.h"
#include "KernelImpl.cuh"

namespace SPH
{
    // Definition of the Impl class
    class Kernel::Impl
    {
    public:
        Impl(float _h);

        float poly6(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj) const;
        Eigen::Vector3f spiky(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj) const;
        float scorr() const;

    private:
        static constexpr float pi = 3.14159265358979323846f;

        float h, h2, h3, h9;
        float deltaqSq;
        float kPoly, kSpiky;
    };

    Kernel::Kernel(float _h)
        : pImpl(new Impl(_h))
    {
    }

    Kernel::~Kernel()
    {
        delete pImpl;
    }

    float Kernel::poly6(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj) const
    {
        return pImpl->poly6(pi, pj);
    }

    Eigen::Vector3f Kernel::spiky(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj) const
    {
        return pImpl->spiky(pi, pj);
    }

    float Kernel::scorr() const
    {
        return pImpl->scorr();
    }

    // Implementation of Impl methods
    Kernel::Impl::Impl(float _h)
        : h(_h),
          h2(_h * _h),
          h3(h2 * _h),
          h9(h3 * h3 * h3),
          deltaqSq(0.1f * h2),
          kPoly(315.0f / (64.0f * pi * h9)),
          kSpiky(45.0f / (pi * h3 * h3))
    {
    }

    float Kernel::Impl::poly6(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj) const
    {
        return poly6_kernel(pi, pj, h2, kPoly);
    }

    Eigen::Vector3f Kernel::Impl::spiky(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj) const
    {
        return spiky_kernel(pi, pj, h, kSpiky);
    }

    float Kernel::Impl::scorr() const
    {
        float x = h2 - deltaqSq;
        return kPoly * x * x * x;
    }

} // namespace SPH
