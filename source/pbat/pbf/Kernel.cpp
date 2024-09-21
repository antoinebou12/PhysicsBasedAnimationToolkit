// Kernel.cpp
#include "Kernel.h"
#include "doctest.h"

namespace pbat {
namespace pbf {

static const float pi = 3.14159265358979323846264338327950288f;

Kernel::Kernel(float _h)
    : h(_h), h2(h * h), h3(h2 * h), h9(h3 * h3 * h3), deltaqSq(0.1f * h * h)
{
    kPoly = 315.0f / (64.0f * pi * h9);
    kSpiky = 45.0f / (pi * h3 * h3);
}

float Kernel::poly6(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj) const
{
    const Eigen::Vector3f pipj = pi - pj;
    const float r2 = pipj.squaredNorm();
    if (r2 < h2)
    {
        const float x = h2 - r2;
        return kPoly * x * x * x;
    }
    return 0.0f;
}

Eigen::Vector3f Kernel::spiky(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj) const
{
    const Eigen::Vector3f pipj = pi - pj;
    const float r = pipj.norm();

    if (r < 1e-6f)
        return Eigen::Vector3f::Zero();
    else if (r < h)
    {
        const float x = (h - r);
        return -((kSpiky * x * x) / r) * pipj;
    }

    return Eigen::Vector3f::Zero();
}

float Kernel::scorr() const
{
    const float x = h2 - deltaqSq;
    return kPoly * x * x * x;
}

} // namespace pbf
} // namespace pbat

// Doctest unit tests
TEST_CASE("Kernel Functions")
{
    using namespace pbat::pbf;

    Kernel kernel(1.0f);
    Eigen::Vector3f pi(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f pj(0.5f, 0.0f, 0.0f);

    SUBCASE("poly6 kernel")
    {
        float value = kernel.poly6(pi, pj);
        CHECK(value > 0.0f);
    }

    SUBCASE("spiky kernel")
    {
        Eigen::Vector3f grad = kernel.spiky(pi, pj);
        CHECK(grad.norm() > 0.0f);
    }

    SUBCASE("scorr")
    {
        float scorrValue = kernel.scorr();
        CHECK(scorrValue > 0.0f);
    }
}
