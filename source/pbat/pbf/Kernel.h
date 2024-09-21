// Kernel.h
#ifndef KERNEL_H
#define KERNEL_H

#include <Eigen/Dense>

namespace pbat {
namespace pbf {

class Kernel
{
public:
    explicit Kernel(float h);

    float poly6(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj) const;
    Eigen::Vector3f spiky(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj) const;
    float scorr() const;

private:
    float h, h2, h3, h9, deltaqSq;
    float kPoly, kSpiky;
};

} // namespace pbf
} // namespace pbat

#endif // KERNEL_H
