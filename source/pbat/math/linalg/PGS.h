#ifndef PBAT_MATH_LINALG_PGS_H
#define PBAT_MATH_LINALG_PGS_H

#include <Eigen/Dense>

namespace pbat {
namespace math {
namespace linalg {

class PGS
{
public:
    PGS(int maxIter = 100);
    ~PGS();

    void Solve(Eigen::MatrixXf const& A,
               Eigen::VectorXf const& b,
               Eigen::VectorXf& x,
               Eigen::VectorXf const& lowerBounds,
               Eigen::VectorXf const& upperBounds) const;

    void SetMaxIterations(int maxIter);
    int GetMaxIterations() const;

private:
    int m_maxIter;
};

} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_PGS_H
