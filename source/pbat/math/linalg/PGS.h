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

    void SetMaxIterations(int maxIter) { m_maxIter = maxIter; }
    int GetMaxIterations() const { return m_maxIter; }

private:
    int m_maxIter;
};

} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_PGS_H
