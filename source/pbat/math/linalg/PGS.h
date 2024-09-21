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

    void SetMaxIterations(int maxIter);
    int GetMaxIterations() const;

    void Solve(const Eigen::MatrixXf& A,
               const Eigen::VectorXf& b,
               Eigen::VectorXf& x,
               const Eigen::VectorXf& lowerBounds,
               const Eigen::VectorXf& upperBounds,
               const Eigen::VectorXf& preconditioner = Eigen::VectorXf()) const;

private:
    int m_maxIter;
};

} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_PGS_H
