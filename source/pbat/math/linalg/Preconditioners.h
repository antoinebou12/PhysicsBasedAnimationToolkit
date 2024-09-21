#ifndef PBAT_MATH_LINALG_PRECONDITIONERS_H
#define PBAT_MATH_LINALG_PRECONDITIONERS_H

#include <Eigen/Dense>

namespace pbat {
namespace math {
namespace linalg {

class Preconditioners
{
public:
    // Diagonal (Jacobi) preconditioner
    static Eigen::VectorXf Diagonal(const Eigen::MatrixXf& A)
    {
        return A.diagonal();
    }

    // Inverse of the diagonal preconditioner
    static Eigen::VectorXf InverseDiagonal(const Eigen::MatrixXf& A)
    {
        return A.diagonal().cwiseInverse();
    }
};

} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_PRECONDITIONERS_H
