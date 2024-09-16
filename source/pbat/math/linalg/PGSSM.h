#ifndef PBAT_MATH_LINALG_PGSSM_H
#define PBAT_MATH_LINALG_PGSSM_H

#include <Eigen/Dense>
#include <vector>

namespace pbat {
namespace math {
namespace linalg {

class PGSSM
{
public:
    PGSSM(int maxIter = 100, int subIter = 5);
    ~PGSSM();

    void Solve(Eigen::MatrixXf const& A,
               Eigen::VectorXf const& b,
               Eigen::VectorXf& x,
               Eigen::VectorXf const& lowerBounds,
               Eigen::VectorXf const& upperBounds) const;

    // New method to update index sets based on bounds
    void updateIndexSet(Eigen::VectorXf const& x,
                        Eigen::VectorXf const& lowerBounds,
                        Eigen::VectorXf const& upperBounds,
                        std::vector<int>& L,
                        std::vector<int>& U,
                        std::vector<int>& A) const;
                        
    void SetMaxIterations(int maxIter);
    int GetMaxIterations() const;

    void SetSubIterations(int subIter);
    int GetSubIterations() const;

private:
    int m_maxIter;
    int m_subIter; // Number of subspace iterations for each iteration
};

} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_PGSSM_H
