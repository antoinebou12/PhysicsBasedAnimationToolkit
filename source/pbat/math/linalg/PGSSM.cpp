#include "PGSSM.h"
#include <tbb/parallel_for.h>
#include <algorithm>

namespace pbat {
namespace math {
namespace linalg {

PGSSM::PGSSM(int maxIter, int subIter) : m_maxIter(maxIter), m_subIter(subIter)
{
}

void PGSSM::Solve(Eigen::MatrixXf const& A, Eigen::VectorXf const& b, Eigen::VectorXf& x, Eigen::VectorXf const& lowerBounds, Eigen::VectorXf const& upperBounds) const
{
    Eigen::VectorXf residual = b - A * x;  // Initial residual
    std::vector<int> L, U, active_set; // Lower, Upper, and Active sets

    // Main loop over the maximum iterations
    for (int iter = 0; iter < m_maxIter; ++iter)
    {
        // Run PGS or subspace iterations (subspace correction)
        for (int subIter = 0; subIter < m_subIter; ++subIter)
        {
            // Update index sets based on current solution 'x'
            updateIndexSet(x, lowerBounds, upperBounds, L, U, active_set);

            // Solve for the active set in parallel
            tbb::parallel_for(tbb::blocked_range<int>(0, static_cast<int>(A.rows())), [&](const tbb::blocked_range<int>& range) {
                for (int i = range.begin(); i != range.end(); ++i)
                {
                    if (std::find(active_set.begin(), active_set.end(), i) != active_set.end()) {
                        float sigma = 0.0f;
                        for (int j = 0; j < static_cast<int>(A.cols()); ++j)
                        {
                            if (i != j)
                            {
                                sigma += A(i, j) * x(j);
                            }
                        }

                        // Update the active set variable λ_A
                        x(i) = (b(i) - sigma) / A(i, i);

                        // Project onto the box constraints: λ_A = min(u_A, max(l_A, λ_A))
                        x(i) = std::max(lowerBounds(i), std::min(upperBounds(i), x(i)));
                    }
                }
            });

            // Update residual after each sub-iteration (optional)
            residual = b - A * x;

            // Termination check can be added here if necessary
        }

        // Check for convergence (e.g., residual tolerance, max iterations)
        if (residual.norm() < 1e-6) {  // Example tolerance criterion
            break;
        }
    }
}

void PGSSM::updateIndexSet(Eigen::VectorXf const& x, Eigen::VectorXf const& lowerBounds, Eigen::VectorXf const& upperBounds, std::vector<int>& L, std::vector<int>& U, std::vector<int>& active_set) const
{
    L.clear();
    U.clear();
    active_set.clear();

    // Classify each element of 'x' into lower, upper, or active set
    for (int i = 0; i < x.size(); ++i)
    {
        if (x(i) == lowerBounds(i)) {
            L.push_back(i); // Lower bound set
        } else if (x(i) == upperBounds(i)) {
            U.push_back(i); // Upper bound set
        } else {
            active_set.push_back(i); // Active set (within bounds)
        }
    }
}

PGSSM::~PGSSM()
{
    // Destructor logic (if needed)
}

} // namespace linalg
} // namespace math
} // namespace pbat

#include <doctest/doctest.h>
TEST_CASE("[math][linalg] PGSSM")
{
    using namespace pbat::math::linalg;

    // Arrange
    PGSSM solver(100, 10); // Set maximum iterations to 100, subspace iterations to 10
    auto constexpr n = 3;
    Eigen::MatrixXf A = Eigen::MatrixXf::Identity(n, n) * 4.0f; // Simple diagonal matrix
    Eigen::VectorXf b = Eigen::VectorXf::Constant(n, 2.0f);     // Constant vector
    Eigen::VectorXf x = Eigen::VectorXf::Zero(n);               // Initial guess (zero vector)
    
    // Define box constraints (lower and upper bounds)
    Eigen::VectorXf lowerBounds = Eigen::VectorXf::Constant(n, 0.0f); // Non-negative solution
    Eigen::VectorXf upperBounds = Eigen::VectorXf::Constant(n, 5.0f); // Upper bound is 5.0

    SUBCASE("Solves simple SPD system with box constraints")
    {
        solver.Solve(A, b, x, lowerBounds, upperBounds);
        // Expected result: x = b / A diagonal = 2 / 4 = 0.5 for all elements, within bounds
        CHECK(x.isApprox(Eigen::VectorXf::Constant(n, 0.5f), 1e-5f));
    }

    SUBCASE("Respects upper bounds in solution")
    {
        b = Eigen::VectorXf::Constant(n, 30.0f); // Change the right-hand side to test upper bound projection
        solver.Solve(A, b, x, lowerBounds, upperBounds);
        // Expect x to be capped at the upper bound (5.0) for all elements
        CHECK(x.isApprox(upperBounds, 1e-5f));
    }

    SUBCASE("Respects lower bounds in solution")
    {
        b = Eigen::VectorXf::Constant(n, -30.0f); // Change the right-hand side to test lower bound projection
        solver.Solve(A, b, x, lowerBounds, upperBounds);
        // Expect x to be capped at the lower bound (0.0) for all elements
        CHECK(x.isApprox(lowerBounds, 1e-5f));
    }

    SUBCASE("Correctly updates index sets")
    {
        // Test with some x values at bounds and some within bounds
        x = Eigen::VectorXf::Constant(n, 3.0f);
        x(0) = lowerBounds(0);  // First element at lower bound
        x(2) = upperBounds(2);  // Last element at upper bound

        std::vector<int> L_, U_, A_;
        solver.updateIndexSet(x, lowerBounds, upperBounds, L_, U_, A_);

        CHECK(L_.size() == 1); // One element at lower bound
        CHECK(U_.size() == 1); // One element at upper bound
        CHECK(A_.size() == 1); // One element within bounds

        CHECK(L_[0] == 0);
        CHECK(U_[0] == 2);
        CHECK(A_[0] == 1);
    }
}
