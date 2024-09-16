#include "PGS.h"

#include <pbat/math/linalg/PGS.h>
#include <pbat/profiling/Profiling.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace math {
namespace linalg {

void BindPGS([[maybe_unused]] pybind11::module& m)
{
    namespace pyb = pybind11;

    const std::string className = "PGS";
    using PGSType = pbat::math::linalg::PGS;

    pyb::class_<PGSType>(m, className.data())
        // Constructor: PGS(maxIter=100)
        .def(pyb::init<int>(), pyb::arg("maxIter") = 100, 
             "Constructor with optional maximum iterations (default is 100).")
        // Solve method
        .def(
            "solve",
            [](PGSType& solver,
               const Eigen::MatrixXf& A,
               const Eigen::VectorXf& b,
               Eigen::VectorXf& x,
               const Eigen::VectorXf& lowerBounds,
               const Eigen::VectorXf& upperBounds) {
                pbat::profiling::Profile("math.linalg.PGS.Solve", [&]() {
                    solver.Solve(A, b, x, lowerBounds, upperBounds);
                });
            },
            pyb::arg("A"),
            pyb::arg("b"),
            pyb::arg("x"),
            pyb::arg("lower_bounds"),
            pyb::arg("upper_bounds"),
            R"pbdoc(
                Solves a linear system using the Projected Gauss-Seidel method with box constraints.

                Parameters:
                - A: Matrix representing the system (MatrixXf).
                - b: Vector representing the right-hand side of the equation (VectorXf).
                - x: Vector where the solution will be stored (VectorXf).
                - lower_bounds: Vector representing the lower bounds for each variable (VectorXf).
                - upper_bounds: Vector representing the upper bounds for each variable (VectorXf).
            )pbdoc")
        // Set max iterations
        .def(
            "set_max_iterations",
            [](PGSType& solver, int maxIter) {
                solver.SetMaxIterations(maxIter);
            },
            pyb::arg("max_iter"),
            "Sets the maximum number of iterations for the solver.")
        // Get max iterations
        .def(
            "get_max_iterations",
            [](PGSType& solver) {
                return solver.GetMaxIterations();
            },
            "Returns the maximum number of iterations for the solver.");
}

} // namespace linalg
} // namespace math
} // namespace py
} // namespace pbat
