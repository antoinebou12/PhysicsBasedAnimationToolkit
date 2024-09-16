#ifndef PYPBAT_MATH_LINALG_PGS_H
#define PYPBAT_MATH_LINALG_PGS_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace math {
namespace linalg {

void BindPGS(pybind11::module& m);

} // namespace linalg
} // namespace math
} // namespace py
} // namespace pbat

#endif // PYPBAT_MATH_LINALG_PGS_H
