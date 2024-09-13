#ifndef PBAT_PY_FLUID_KERNEL_BINDINGS_H
#define PBAT_PY_FLUID_KERNEL_BINDINGS_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace fluid {

/**
 * @brief Binds the SPH::Kernel class to Python using Pybind11.
 *
 * This function exposes the SPH kernel functionality (e.g., poly6, spiky, and scorr) to Python.
 * It should be called within the Pybind11 module definition to add the Kernel class to the Python module.
 *
 * @param m The Pybind11 module to which the Kernel class is bound.
 */
void BindKernel(pybind11::module& m);

} // namespace fluid
} // namespace py
} // namespace pbat

#endif // PBAT_PY_FLUID_KERNEL_BINDINGS_H
