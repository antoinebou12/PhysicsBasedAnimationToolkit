// Pbf.cpp
#include "Pbf.h"
#include "../../pbf/Pbf.h"  // Include the PBF simulation header

#include <pybind11/stl.h>   // For automatic conversion of STL containers
#include <pybind11/eigen.h> // For automatic conversion of Eigen types

namespace py = pybind11;

namespace pbat {
namespace py {
namespace pbf {

void Bind(py::module& m)
{
    // Create a submodule for pbf
    auto mpbf = m.def_submodule("pbf");

    // Expose the Particle struct
    py::class_<pbat::pbf::Particle>(mpbf, "Particle")
        .def(py::init<>())
        .def_readwrite("x", &pbat::pbf::Particle::x)
        .def_readwrite("v", &pbat::pbf::Particle::v)
        .def_readwrite("key", &pbat::pbf::Particle::key);

    // Expose the Pbf class
    py::class_<pbat::pbf::Pbf>(mpbf, "Pbf")
        .def(py::init<float, float, float, int, float, float>(),
             py::arg("radius"),
             py::arg("rho0"),
             py::arg("eps"),
             py::arg("maxIter"),
             py::arg("c"),
             py::arg("kCorr"))
        .def("setParticles", &pbat::pbf::Pbf::setParticles)
        .def("getParticles", &pbat::pbf::Pbf::getParticles, py::return_value_policy::reference_internal)
        .def("step", &pbat::pbf::Pbf::step);
}

} // namespace pbf
} // namespace py
} // namespace pbat
