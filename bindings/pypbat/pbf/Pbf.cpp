// Pbf.cpp
#include "Pbf.h"

#include <pbat/Aliases.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/pbf/Pbf.h>
#include <pbat/profiling/Profiling.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <string>
#include <type_traits>

namespace pbat {
namespace py {
namespace pbf {

void BindPbf([[maybe_unused]] pybind11::module& m)
{
    namespace pyb = pybind11;

    // Expose the Particle struct
    pyb::class_<pbat::pbf::Particle> particle_class(m, "Particle");
    particle_class.def(pyb::init<>())
        .def_readwrite("x", &pbat::pbf::Particle::x)
        .def_readwrite("v", &pbat::pbf::Particle::v)
        .def_readwrite("key", &pbat::pbf::Particle::key);

    // Expose the Pbf class
    using PbfType = pbat::pbf::Pbf;
    pyb::class_<PbfType> pbf_class(m, "Pbf");
    pbf_class.def(
        pyb::init<float, float, float, int, float, float>(),
        pyb::arg("radius"),
        pyb::arg("rho0"),
        pyb::arg("eps"),
        pyb::arg("maxIter"),
        pyb::arg("c"),
        pyb::arg("kCorr"));

    pbf_class.def(
        "setParticles",
        [](PbfType& pbf, const std::vector<pbat::pbf::Particle>& particles) {
            pbat::profiling::Profile("pbf.Pbf.setParticles", [&]() {
                pbf.setParticles(particles);
            });
        },
        pyb::arg("particles"));

    pbf_class.def(
        "getParticles",
        [](PbfType& pbf) -> const std::vector<pbat::pbf::Particle>& {
            return pbat::profiling::Profile(
                "pbf.Pbf.getParticles",
                [&]() -> const std::vector<pbat::pbf::Particle>& { return pbf.getParticles(); });
        },
        pyb::return_value_policy::reference_internal);

    pbf_class.def(
        "step",
        [](PbfType& pbf, float dt) {
            pbat::profiling::Profile("pbf.Pbf.step", [&]() { pbf.step(dt); });
        },
        pyb::arg("dt"));

    // Documentation for the Pbf class
    pbf_class.doc() = "Position Based Fluids (PBF) simulation class.";
}

} // namespace pbf
} // namespace py
} // namespace pbat
