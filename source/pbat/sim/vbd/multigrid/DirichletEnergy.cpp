#include "DirichletEnergy.h"

#include "pbat/fem/Jacobian.h"
#include "pbat/fem/ShapeFunctions.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

DirichletEnergy::DirichletEnergy(
    Data const& problem,
    DirichletQuadrature const& DQ)
    : muD(problem.muD), dg(DQ.Xg)
{
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat