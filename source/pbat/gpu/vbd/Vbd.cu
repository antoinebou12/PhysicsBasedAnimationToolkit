// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Vbd.h"
#include "VbdImpl.cuh"
#include "pbat/common/Eigen.h"

namespace pbat {
namespace gpu {
namespace vbd {

Vbd::Vbd(
    Eigen::Ref<GpuMatrixX const> const& X,
    Eigen::Ref<GpuIndexMatrixX const> const& V,
    Eigen::Ref<GpuIndexMatrixX const> const& F,
    Eigen::Ref<GpuIndexMatrixX const> const& T)
    : mImpl(new VbdImpl(X, V, F, T))
{
}

Vbd::Vbd(Vbd&& other) noexcept : mImpl(other.mImpl)
{
    other.mImpl = nullptr;
}

Vbd& Vbd::operator=(Vbd&& other)
{
    if (mImpl != nullptr)
        delete mImpl;
    mImpl       = other.mImpl;
    other.mImpl = nullptr;
    return *this;
}

Vbd::~Vbd()
{
    if (mImpl != nullptr)
        delete mImpl;
}

void Vbd::Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps, GpuScalar rho)
{
    mImpl->Step(dt, iterations, substeps, rho);
}

void Vbd::SetPositions(Eigen::Ref<GpuMatrixX const> const& X, bool bResetHistory)
{
    mImpl->SetPositions(X, bResetHistory);
}

void Vbd::SetVelocities(Eigen::Ref<GpuMatrixX const> const& v, bool bResetHistory)
{
    mImpl->SetVelocities(v, bResetHistory);
}

void Vbd::SetExternalAcceleration(Eigen::Ref<GpuMatrixX const> const& aext)
{
    mImpl->SetExternalAcceleration(aext);
}

void Vbd::SetMass(Eigen::Ref<GpuVectorX const> const& m)
{
    mImpl->SetMass(m);
}

void Vbd::SetQuadratureWeights(Eigen::Ref<GpuVectorX const> const& wg)
{
    mImpl->SetQuadratureWeights(wg);
}

void Vbd::SetShapeFunctionGradients(Eigen::Ref<GpuMatrixX const> const& GP)
{
    mImpl->SetShapeFunctionGradients(GP);
}

void Vbd::SetLameCoefficients(Eigen::Ref<GpuMatrixX const> const& l)
{
    mImpl->SetLameCoefficients(l);
}

void Vbd::SetVertexTetrahedronAdjacencyList(
    Eigen::Ref<GpuIndexVectorX const> const& GVTp,
    Eigen::Ref<GpuIndexVectorX const> const& GVTn,
    Eigen::Ref<GpuIndexVectorX const> const& GVTilocal)
{
    mImpl->SetVertexTetrahedronAdjacencyList(GVTp, GVTn, GVTilocal);
}

void Vbd::SetRayleighDampingCoefficient(GpuScalar kD)
{
    mImpl->SetRayleighDampingCoefficient(kD);
}

void Vbd::SetVertexPartitions(std::vector<std::vector<GpuIndex>> const& partitions)
{
    mImpl->SetVertexPartitions(partitions);
}

void Vbd::SetBlockSize(GpuIndex blockSize)
{
    mImpl->SetBlockSize(blockSize);
}

GpuMatrixX Vbd::GetPosition() const
{
    using pbat::common::ToEigen;
    return ToEigen(mImpl->X.x.Get()).reshaped(mImpl->X.x.Size(), mImpl->X.x.Dimensions());
}

GpuMatrixX Vbd::GetVelocity() const
{
    using pbat::common::ToEigen;
    auto const& velocity = mImpl->GetVelocity();
    return ToEigen(velocity.Get()).reshaped(velocity.Size(), velocity.Dimensions());
}

} // namespace vbd
} // namespace gpu
} // namespace pbat