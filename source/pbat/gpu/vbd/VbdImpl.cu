// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "VbdImpl.cuh"
#include "VbdImplKernels.cuh"

#include <cuda/api.hpp>
#include <thrust/async/copy.h>
#include <thrust/async/for_each.h>
#include <thrust/execution_policy.h>

namespace pbat {
namespace gpu {
namespace vbd {

VbdImpl::VbdImpl(
    Eigen::Ref<GpuMatrixX const> const& Xin,
    Eigen::Ref<GpuIndexMatrixX const> const& Vin,
    Eigen::Ref<GpuIndexMatrixX const> const& Fin,
    Eigen::Ref<GpuIndexMatrixX const> const& Tin)
    : X(Xin),
      V(Vin),
      F(Fin),
      T(Tin),
      mPositionsAtTMinus1(Xin.cols()),
      mPositionsAtT(Xin.cols()),
      mInertialTargetPositions(Xin.cols()),
      mVelocitiesAtT(Xin.cols()),
      mVelocities(Xin.cols()),
      mExternalAcceleration(Xin.cols()),
      mMass(Xin.cols()),
      mQuadratureWeights(Tin.cols()),
      mShapeFunctionGradients(Tin.cols() * 4 * 3),
      mLameCoefficients(Tin.cols()),
      mVertexTetrahedronNeighbours(),
      mVertexTetrahedronPrefix(Xin.cols() + 1),
      mVertexTetrahedronLocalVertexIndices(),
      mRayleighDamping(GpuScalar{0}),
      mCollisionPenalty(GpuScalar{1e3}),
      mMaxCollidingTrianglesPerVertex(8),
      mCollidingTriangles(8 * Xin.cols()),
      mCollidingTriangleCount(Xin.cols()),
      mPartitions(),
      mGpuThreadBlockSize(64)
{
    for (auto d = 0; d < X.Dimensions(); ++d)
    {
        thrust::copy(X.x[d].begin(), X.x[d].end(), mPositionsAtTMinus1[d].begin());
        thrust::copy(X.x[d].begin(), X.x[d].end(), mPositionsAtT[d].begin());
        thrust::fill(mVelocitiesAtT[d].begin(), mVelocitiesAtT[d].end(), GpuScalar{0});
        thrust::fill(mVelocities[d].begin(), mVelocities[d].end(), GpuScalar{0});
        thrust::fill(
            mExternalAcceleration[d].begin(),
            mExternalAcceleration[d].end(),
            GpuScalar{0});
    }
    thrust::fill(mMass.Data(), mMass.Data() + mMass.Size(), GpuScalar{1e3});
}

void VbdImpl::Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps, GpuScalar rho)
{
    GpuScalar sdt            = dt / static_cast<GpuScalar>(substeps);
    GpuScalar sdt2           = sdt * sdt;
    GpuIndex const nVertices = static_cast<GpuIndex>(X.NumberOfPoints());

    kernels::BackwardEulerMinimization bdf{};
    bdf.dt                              = sdt;
    bdf.dt2                             = sdt2;
    bdf.m                               = mMass.Raw();
    bdf.xtilde                          = mInertialTargetPositions.Raw();
    bdf.xt                              = mPositionsAtT.Raw();
    bdf.x                               = X.x.Raw();
    bdf.T                               = T.inds.Raw();
    bdf.GP                              = mShapeFunctionGradients.Raw();
    bdf.lame                            = mLameCoefficients.Raw();
    bdf.GVTn                            = mVertexTetrahedronNeighbours.Raw();
    bdf.GVTp                            = mVertexTetrahedronPrefix.Raw();
    bdf.GVTilocal                       = mVertexTetrahedronLocalVertexIndices.Raw();
    bdf.kD                              = mRayleighDamping;
    bdf.kC                              = mCollisionPenalty;
    bdf.nMaxCollidingTrianglesPerVertex = mMaxCollidingTrianglesPerVertex;
    bdf.FC                              = mCollidingTriangles.Raw();
    bdf.nCollidingTriangles             = mCollidingTriangleCount.Raw();
    bdf.F                               = F.inds.Raw();

    for (auto s = 0; s < substeps; ++s)
    {
        // Compute inertial target positions
        thrust::device_event e = thrust::async::for_each(
            thrust::device,
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nVertices),
            kernels::FInertialTarget{
                sdt,
                sdt2,
                X.x.Raw(),
                mVelocities.Raw(),
                mExternalAcceleration.Raw(),
                mInertialTargetPositions.Raw()});
        // Initialize block coordinate descent's, i.e. BCD's, solution
        e = thrust::async::for_each(
            thrust::device.after(e),
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nVertices),
            kernels::FAdaptiveInitialization{
                sdt,
                sdt2,
                X.x.Raw(),
                mVelocitiesAtT.Raw(),
                mVelocities.Raw(),
                mExternalAcceleration.Raw(),
                X.x.Raw()});
        // Prepare state history for Chebyshev acceleration, i.e.
        // x(t-2) <- x(t-1)
        // x(t-1) <- x(t)
        for (auto d = 0; d < X.x.Dimensions(); ++d)
        {
            e = thrust::async::copy(
                thrust::device.after(e),
                mPositionsAtT[d].begin(),
                mPositionsAtT[d].end(),
                mPositionsAtTMinus1[d].begin());
            e = thrust::async::copy(
                thrust::device.after(e),
                X.x[d].begin(),
                X.x[d].end(),
                mPositionsAtT[d].begin());
        }
        // Initialize Chebyshev semi-iterative method
        kernels::FChebyshev fChebyshev{
            rho,
            mPositionsAtTMinus1.Raw(),
            mPositionsAtT.Raw(),
            X.x.Raw()};
        // Share thrust's underlying CUDA stream with cuda-api-wrappers
        auto device                           = cuda::device::get(e.where());
        auto stream                           = device.default_stream();
        CUstream_st* thrustNativeStreamHandle = e.stream().get();
        CUstream_st* cawNativeStreamHandle    = stream.handle();
        assert(thrustNativeStreamHandle == cawNativeStreamHandle);
        auto kDynamicSharedMemoryCapacity = static_cast<cuda::memory::shared::size_t>(
            mGpuThreadBlockSize * bdf.ExpectedSharedMemoryPerThreadInBytes());
        auto bcdLaunchConfiguration = cuda::launch_config_builder()
                                          .grid_size(nVertices)
                                          .block_size(mGpuThreadBlockSize)
                                          .dynamic_shared_memory_size(kDynamicSharedMemoryCapacity)
                                          .build();
        // Minimize Backward Euler, i.e. BDF1, objective
        for (auto k = 0; k < iterations; ++k)
        {
            for (auto& partition : mPartitions)
            {
                bdf.partition = partition.Raw();
                stream.enqueue.kernel_launch(
                    kernels::MinimizeBackwardEuler,
                    bcdLaunchConfiguration,
                    bdf);
            }
            e = thrust::async::for_each(
                thrust::device.on(stream.handle()),
                thrust::make_counting_iterator<GpuIndex>(0),
                thrust::make_counting_iterator<GpuIndex>(nVertices),
                fChebyshev);
            fChebyshev.Update(k);
        }
        // Update velocities
        for (auto d = 0; d < mVelocities.Dimensions(); ++d)
        {
            e = thrust::async::copy(
                thrust::device.after(e),
                mVelocities[d].begin(),
                mVelocities[d].end(),
                mVelocitiesAtT[d].begin());
        }
        e = thrust::async::for_each(
            thrust::device.after(e),
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nVertices),
            kernels::FFinalizeSolution{sdt, mPositionsAtT.Raw(), X.x.Raw(), mVelocities.Raw()});

        e.wait();
    }
}

void VbdImpl::SetPositions(Eigen::Ref<GpuMatrixX const> const& Xin, bool bResetHistory)
{
    auto const nVertices = static_cast<GpuIndex>(X.x.Size());
    if (Xin.rows() != 3 and Xin.cols() != nVertices)
    {
        std::ostringstream ss{};
        ss << "Expected positions of dimensions " << X.x.Dimensions() << "x" << X.x.Size()
           << ", but got " << Xin.rows() << "x" << Xin.cols() << "\n";
        throw std::invalid_argument(ss.str());
    }
    for (auto d = 0; d < X.x.Dimensions(); ++d)
    {
        thrust::copy(Xin.row(d).begin(), Xin.row(d).end(), X.x[d].begin());
        if (bResetHistory)
        {
            thrust::copy(Xin.row(d).begin(), Xin.row(d).end(), mPositionsAtTMinus1[d].begin());
            thrust::copy(Xin.row(d).begin(), Xin.row(d).end(), mPositionsAtT[d].begin());
        }
    }
}

void VbdImpl::SetVelocities(Eigen::Ref<GpuMatrixX const> const& v, bool bResetHistory)
{
    auto const nVertices = static_cast<GpuIndex>(mVelocities.Size());
    if (v.rows() != 3 and v.cols() != nVertices)
    {
        std::ostringstream ss{};
        ss << "Expected velocities of dimensions " << mVelocities.Dimensions() << "x"
           << mVelocities.Size() << ", but got " << v.rows() << "x" << v.cols() << "\n";
        throw std::invalid_argument(ss.str());
    }
    for (auto d = 0; d < mVelocities.Dimensions(); ++d)
    {
        thrust::copy(v.row(d).begin(), v.row(d).end(), mVelocities[d].begin());
        if (bResetHistory)
        {
            thrust::copy(v.row(d).begin(), v.row(d).end(), mVelocitiesAtT[d].begin());
        }
    }
}

void VbdImpl::SetExternalAcceleration(Eigen::Ref<GpuMatrixX const> const& aext)
{
    auto const nVertices = static_cast<GpuIndex>(mExternalAcceleration.Size());
    if (aext.rows() != 3 and aext.cols() != nVertices)
    {
        std::ostringstream ss{};
        ss << "Expected accelerations of dimensions " << mExternalAcceleration.Dimensions() << "x"
           << mExternalAcceleration.Size() << ", but got " << aext.rows() << "x" << aext.cols()
           << "\n";
        throw std::invalid_argument(ss.str());
    }
    for (auto d = 0; d < mExternalAcceleration.Dimensions(); ++d)
        thrust::copy(aext.row(d).begin(), aext.row(d).end(), mExternalAcceleration[d].begin());
}

void VbdImpl::SetMass(Eigen::Ref<GpuVectorX const> const& m)
{
    auto const nVertices = static_cast<GpuIndex>(mMass.Size());
    if (m.size() != nVertices)
    {
        std::ostringstream ss{};
        ss << "Expected masses of dimensions " << nVertices << "x1 or its transpose, but got "
           << m.size() << "\n";
        throw std::invalid_argument(ss.str());
    }
    thrust::copy(m.data(), m.data() + m.size(), mMass.Data());
}

void VbdImpl::SetQuadratureWeights(Eigen::Ref<GpuVectorX const> const& wg)
{
    auto const nTetrahedra = static_cast<GpuIndex>(T.inds.Size());
    if (wg.size() != nTetrahedra)
    {
        std::ostringstream ss{};
        ss << "Expected quadrature weights of dimensions " << nTetrahedra
           << "x1 or its transpose, but got " << wg.rows() << "x" << wg.cols() << "\n";
        throw std::invalid_argument(ss.str());
    }
    thrust::copy(wg.data(), wg.data() + wg.size(), mQuadratureWeights.Data());
}

void VbdImpl::SetShapeFunctionGradients(Eigen::Ref<GpuMatrixX const> const& GP)
{
    auto const nTetrahedra = static_cast<GpuIndex>(T.inds.Size());
    if (GP.rows() != 4 and GP.cols() != nTetrahedra * 3)
    {
        std::ostringstream ss{};
        ss << "Expected shape function gradients of dimensions 4x" << nTetrahedra * 3
           << ", but got " << GP.rows() << "x" << GP.cols() << "\n";
        throw std::invalid_argument(ss.str());
    }
    thrust::copy(GP.data(), GP.data() + GP.size(), mShapeFunctionGradients.Data());
}

void VbdImpl::SetLameCoefficients(Eigen::Ref<GpuMatrixX const> const& l)
{
    auto const nTetrahedra = static_cast<GpuIndex>(T.inds.Size());
    if (l.rows() != 2 and l.cols() != nTetrahedra)
    {
        std::ostringstream ss{};
        ss << "Expected Lame coefficients of dimensions 2x" << nTetrahedra << ", but got "
           << l.rows() << "x" << l.cols() << "\n";
        throw std::invalid_argument(ss.str());
    }
    thrust::copy(l.data(), l.data() + l.size(), mLameCoefficients.Data());
}

void VbdImpl::SetVertexTetrahedronAdjacencyList(
    Eigen::Ref<GpuIndexVectorX const> const& GVTn,
    Eigen::Ref<GpuIndexVectorX const> const& GVTp,
    Eigen::Ref<GpuIndexVectorX const> const& GVTilocal)
{
    thrust::copy(GVTn.data(), GVTn.data() + GVTn.size(), mVertexTetrahedronNeighbours.Data());
    thrust::copy(GVTp.data(), GVTp.data() + GVTp.size(), mVertexTetrahedronPrefix.Data());
    thrust::copy(
        GVTilocal.data(),
        GVTilocal.data() + GVTilocal.size(),
        mVertexTetrahedronLocalVertexIndices.Data());
}

void VbdImpl::SetRayleighDampingCoefficient(GpuScalar kD)
{
    mRayleighDamping = kD;
}

void VbdImpl::SetVertexPartitions(std::vector<std::vector<GpuIndex>> const& partitions)
{
    mPartitions.resize(partitions.size());
    for (auto p = 0; p < partitions.size(); ++p)
    {
        mPartitions[p].Resize(partitions[p].size());
        thrust::copy(partitions[p].begin(), partitions[p].end(), mPartitions[p].Data());
    }
}

void VbdImpl::SetBlockSize(GpuIndex blockSize)
{
    mGpuThreadBlockSize = blockSize;
}

common::Buffer<GpuScalar, 3> const& VbdImpl::GetVelocity() const
{
    return mVelocities;
}

common::Buffer<GpuScalar, 3> const& VbdImpl::GetExternalAcceleration() const
{
    return mExternalAcceleration;
}

common::Buffer<GpuScalar> const& VbdImpl::GetMass() const
{
    return mMass;
}

common::Buffer<GpuScalar> const& VbdImpl::GetShapeFunctionGradients() const
{
    return mShapeFunctionGradients;
}

common::Buffer<GpuScalar> const& VbdImpl::GetLameCoefficients() const
{
    return mLameCoefficients;
}

std::vector<common::Buffer<GpuIndex>> const& VbdImpl::GetPartitions() const
{
    return mPartitions;
}

} // namespace vbd
} // namespace gpu
} // namespace pbat

#include "pbat/common/Eigen.h"
#include "pbat/fem/Jacobian.h"
#include "pbat/fem/Mesh.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/physics/HyperElasticity.h"

#include <Eigen/SparseCore>
#include <doctest/doctest.h>
#include <span>
#include <vector>

TEST_CASE("[gpu][xpbd] Xpbd")
{
    using namespace pbat;
    // Arrange
    // Cube mesh
    GpuMatrixX P(3, 8);
    GpuIndexMatrixX V(1, 8);
    GpuIndexMatrixX T(4, 5);
    GpuIndexMatrixX F(3, 12);
    // clang-format off
    P << 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f,
         0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f,
         0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f;
    T << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    F << 0, 1, 1, 3, 3, 2, 2, 0, 0, 0, 4, 5,
         1, 5, 3, 7, 2, 6, 0, 4, 3, 2, 5, 7,
         4, 4, 5, 5, 7, 7, 6, 6, 1, 3, 6, 6;
    // clang-format on
    V.reshaped().setLinSpaced(0, static_cast<GpuIndex>(P.cols() - 1));
    // Parallel graph information
    using SparseMatrixType = Eigen::SparseMatrix<GpuIndex, Eigen::ColMajor>;
    using TripletType      = Eigen::Triplet<GpuIndex, typename SparseMatrixType::StorageIndex>;
    SparseMatrixType G(T.cols(), P.cols());
    std::vector<TripletType> Gij{};
    for (auto e = 0; e < T.cols(); ++e)
    {
        for (auto ilocal = 0; ilocal < T.rows(); ++ilocal)
        {
            auto i = e;
            auto j = T(ilocal, e);
            Gij.push_back(TripletType{i, j, ilocal});
        }
    }
    G.setFromTriplets(Gij.begin(), Gij.end());
    CHECK(G.isCompressed());
    std::span<GpuIndex> vertexTetrahedronNeighbours{
        G.innerIndexPtr(),
        static_cast<std::size_t>(G.nonZeros())};
    std::span<GpuIndex> vertexTetrahedronPrefix{
        G.outerIndexPtr(),
        static_cast<std::size_t>(G.outerSize() + 1)};
    std::span<GpuIndex> vertexTetrahedronLocalVertexIndices{
        G.innerNonZeroPtr(),
        static_cast<std::size_t>(G.nonZeros())};
    std::vector<std::vector<GpuIndex>> partitions(P.cols());
    for (auto p = 0; p < partitions.size(); ++p)
        partitions[p].push_back(p);
    // Material parameters
    fem::Mesh<fem::Tetrahedron<1>, 3> mesh{P.cast<Scalar>(), T.cast<Index>()};
    GpuMatrixX wg           = fem::DeterminantOfJacobian<1>(mesh).cast<GpuScalar>();
    GpuMatrixX GP           = fem::ShapeFunctionGradients<1>(mesh).cast<GpuScalar>();
    auto constexpr Y        = 1e6;
    auto constexpr nu       = 0.45;
    auto const [mu, lambda] = physics::LameCoefficients(Y, nu);
    GpuMatrixX lame(2, T.cols());
    lame.row(0).array() = static_cast<GpuScalar>(mu);
    lame.row(1).array() = static_cast<GpuScalar>(lambda);
    // Problem parameters
    GpuMatrixX aext(3, P.cols());
    aext.colwise()    = Eigen::Vector<GpuScalar, 3>{GpuScalar{0}, GpuScalar{0}, GpuScalar{-9.81}};
    auto constexpr dt = GpuScalar{1e-2};
    auto constexpr substeps   = 1;
    auto constexpr iterations = 1;
    auto constexpr rhoC       = GpuScalar{0.95};
    // Act
    using pbat::gpu::vbd::VbdImpl;
    VbdImpl vbd{P, V, F, T};
    vbd.SetExternalAcceleration(aext);
    vbd.SetQuadratureWeights(wg);
    vbd.SetShapeFunctionGradients(GP);
    vbd.SetLameCoefficients(lame);
    vbd.SetVertexTetrahedronAdjacencyList(
        common::ToEigen(vertexTetrahedronNeighbours),
        common::ToEigen(vertexTetrahedronPrefix),
        common::ToEigen(vertexTetrahedronLocalVertexIndices));
    vbd.SetVertexPartitions(partitions);

    vbd.Step(dt, iterations, substeps, rhoC);

    // Assert
}
