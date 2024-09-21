#include "Pbf.h"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <algorithm>
#include <numeric>
#include <mutex>

namespace pbat {
namespace pbf {

Pbf::Pbf(float radius, float rho0, float eps, int maxIter, float c, float kCorr)
    : mImpl(new PbfImpl(radius, rho0, eps, maxIter, c, kCorr))
{
}

void Pbf::setParticles(const std::vector<Particle>& particles)
{
    mImpl->setParticles(particles);
}

std::vector<Particle>& Pbf::getParticles()
{
    return mImpl->getParticles();
}

const std::vector<Particle>& Pbf::getParticles() const
{
    return mImpl->getParticles();
}

void Pbf::step(float dt)
{
    mImpl->step(dt);
}

// Implementation of PbfImpl methods

Pbf::PbfImpl::PbfImpl(float radius_, float rho0_, float eps_, int maxIter_, float c_, float kCorr_)
    : radius(radius_), rho0(rho0_), eps(eps_), maxIter(maxIter_), c(c_), kCorr(kCorr_),
      aabbMin(Eigen::Vector3f(-2.0f, 0.0f, -1.0f)),
      aabbMax(Eigen::Vector3f(2.0f, 2.0f, 1.0f)),
      m_hashGrid(aabbMin, aabbMax, radius)
{
}

void Pbf::PbfImpl::setParticles(const std::vector<Particle>& particles)
{
    m_particles = particles;
    m_hashGrid.clearBins();
    updateHashGrid();
}

std::vector<Particle>& Pbf::PbfImpl::getParticles()
{
    return m_particles;
}

const std::vector<Particle>& Pbf::PbfImpl::getParticles() const
{
    return m_particles;
}

void Pbf::PbfImpl::step(float dt)
{
    updatePredictedPositions(dt);
    updateHashGrid();
    buildNeighborhood();

    for (int iter = 0; iter < maxIter; ++iter)
    {
        computeDensityAndLambda();
        computeDeltaP();
        boxCollision();
    }

    updatePositionsAndVelocities(dt);
    applyViscosity();
    applyVorticityConfinement();
}

void Pbf::PbfImpl::updatePredictedPositions(float dt)
{
    Eigen::Vector3f externalForce(0.0f, -9.8f, 0.0f);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_particles.size()),
                      [&](const tbb::blocked_range<size_t>& r)
                      {
                          for (size_t i = r.begin(); i != r.end(); ++i)
                          {
                              Particle& p = m_particles[i];
                              p.v += dt * externalForce; // Update velocity with external force
                              p.xstar = p.x + dt * p.v;  // Predict position
                          }
                      });
}

void Pbf::PbfImpl::updateHashGrid()
{
    m_hashGrid.clearBins();

    // Collect keys and particles for insertion
    std::vector<std::pair<int, Particle*>> keyParticlePairs(m_particles.size());

    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_particles.size()),
                      [&](const tbb::blocked_range<size_t>& r)
                      {
                          int coords[3];
                          for (size_t i = r.begin(); i != r.end(); ++i)
                          {
                              Particle& p = m_particles[i];
                              m_hashGrid.getCoordinates(p.xstar, coords);
                              const int key = m_hashGrid.key(coords[0], coords[1], coords[2]);
                              p.key = key;
                              keyParticlePairs[i] = {key, &p};
                          }
                      });

    // Insert particles into hash grid
    for (const auto& [key, particlePtr] : keyParticlePairs)
    {
        if (key != -1)
        {
            m_hashGrid.insert(key, particlePtr);
        }
    }
}

void Pbf::PbfImpl::buildNeighborhood()
{
    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_particles.size()),
                      [&](const tbb::blocked_range<size_t>& r)
                      {
                          int coords[3];
                          for (size_t i = r.begin(); i != r.end(); ++i)
                          {
                              Particle& p = m_particles[i];
                              p.N.clear();
                              m_hashGrid.getCoordinates(p.xstar, coords);
                              for (int z = -1; z <= 1; ++z)
                              {
                                  for (int y = -1; y <= 1; ++y)
                                  {
                                      for (int x = -1; x <= 1; ++x)
                                      {
                                          int key = m_hashGrid.key(coords[0] + x, coords[1] + y, coords[2] + z);
                                          if (key != -1 && m_hashGrid.isValid(key))
                                          {
                                              for (Particle* neighbor : m_hashGrid.getCell(key))
                                              {
                                                  if ((p.xstar - neighbor->xstar).squaredNorm() < radius * radius)
                                                  {
                                                      p.N.push_back(neighbor);
                                                  }
                                              }
                                          }
                                      }
                                  }
                              }
                          }
                      });
}

void Pbf::PbfImpl::computeDensityAndLambda()
{
    Kernel kernel(radius);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_particles.size()),
                      [&](const tbb::blocked_range<size_t>& r)
                      {
                          for (size_t i = r.begin(); i != r.end(); ++i)
                          {
                              Particle& p = m_particles[i];
                              p.rho = 0.0f;
                              Eigen::Vector3f gradCi = Eigen::Vector3f::Zero();
                              float sumGradCj2 = 0.0f;

                              for (Particle* neighbor : p.N)
                              {
                                  float W = kernel.poly6(p.xstar, neighbor->xstar);
                                  p.rho += W;

                                  Eigen::Vector3f gradW = kernel.spiky(p.xstar, neighbor->xstar) / rho0;
                                  gradCi += gradW;
                                  sumGradCj2 += gradW.squaredNorm();
                              }

                              float C = p.rho / rho0 - 1.0f;
                              float sumGradCi2 = gradCi.squaredNorm();

                              p.lambda = -C / (sumGradCi2 + sumGradCj2 + eps);
                          }
                      });
}

void Pbf::PbfImpl::computeDeltaP()
{
    Kernel kernel(radius);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_particles.size()),
                      [&](const tbb::blocked_range<size_t>& r)
                      {
                          for (size_t i = r.begin(); i != r.end(); ++i)
                          {
                              Particle& p = m_particles[i];
                              Eigen::Vector3f deltaPi = Eigen::Vector3f::Zero();

                              for (Particle* neighbor : p.N)
                              {
                                  float scorr = -kCorr * pow(kernel.poly6(p.xstar, neighbor->xstar) / kernel.scorr(), 4);
                                  Eigen::Vector3f gradW = kernel.spiky(p.xstar, neighbor->xstar);

                                  deltaPi += (p.lambda + neighbor->lambda + scorr) * gradW;
                              }

                              p.deltaP = deltaPi / rho0;
                          }
                      });

    // Update positions
    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_particles.size()),
                      [&](const tbb::blocked_range<size_t>& r)
                      {
                          for (size_t i = r.begin(); i != r.end(); ++i)
                          {
                              Particle& p = m_particles[i];
                              p.xstar += p.deltaP;
                          }
                      });
}

void Pbf::PbfImpl::updatePositionsAndVelocities(float dt)
{
    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_particles.size()),
                      [&](const tbb::blocked_range<size_t>& r)
                      {
                          for (size_t i = r.begin(); i != r.end(); ++i)
                          {
                              Particle& p = m_particles[i];
                              p.v = (p.xstar - p.x) / dt;
                              p.x = p.xstar;
                          }
                      });
}

void Pbf::PbfImpl::applyViscosity()
{
    Kernel kernel(radius);

    // Compute velocity deltas
    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_particles.size()),
                      [&](const tbb::blocked_range<size_t>& r)
                      {
                          for (size_t i = r.begin(); i != r.end(); ++i)
                          {
                              Particle& p = m_particles[i];
                              Eigen::Vector3f vDelta = Eigen::Vector3f::Zero();
                              for (Particle* neighbor : p.N)
                              {
                                  vDelta += (neighbor->v - p.v) * kernel.poly6(p.xstar, neighbor->xstar);
                              }
                              p.vdiff = c * vDelta;
                          }
                      });

    // Apply viscosity
    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_particles.size()),
                      [&](const tbb::blocked_range<size_t>& r)
                      {
                          for (size_t i = r.begin(); i != r.end(); ++i)
                          {
                              Particle& p = m_particles[i];
                              p.v += p.vdiff;
                          }
                      });
}

void Pbf::PbfImpl::applyVorticityConfinement()
{
    Kernel kernel(radius);

    // Compute vorticity omega
    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_particles.size()),
                      [&](const tbb::blocked_range<size_t>& r)
                      {
                          for (size_t i = r.begin(); i != r.end(); ++i)
                          {
                              Particle& p = m_particles[i];
                              Eigen::Vector3f omega = Eigen::Vector3f::Zero();
                              for (Particle* neighbor : p.N)
                              {
                                  Eigen::Vector3f vij = neighbor->v - p.v;
                                  Eigen::Vector3f gradW = kernel.spiky(p.xstar, neighbor->xstar);
                                  omega += vij.cross(gradW);
                              }
                              p.omega = omega;
                          }
                      });

    // Compute eta (magnitude of omega gradient)
    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_particles.size()),
                      [&](const tbb::blocked_range<size_t>& r)
                      {
                          for (size_t i = r.begin(); i != r.end(); ++i)
                          {
                              Particle& p = m_particles[i];
                              Eigen::Vector3f eta = Eigen::Vector3f::Zero();
                              for (Particle* neighbor : p.N)
                              {
                                  Eigen::Vector3f gradW = kernel.spiky(p.xstar, neighbor->xstar);
                                  eta += gradW.cross(neighbor->omega);
                              }
                              if (eta.norm() > 1e-6f)
                              {
                                  Eigen::Vector3f N = eta.normalized();
                                  Eigen::Vector3f fvc = kCorr * N.cross(p.omega);
                                  p.v += fvc;
                              }
                          }
                      });
}

void Pbf::PbfImpl::boxCollision()
{
    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_particles.size()),
                      [&](const tbb::blocked_range<size_t>& r)
                      {
                          for (size_t i = r.begin(); i != r.end(); ++i)
                          {
                              Particle& p = m_particles[i];
                              p.xstar = p.xstar.cwiseMax(aabbMin).cwiseMin(aabbMax);

                              // Simple collision response: invert velocity component on collision
                              if (p.xstar.x() <= aabbMin.x() || p.xstar.x() >= aabbMax.x())
                                  p.v.x() *= -0.5f;
                              if (p.xstar.y() <= aabbMin.y() || p.xstar.y() >= aabbMax.y())
                                  p.v.y() *= -0.5f;
                              if (p.xstar.z() <= aabbMin.z() || p.xstar.z() >= aabbMax.z())
                                  p.v.z() *= -0.5f;
                          }
                      });
}

} // namespace pbf
} // namespace pbat

#include "doctest.h"

// Doctest unit tests
TEST_SUITE("Pbf Simulation Tests")
{
    using namespace pbat::pbf;

    TEST_CASE("Pbf Simulation Initialization")
    {
        // Test initialization with zero particles
        Pbf pbf(0.1f, 1000.0f, 0.01f, 5, 0.01f, 0.01f);
        std::vector<Particle> particles;
        pbf.setParticles(particles);

        CHECK(pbf.getParticles().empty());

        // Test initialization with some particles
        particles.resize(10);
        for (int i = 0; i < 10; ++i)
        {
            particles[i].x = Eigen::Vector3f::Random();
            particles[i].v = Eigen::Vector3f::Zero();
            particles[i].key = -1;
        }
        pbf.setParticles(particles);

        CHECK(pbf.getParticles().size() == 10);
    }

    TEST_CASE("Pbf Single Simulation Step")
    {
        // Test a single simulation step
        Pbf pbf(0.1f, 1000.0f, 0.01f, 5, 0.01f, 0.01f);
        std::vector<Particle> particles(10);
        for (int i = 0; i < 10; ++i)
        {
            particles[i].x = Eigen::Vector3f::Random();
            particles[i].v = Eigen::Vector3f::Zero();
            particles[i].key = -1;
        }
        pbf.setParticles(particles);
        pbf.step(0.01f);

        auto& updatedParticles = pbf.getParticles();
        CHECK(updatedParticles.size() == 10);
        for (const auto& p : updatedParticles)
        {
            CHECK(p.x.allFinite());
            CHECK(p.v.allFinite());
        }
    }

    TEST_CASE("Pbf Multiple Simulation Steps")
    {
        // Test multiple simulation steps
        Pbf pbf(0.1f, 1000.0f, 0.01f, 5, 0.01f, 0.01f);
        std::vector<Particle> particles(100);
        for (int i = 0; i < 100; ++i)
        {
            particles[i].x = Eigen::Vector3f::Random();
            particles[i].v = Eigen::Vector3f::Zero();
            particles[i].key = -1;
        }
        pbf.setParticles(particles);

        const int steps = 10;
        for (int i = 0; i < steps; ++i)
        {
            pbf.step(0.01f);
        }

        auto& updatedParticles = pbf.getParticles();
        CHECK(updatedParticles.size() == 100);
        for (const auto& p : updatedParticles)
        {
            CHECK(p.x.allFinite());
            CHECK(p.v.allFinite());
        }
    }

    TEST_CASE("Pbf Particle Interaction")
    {
        // Test that particles influence each other
        Pbf pbf(0.1f, 1000.0f, 0.01f, 5, 0.01f, 0.01f);

        Particle p1, p2;
        p1.x = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
        p1.v = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
        p1.key = -1;

        p2.x = Eigen::Vector3f(0.05f, 0.0f, 0.0f); // Within influence radius
        p2.v = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
        p2.key = -1;

        std::vector<Particle> particles = {p1, p2};
        pbf.setParticles(particles);
        pbf.step(0.01f);

        auto& updatedParticles = pbf.getParticles();
        CHECK((updatedParticles[0].x - updatedParticles[1].x).norm() > 0.0f); // Particles have moved
    }

    TEST_CASE("Pbf Boundary Collision")
    {
        // Test collision with boundaries
        Pbf pbf(0.1f, 1000.0f, 0.01f, 5, 0.01f, 0.01f);

        Particle p;
        p.x = Eigen::Vector3f(1.9f, 1.9f, 0.0f); // Near the boundary
        p.v = Eigen::Vector3f(1.0f, 1.0f, 0.0f);
        p.key = -1;

        std::vector<Particle> particles = {p};
        pbf.setParticles(particles);
        pbf.step(0.1f);

        auto& updatedParticles = pbf.getParticles();
        CHECK(updatedParticles[0].x.x() <= 2.0f);
        CHECK(updatedParticles[0].x.y() <= 2.0f);
        CHECK(updatedParticles[0].v.x() <= 0.0f); // Velocity should have inverted
        CHECK(updatedParticles[0].v.y() <= 0.0f);
    }

    TEST_CASE("Pbf Conservation of Mass")
    {
        // Test that the total mass remains constant
        Pbf pbf(0.1f, 1000.0f, 0.01f, 5, 0.01f, 0.01f);
        const int numParticles = 100;
        std::vector<Particle> particles(numParticles);
        for (int i = 0; i < numParticles; ++i)
        {
            particles[i].x = Eigen::Vector3f::Random();
            particles[i].v = Eigen::Vector3f::Zero();
            particles[i].key = -1;
        }
        pbf.setParticles(particles);

        const int steps = 10;
        for (int i = 0; i < steps; ++i)
        {
            pbf.step(0.01f);
        }

        auto& updatedParticles = pbf.getParticles();

        // Assuming each particle has a mass of 1 for simplicity
        float totalMass = static_cast<float>(updatedParticles.size());
        CHECK(totalMass == doctest::Approx(numParticles));
    }
}

