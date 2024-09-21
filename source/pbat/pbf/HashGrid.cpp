#include "HashGrid.h"


namespace pbat {
namespace pbf {

HashGrid::HashGrid(const Eigen::Vector3f& min_, const Eigen::Vector3f& max_, float cellSize_)
    : min(min_), max(max_), cellSize(cellSize_)
{
    dims = ((max - min) / cellSize).cast<int>();
}

int HashGrid::key(int x, int y, int z) const
{
    if (x < 0 || x >= dims.x() || y < 0 || y >= dims.y() || z < 0 || z >= dims.z())
        return -1;
    return x + dims.x() * (y + dims.y() * z);
}

bool HashGrid::isValid(int key) const
{
    return cells.find(key) != cells.end();
}

void HashGrid::clearBins()
{
    std::lock_guard<std::mutex> lock(cellsMutex);
    cells.clear();
}

void HashGrid::insert(int key, Particle* particle)
{
    std::lock_guard<std::mutex> lock(cellsMutex);
    cells[key].push_back(particle);
}

std::vector<Particle*>& HashGrid::getCell(int key)
{
    return cells[key];
}

void HashGrid::getCoordinates(const Eigen::Vector3f& pos, int coords[3]) const
{
    Eigen::Vector3f posRel = pos - min;
    coords[0] = static_cast<int>(posRel.x() / cellSize);
    coords[1] = static_cast<int>(posRel.y() / cellSize);
    coords[2] = static_cast<int>(posRel.z() / cellSize);
}

} // namespace pbf
} // namespace pbat

#include "doctest.h"

// Doctest unit tests
TEST_CASE("HashGrid Functionality")
{
    using namespace pbat::pbf;

    Eigen::Vector3f min(-1.0f, -1.0f, -1.0f);
    Eigen::Vector3f max(1.0f, 1.0f, 1.0f);
    float cellSize = 0.5f;
    HashGrid hashGrid(min, max, cellSize);

    Particle p1;
    p1.xstar = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    int coords[3];
    hashGrid.getCoordinates(p1.xstar, coords);
    int key = hashGrid.key(coords[0], coords[1], coords[2]);

    hashGrid.insert(key, &p1);
    CHECK(hashGrid.isValid(key));

    auto& cell = hashGrid.getCell(key);
    CHECK(cell.size() == 1);
    CHECK(cell[0] == &p1);

    hashGrid.clearBins();
    CHECK(!hashGrid.isValid(key));
}
