// HashGrid.h
#ifndef HASHGRID_H
#define HASHGRID_H

#include "Particle.h"
#include <vector>
#include <unordered_map>
#include <mutex>

namespace pbat {
namespace pbf {

class HashGrid
{
public:
    HashGrid(const Eigen::Vector3f& min, const Eigen::Vector3f& max, float cellSize);

    int key(int x, int y, int z) const;
    bool isValid(int key) const;

    void clearBins();
    void insert(int key, Particle* particle);
    std::vector<Particle*>& getCell(int key);

    void getCoordinates(const Eigen::Vector3f& pos, int coords[3]) const;

private:
    Eigen::Vector3i dims;
    Eigen::Vector3f min, max;
    float cellSize;
    std::unordered_map<int, std::vector<Particle*>> cells;
    std::mutex cellsMutex;
};

} // namespace pbf
} // namespace pbat

#endif // HASHGRID_H
