// HashGrid.h
#ifndef HASHGRID_H
#define HASHGRID_H

#include <cuda_runtime.h>
#include <vector>

class HashGrid
{
public:
    HashGrid(float3 min, float3 max, float cellSize);

    int key(int x, int y, int z) const;
    bool isValid(int key) const;

    void clear();
    void insert(int key, int index);
    const std::vector<int>& getCell(int key) const;

    void getCoordinates(const float3& pos, int coords[3]) const;

private:
    int3 dims;
    float3 min, max;
    float cellSize;
    std::vector<std::vector<int>> cells;
};

#endif // HASHGRID_H
