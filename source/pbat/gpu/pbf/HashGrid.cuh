// HashGrid.cuh
#ifndef HASHGRID_CUH
#define HASHGRID_CUH

#include "HashGrid.h"

inline HashGrid::HashGrid(float3 min_, float3 max_, float cellSize_)
    : min(min_), max(max_), cellSize(cellSize_)
{
    dims.x = static_cast<int>((max.x - min.x) / cellSize);
    dims.y = static_cast<int>((max.y - min.y) / cellSize);
    dims.z = static_cast<int>((max.z - min.z) / cellSize);
    cells.resize(dims.x * dims.y * dims.z);
}

inline int HashGrid::key(int x, int y, int z) const
{
    if (x < 0 || x >= dims.x || y < 0 || y >= dims.y || z < 0 || z >= dims.z)
        return -1;
    return x + dims.x * (y + dims.y * z);
}

inline bool HashGrid::isValid(int key) const
{
    return key >= 0 && key < cells.size();
}

inline void HashGrid::clear()
{
    for (auto& cell : cells)
    {
        cell.clear();
    }
}

inline void HashGrid::insert(int key, int index)
{
    if (isValid(key))
    {
        cells[key].push_back(index);
    }
}

inline const std::vector<int>& HashGrid::getCell(int key) const
{
    return cells[key];
}

inline void HashGrid::getCoordinates(const float3& pos, int coords[3]) const
{
    coords[0] = static_cast<int>((pos.x - min.x) / cellSize);
    coords[1] = static_cast<int>((pos.y - min.y) / cellSize);
    coords[2] = static_cast<int>((pos.z - min.z) / cellSize);
}

#endif // HASHGRID_CUH
