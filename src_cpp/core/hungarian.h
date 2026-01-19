#pragma once

#include <stdexcept>
#include <vector>

namespace hungarian
{
inline std::vector<int> solve_rectangular(const std::vector<std::vector<double>> &cost)
{
    const int num_rows = static_cast<int>(cost.size()) - 1;
    const int num_cols = static_cast<int>(cost.front().size()) - 1;
    const double kInf = 1e18;

    std::vector<double> u(num_rows + 1, 0.0);
    std::vector<double> v(num_cols + 1, 0.0);
    std::vector<int> p(num_cols + 1, 0);
    std::vector<int> way(num_cols + 1, 0);

    for (int i = 1; i <= num_rows; ++i)
    {
        p[0] = i;
        std::vector<double> minv(num_cols + 1, kInf);
        std::vector<char> used(num_cols + 1, false);
        int j0 = 0;
        do
        {
            used[j0] = true;
            const int i0 = p[j0];
            double delta = kInf;
            int j1 = 0;
            for (int j = 1; j <= num_cols; ++j)
            {
                if (used[j])
                {
                    continue;
                }
                const double cur = cost[i0][j] - u[i0] - v[j];
                if (cur < minv[j])
                {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta)
                {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= num_cols; ++j)
            {
                if (used[j])
                {
                    u[p[j]] += delta;
                    v[j] -= delta;
                }
                else
                {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do
        {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> assignment(num_rows, -1);
    for (int j = 1; j <= num_cols; ++j)
    {
        if (p[j] != 0)
        {
            assignment[p[j] - 1] = j - 1;
        }
    }

    return assignment;
}

inline std::vector<int> solve(const std::vector<std::vector<float>> &cost_matrix)
{
    const int num_rows = static_cast<int>(cost_matrix.size());
    if (num_rows == 0)
    {
        return {};
    }
    const int num_cols = static_cast<int>(cost_matrix.front().size());
    if (num_cols == 0)
    {
        return std::vector<int>(num_rows, -1);
    }
    for (const auto &row : cost_matrix)
    {
        if (static_cast<int>(row.size()) != num_cols)
        {
            throw std::invalid_argument("Cost matrix rows must have equal length.");
        }
    }

    if (num_rows <= num_cols)
    {
        std::vector<std::vector<double>> cost(num_rows + 1, std::vector<double>(num_cols + 1, 0.0));
        for (int i = 0; i < num_rows; ++i)
        {
            for (int j = 0; j < num_cols; ++j)
            {
                cost[i + 1][j + 1] = static_cast<double>(cost_matrix[i][j]);
            }
        }
        return solve_rectangular(cost);
    }

    std::vector<std::vector<double>> cost(num_cols + 1, std::vector<double>(num_rows + 1, 0.0));
    for (int i = 0; i < num_rows; ++i)
    {
        for (int j = 0; j < num_cols; ++j)
        {
            cost[j + 1][i + 1] = static_cast<double>(cost_matrix[i][j]);
        }
    }
    const std::vector<int> assignment_t = solve_rectangular(cost);
    std::vector<int> assignment(num_rows, -1);
    for (int col = 0; col < num_cols; ++col)
    {
        const int row = assignment_t[col];
        if (row >= 0 && row < num_rows)
        {
            assignment[row] = col;
        }
    }

    return assignment;
}
} // namespace hungarian
