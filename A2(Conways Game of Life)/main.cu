#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

int *init_grid(int width, int height)
{
    // Plus one to account for ghost corners/rows/cols
    int new_width = width + 1;
    int new_height = height + 1;
    int *grid = new int[(new_width) * (new_height)];
    for (unsigned i = 0; i < (new_width * new_height); ++i)
    {
        grid[i] = rand() % 2;
    }
    return grid;
}

void print_grid(int *grid, int width, int height)
{

    for (int i = 0; i < width ; i++)
    {
        for (int j = 0; j < height; j++)
        {
            if (grid[i*j])
            {
                printf("\u2588");
            }
            if (!grid[i*j])
            {
                printf("  ");
            }
            
            // printf("%d", grid[i*j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    // 8 Digit seed
    srand(62006192);
    int *grid = init_grid(5, 5);

    for (size_t i = 0; i < (6 * 6); i++)
    {
        printf("%d", grid[i]);
    }
    printf("\n \n");
    print_grid(grid,6,6);

    return 1;
}