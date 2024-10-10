#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

int *init_grid(int width, int height)
{
    // Plus one to account for ghost corners/rows/cols
    int *grid = new int[(width) * (height)];
    for (unsigned i = 0; i < (width * height); ++i)
    {
        grid[i] = rand() % 2;
    }
    return grid;
}

// __global__ void calc_new_grid(int *A, int *B,int numARows, int numAColumns,int numBRows, int numBColumns,)
// {
// 	// If Corner 
//         //If Top Left
//         // If Top Right
//         //If Bottom Left
//         //If Bottom Right 
//     //If Ghost Row
//         //Top
//         //Bottom
//     //If Ghost Col
//         //Right 
//         //Left
// }

void print_grid(int *grid, int width, int height)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
           
            if (grid[i+j])
            {
                printf("\u2588");
                // printf("1");
            }
            if (!grid[i+j])
            {
                printf("  ");
                // printf("0");
            }
            
            // printf("%d", grid[i*j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    int GRIDWITH = 5;
    int GRIDHEIGHT = 5;
    // 8 Digit seed
    srand(92006191);

    int *grid = init_grid(GRIDWITH, GRIDHEIGHT);

    for (size_t i = 0; i < (GRIDWITH * GRIDHEIGHT); i++)
    {
        printf("%d", grid[i]);
    }
    printf("\n \n");
    print_grid(grid,GRIDWITH,GRIDHEIGHT);

    return 1;
}