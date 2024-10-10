#include <iostream>
#include <cstdlib>
#include <ctime>

// Define the dimensions of the grid
#define GRIDWIDTH 4
#define GRIDHEIGHT 4

// Function to initialize the grid with random live/dead cells
int *init_grid(int width, int height)
{
    // Allocate memory for the grid
    int *grid = new int[width * height];
    
    // Iterate through each cell in the grid
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            // Randomly set each cell to either alive (1) or dead (0)
            grid[i * width + j] = rand() % 2;
        }
    }
    return grid;
}

// Function to print the current state of the grid
void print_grid(int *grid, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            // Print '█' for live cells and ' ' for dead cells
            printf(grid[i * width + j] ? "█" : " ");
        }
        printf("\n");
    }
}

// Function to calculate the next generation of the grid on CPU
void calc_new_grid_cpu(int *oldGrid, int *newGrid, int width, int height)
{
    // Iterate through each cell in the grid
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int index = row * width + col;
            int liveNeighbors = 0;

            // Check all 8 neighboring cells
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    // Skip the cell itself
                    if (i == 0 && j == 0) continue;

                    // Calculate neighbor coordinates with wrapping
                    int newRow = (row + i + height) % height;
                    int newCol = (col + j + width) % width;

                    // Count live neighbors
                    liveNeighbors += oldGrid[newRow * width + newCol];
                }
            }

            // Apply Conway's Game of Life rules
            if (oldGrid[index] == 1) {
                // Cell is currently alive
                newGrid[index] = (liveNeighbors == 2 || liveNeighbors == 3) ? 1 : 0;
            } else {
                // Cell is currently dead
                newGrid[index] = (liveNeighbors == 3) ? 1 : 0;
            }
        }
    }
}

int main()
{
    // Seed the random number generator
    srand(time(NULL));

    // Initialize the grid
    int *grid = init_grid(GRIDWIDTH, GRIDHEIGHT);
    
    // Allocate memory for the new grid
    int *newGrid = new int[GRIDWIDTH * GRIDHEIGHT];

    // Print the initial state of the grid
    printf("Initial Grid:\n");
    print_grid(grid, GRIDWIDTH, GRIDHEIGHT);
    printf("\n");

    // Calculate the next generation
    calc_new_grid_cpu(grid, newGrid, GRIDWIDTH, GRIDHEIGHT);

    // Print the grid after one generation
    printf("Grid after one generation:\n");
    print_grid(newGrid, GRIDWIDTH, GRIDHEIGHT);

    // Free allocated memory
    delete[] grid;
    delete[] newGrid;

    return 0;
}