import numpy as np
from numba import cuda

@cuda.jit
def merge_sorted_arrays(arr1, arr2, merged_array, n1, n2):

    idx = cuda.grid(1) # get thread id for 1d grid
    
    if idx < n1 + n2:
        # pointers to traverse both arrays
        i = 0
        j = 0

        # position in the merged array
        pos = 0  
        
        # comparing values in arrays and placing them in merged array
        # works as long as there are elements remaining in BOTH arrays
        while i < n1 and j < n2:
            if arr1[i] <= arr2[j]:
                merged_array[pos] = arr1[i]
                i += 1
            else:
                merged_array[pos] = arr2[j]
                j += 1
            pos += 1
        
        # need these while loops if either of the arrays run is fully processed
        # if any elements remain in arr1
        while i < n1:
            merged_array[pos] = arr1[i]
            i += 1
            pos += 1

        # if any elements remain in arr2
        while j < n2:
            merged_array[pos] = arr2[j]
            j += 1
            pos += 1

# host code
def merge_sorted(arr1, arr2):
    # array lengths
    n1, n2 = len(arr1), len(arr2)
    
    # device arrays (gpu)
    d_arr1 = cuda.to_device(arr1)
    d_arr2 = cuda.to_device(arr2)
    d_merged = cuda.device_array(n1 + n2, dtype=np.int32)
    
    # kernel launch
    threads_per_block = 32
    # this calc ensures proper division when n1 + n2 is not a multiple of threads_per_block
    blocks_per_grid = (n1 + n2 + threads_per_block - 1) // threads_per_block
    merge_sorted_arrays[blocks_per_grid, threads_per_block](d_arr1, d_arr2, d_merged, n1, n2)
    
    # copy result back to host
    return d_merged.copy_to_host()

# example
arr1 = np.array([1, 3, 5, 7, 9, 10, 11, 12, 15], dtype=np.int32)
arr2 = np.array([2, 4, 6, 8, 16, 17, 21], dtype=np.int32)

merged = merge_sorted(arr1, arr2)
print("Merged array:", merged)
