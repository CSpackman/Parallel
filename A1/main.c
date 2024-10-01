#include <stdio.h>
#include <sys/sysinfo.h>
#include <time.h>

int main(int argc, char *argv[])
{
    printf("This system has %d processors configured and "
        "%d processors available.\n",
        get_nprocs_conf(), get_nprocs());


    struct timespec begin, end;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &begin);

// spawn threads to do work here

    clock_gettime(CLOCK_MONOTONIC, &end);

    elapsed = end.tv_sec - begin.tv_sec;
    elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("Elapsed Time: %f \n",elapsed);
    return 0;
}