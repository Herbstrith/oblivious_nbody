#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <memory.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include "nbody_kernel.cu"

// for time measurement
clock_t start, end;



// for time measurement
static double gettime(void){
  struct timeval tr;
  gettimeofday(&tr,NULL);
  return (double)tr.tv_sec+(double)tr.tv_usec/1000000;
}



int main( int argc, char** argv) 
{
    int num_iterations = 1;
    int num_bodies = 1000;

    sscanf (argv[1],"%d",&num_bodies);
    sscanf (argv[2],"%d",&num_iterations);

    

    // Generate a random set of bodies (size numBodies

    // Move data to GPU

    // run kernel

    // clean up and finish

    return 0;
}
