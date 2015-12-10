#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <memory.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "nbody_kernel.cu"

// for time measurement
clock_t start, end;

// cpu data
float* m_hPos;
float* m_hVel;

// gpu data
float* m_dPos[2];
float* m_dVel[2];

// for time measurement
static double gettime(void) {
  struct timeval tr;
  gettimeofday(&tr, NULL);
  return (double)tr.tv_sec+(double)tr.tv_usec/1000000;
}


void init(int num_bodies) {
  // allocate cpu memory
  m_hPos = new float[num_bodies * 4];
  m_hVel = new float[num_bodies * 4];

  // 4 floats each for alignment reasons
  unsigned int memSize = sizeof( float) * 4 * num_bodies;

  // allocate gpu memory
  cudaMalloc((void**)&m_dVel[0], memSize);
  cudaMalloc((void**)&m_dVel[1], memSize);

  cudaMalloc((void**)&m_dPos[0], memSize);
  cudaMalloc((void**)&m_dPos[1], memSize);
  
}

// copy array from the host (CPU) to the device (GPU)
void copyArrayToDevice(float *device, const float *host, int num_bodies) {
  cudaMemcpy(device, host, num_bodies * 4 * sizeof(float), cudaMemcpyHostToDevice);
}


int main(int argc, char** argv) {
    int num_iterations = 1;
    int num_bodies = 1000;

    sscanf(argv[1], "%d", &num_bodies);
    sscanf(argv[2], "%d", &num_iterations);


    init(num_bodies);

    // Generate a random set of bodies (size numBodies

    // Move data to GPU

    // run kernel

    // clean up and finish

    return 0;
}
