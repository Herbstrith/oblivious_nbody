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

#include "nbody_kernel.cuh"

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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



void init(int num_bodies) {
  // allocate cpu memory
  m_hPos = new float[num_bodies * 4];
  m_hVel = new float[num_bodies * 4];

  // 4 floats each for alignment reasons
  unsigned int memSize = sizeof( float) * 4 * num_bodies;

  // allocate gpu memory
  gpuErrchk(cudaMalloc((void**)&m_dVel[0], memSize));
  gpuErrchk(cudaMalloc((void**)&m_dVel[1], memSize));

  gpuErrchk(cudaMalloc((void**)&m_dPos[0], memSize));
  gpuErrchk(cudaMalloc((void**)&m_dPos[1], memSize));
  
}

// copy array from the host (CPU) to the device (GPU)
void copyArrayToDevice(float *device, const float *host, int num_bodies) {
  gpuErrchk(cudaMemcpy(device, host, num_bodies * 4 * sizeof(float), cudaMemcpyHostToDevice));
}


void randomizeBodies(float *pos, float *vel, int num_bodies) {
  int i = 0;
  int p_index = 0, v_index = 0;
  while (i < num_bodies) {
    float3 point;
    float3 velocity;
    
    point.x = rand();
    point.y = rand();
    point.z = rand();

    velocity.x = rand();
    velocity.y = rand();
    velocity.z = rand();
    
    pos[p_index++] = point.x;
    pos[p_index++] = point.y;
    pos[p_index++] = point.z;
    pos[p_index++] = 1.0f;  // this is the mass

    vel[v_index++] = velocity.x;
    vel[v_index++] = velocity.y;
    vel[v_index++] = velocity.z;
    vel[v_index++] = 1.0f; // inverse mass

    i++;
    
  }
  
}

void runNbodySimulation(int num_iterations, int num_bodies) {
  
  cudaEvent_t startEvent, stopEvent;
  cudaEvent_t startEventIteration, stopEventIteration;
  
  gpuErrchk(cudaEventRecord(startEvent, 0));
  printf("Starting iterations \n \n");
  for (int i =0; i < num_iterations; i++) {
    float milliseconds_iteration = 0;
    gpuErrchk(cudaEventRecord(startEventIteration, 0));
      printf("Start execution \n  \n \n ");

    calculate_forces(num_bodies, (float4*)m_dVel[0]); 

    gpuErrchk(cudaEventRecord(stopEventIteration, 0));
    gpuErrchk(cudaEventSynchronize(stopEventIteration));
    gpuErrchk(cudaEventElapsedTime(&milliseconds_iteration, startEventIteration, stopEventIteration));
    printf("%.3f iteration\n", milliseconds_iteration);
  }
  float milliseconds = 0;

  gpuErrchk(cudaEventRecord(stopEvent, 0));
  gpuErrchk(cudaEventSynchronize(stopEvent));
  gpuErrchk(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

  printf("%.3f gputime\n", milliseconds);
}


int main(int argc, char** argv) {
    int num_iterations = 1;
    int num_bodies = 1000;

    sscanf(argv[1], "%d", &num_bodies);
    sscanf(argv[2], "%d", &num_iterations);

    printf("initializing bodies \n");
    init(num_bodies);

    // Generate a random set of bodies (size numBodies)

    randomizeBodies(m_hPos, m_hVel, num_bodies);
    
    // Move data to GPU
    printf("Moving data to GPU \n");
    copyArrayToDevice(*m_dPos, m_hPos, num_bodies);
    copyArrayToDevice(*m_dVel, m_hVel, num_bodies);
    
    // run kernel
    printf("Kernel start \n");
    runNbodySimulation(num_iterations, num_bodies);
    // clean up and finish
    printf("Finishing execution\n");
    if (m_hPos)
      delete [] m_hPos;

    if (m_hVel)
      delete [] m_hVel;

    // clean up the gpu
    gpuErrchk(cudaDeviceReset());
    
    return 0;
}
