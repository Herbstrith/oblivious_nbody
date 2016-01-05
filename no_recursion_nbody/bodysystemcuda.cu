/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <helper_cuda.h>
#include <math.h>

#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "bodysystem.h"

__constant__ float softeningSquared;
__constant__ double softeningSquared_fp64;

cudaError_t setSofteningSquared(float softeningSq)
{
    return cudaMemcpyToSymbol(softeningSquared,
                              &softeningSq,
                              sizeof(float), 0,
                              cudaMemcpyHostToDevice);
}

cudaError_t setSofteningSquared(double softeningSq)
{
    return cudaMemcpyToSymbol(softeningSquared_fp64,
                              &softeningSq,
                              sizeof(double), 0,
                              cudaMemcpyHostToDevice);
}

template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

template<typename T>
__device__ T rsqrt_T(T x)
{
    return rsqrt(x);
}

template<>
__device__ float rsqrt_T<float>(float x)
{
    return rsqrtf(x);
}

template<>
__device__ double rsqrt_T<double>(double x)
{
    return rsqrt(x);
}


// Macros to simplify shared memory addressing
#define SX(i) sharedPos[i+blockDim.x*threadIdx.y]
// This macro is only used when multithreadBodies is true (below)
#define SX_SUM(i,j) sharedPos[i+blockDim.x*j]

template <typename T>
__device__ T getSofteningSquared()
{
    return softeningSquared;
}
template <>
__device__ double getSofteningSquared<double>()
{
    return softeningSquared_fp64;
}

template <typename T>
struct DeviceData
{
    T *dPos[2]; // mapped host pointers
    T *dVel;
    cudaEvent_t  event;
    unsigned int offset;
    unsigned int numBodies;
};


template <typename T>
__device__ typename vec3<T>::Type
bodyBodyInteraction(typename vec3<T>::Type ai,
                    typename vec4<T>::Type bi,
                    typename vec4<T>::Type bj)
{
    typename vec3<T>::Type r;

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    T distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += getSofteningSquared<T>();

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    T invDist = rsqrt_T(distSqr);
    T invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    T s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

template <typename T>
__device__ typename vec3<T>::Type
computeBodyAccel(typename vec4<T>::Type bodyPos,
                 typename vec4<T>::Type *positions,
                 int numTiles)
{
    typename vec4<T>::Type *sharedPos = SharedMemory<typename vec4<T>::Type>();

    typename vec3<T>::Type acc = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < numTiles; tile++)
    {
        sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];

        __syncthreads();

        // This is the "tile_calculation" from the GPUG3 article.
#pragma unroll 128

        for (unsigned int counter = 0; counter < blockDim.x; counter++)
        {
            acc = bodyBodyInteraction<T>(acc, bodyPos, sharedPos[counter]);
        }

        __syncthreads();
    }

    return acc;
}

template<typename T>
__global__ void
integrateBodies(typename vec4<T>::Type *__restrict__ newPos,
                typename vec4<T>::Type *__restrict__ oldPos,
                typename vec4<T>::Type *vel,
                unsigned int deviceOffset, unsigned int deviceNumBodies,
                float deltaTime, float damping, int numTiles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= deviceNumBodies)
    {
        return;
    }

    typename vec4<T>::Type position = oldPos[deviceOffset + index];

    typename vec3<T>::Type accel = computeBodyAccel<T>(position,
                                                       oldPos,
                                                       numTiles);

    // acceleration = force / mass;
    // new velocity = old velocity + acceleration * deltaTime
    // note we factor out the body's mass from the equation, here and in bodyBodyInteraction
    // (because they cancel out).  Thus here force == acceleration
    typename vec4<T>::Type velocity = vel[deviceOffset + index];

    velocity.x += accel.x * deltaTime;
    velocity.y += accel.y * deltaTime;
    velocity.z += accel.z * deltaTime;

    velocity.x *= damping;
    velocity.y *= damping;
    velocity.z *= damping;

    // new position = old position + velocity * deltaTime
    position.x += velocity.x * deltaTime;
    position.y += velocity.y * deltaTime;
    position.z += velocity.z * deltaTime;

    // store new position and velocity
    newPos[deviceOffset + index] = position;
    vel[deviceOffset + index]    = velocity;
}


/* ----------------------------------------------- half the work --------------------------------------------*/


template<typename T>
__device__ void
CalculateForces(typename vec3<T>::Type index, typename vec4<T>::Type *__restrict__ oldPos, int numTiles, unsigned int deviceOffset, unsigned int deviceNumBodies, float deltaTime, float damping, typename vec4<T>::Type *vel)
{

  // work on the tile
  for (int tile_i = 0; tile_i < numTiles; tile_i++)
  {
    int index_i = (index.x - tile_i - index.y)*deviceNumBodies + (index.y  + tile_i);
    typename vec4<T>::Type pos_i = oldPos[index_i];
    
    for (int tile_j = tile_i+1; tile_j < numTiles; tile_j++) 
    {
      int index_j = (index.x - tile_j - index.y)*deviceNumBodies + (index.y  + tile_j);
      typename vec4<T>::Type pos_j = oldPos[index_j];

      typename vec3<T>::Type accel = {0.0f, 0.0f, 0.0f};

      //aceleration, body 1 , body 2
      accel = bodyBodyInteraction<T>(accel, pos_i, pos_j);

      // acceleration = force / mass;
      // new velocity = old velocity + acceleration * deltaTime
      // note we factor out the body's mass from the equation, here and in bodyBodyInteraction
      // (because they cancel out).  Thus here force == acceleration
      typename vec4<T>::Type velocity = vel[deviceOffset + index_i];
    
      velocity.x += accel.x * deltaTime;
      velocity.y += accel.y * deltaTime;
      velocity.z += accel.z * deltaTime;

      velocity.x *= damping;
      velocity.y *= damping;
      velocity.z *= damping;

      // new position = old position + velocity * deltaTime
      pos_i.x += velocity.x * deltaTime;
      pos_i.y += velocity.y * deltaTime;
      pos_i.z += velocity.z * deltaTime;
    
      // we also update the second particle with the inverse force (velocity) by the principle of newtons third law
      pos_j.x += -velocity.x * deltaTime;
      pos_j.y += -velocity.y * deltaTime;
      pos_j.z += -velocity.z * deltaTime;
    }
    
  }
  
}

// global variable   -- not the best place for it to be --
__device__ int diagonalCounter_i;
__device__ int block_atomic;  
__device__ bool stillWorking;
__device__ int actualNumBlocks;
/* Integrate the bodies by doing only half the works thanks to newtons third law : by Herbstrith */
template<typename T>
__global__ void
integrateBodiesHalfWork(typename vec4<T>::Type *__restrict__ newPos,
                        typename vec4<T>::Type *__restrict__ oldPos,
                        typename vec4<T>::Type *vel,
                        unsigned int deviceOffset, unsigned int deviceNumBodies,
                        float deltaTime, float damping, int numTiles,
                        int blockSize, int numBlocks)
{
  // will be shared between the threads
  stillWorking = true;
  block_atomic = 0;
  actualNumBlocks = numBlocks;
  diagonalCounter_i = deviceNumBodies;
  __shared__ int diagonalCounter_j;
  diagonalCounter_j = 0;
  __shared__ bool blockEnd;
  blockEnd = false;
  
  int j_end = blockSize * blockIdx.x + blockSize + (numBlocks*blockSize*diagonalCounter_j);
  j_end = ( j_end > deviceNumBodies) ? deviceNumBodies : j_end;
  
  // x = i start, y = i end, z = j start, w = j end
  __shared__ typename vec4<T>::Type workRange;
     
  workRange.x = deviceNumBodies - diagonalCounter_i;
  workRange.y = deviceNumBodies -diagonalCounter_i;    
  workRange.z =  (blockSize * blockIdx.x) +(numBlocks*blockSize*diagonalCounter_j) ;
  workRange.w = j_end;
  
  
  while ( stillWorking ) {
    
    typename vec3<T>::Type index;
    index.x = (workRange.x - numTiles) - threadIdx.x;
    index.y = (workRange.z + numTiles) + threadIdx.x;
   
    if(index.x > 0 || index.y < deviceNumBodies) {
      // calculate  the index particle tile  on the system... each thread will work on numTiles particles
      CalculateForces<T>(index,oldPos, numTiles, deviceOffset, deviceNumBodies, deltaTime, damping, vel);
    } 
    

    //sync threads
    __syncthreads();
    
    if (threadIdx.x  == 0) {
            
     diagonalCounter_j++;            
     j_end = blockSize * blockIdx.x + blockSize + (numBlocks*blockSize*diagonalCounter_j);
     j_end = (j_end > deviceNumBodies) ? deviceNumBodies - diagonalCounter_i : j_end;
     
     workRange.x = deviceNumBodies - diagonalCounter_i;
     workRange.y = deviceNumBodies -diagonalCounter_i;
     workRange.z =  (blockSize * blockIdx.x) +(numBlocks*blockSize*diagonalCounter_j) ;
     workRange.w = j_end;

     //  we reached the end of the diagonal line
     if(workRange.z >= (deviceNumBodies - diagonalCounter_i)) {
       diagonalCounter_j = 0;
       atomicAdd(&block_atomic, 1);
       //busy waiting
       while (block_atomic < actualNumBlocks) continue;

       if (blockIdx.x == 0) {
         atomicSub(&diagonalCounter_i, 1);
         if(diagonalCounter_i < 0) {
           atomicSub(&actualNumBlocks, 1);
           stillWorking = false;
         }
         block_atomic = 0;
       }
            
       int j_end = blockSize * blockIdx.x + blockSize + (numBlocks*blockSize*diagonalCounter_j);
       j_end = (j_end > deviceNumBodies) ? deviceNumBodies - diagonalCounter_i : j_end;
     
       workRange.x = deviceNumBodies - diagonalCounter_i;
       workRange.y = deviceNumBodies -diagonalCounter_i;    
       workRange.z =  (blockSize * blockIdx.x) +(numBlocks*blockSize*diagonalCounter_j) ;
       workRange.w = j_end;
       
       // this block wont work anymore
       if(workRange.z >= (deviceNumBodies - diagonalCounter_i) && blockIdx.x != 0) {
         atomicSub(&actualNumBlocks, 1);
         blockEnd = true;
       }
       
     }
       
    }
    
    __syncthreads();

    //end the idle block threads
    if (blockEnd) {
      return;
    }

  }

}



/* ----------------------------------------------- half the work end --------------------------------------------*/



template <typename T>
void integrateNbodySystem(DeviceData<T> *deviceData,
                          cudaGraphicsResource **pgres,
                          unsigned int currentRead,
                          float deltaTime,
                          float damping,
                          unsigned int numBodies,
                          unsigned int numDevices,
                          int blockSize,
                          bool bUsePBO)
{
    if (bUsePBO)
    {
        checkCudaErrors(cudaGraphicsResourceSetMapFlags(pgres[currentRead], cudaGraphicsMapFlagsReadOnly));
        checkCudaErrors(cudaGraphicsResourceSetMapFlags(pgres[1-currentRead], cudaGraphicsMapFlagsWriteDiscard));
        checkCudaErrors(cudaGraphicsMapResources(2, pgres, 0));
        size_t bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&(deviceData[0].dPos[currentRead]), &bytes, pgres[currentRead]));
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&(deviceData[0].dPos[1-currentRead]), &bytes, pgres[1-currentRead]));
    }

    for (unsigned int dev = 0; dev != numDevices; dev++)
    {
        if (numDevices > 1)
        {
            cudaSetDevice(dev);
        }

        int numBlocks = (deviceData[dev].numBodies + blockSize-1) / blockSize;
        int numTiles = (numBodies + blockSize - 1) / blockSize;
        numTiles = 10;
        int sharedMemSize = blockSize * 4 * sizeof(T); // 4 floats for pos

        integrateBodiesHalfWork<T><<< numBlocks, blockSize, sharedMemSize >>>
            ((typename vec4<T>::Type *)deviceData[dev].dPos[1-currentRead],
             (typename vec4<T>::Type *)deviceData[dev].dPos[currentRead],
             (typename vec4<T>::Type *)deviceData[dev].dVel,
             deviceData[dev].offset, deviceData[dev].numBodies,
             deltaTime, damping, numTiles,
             blockSize, numBlocks);
        
        if (numDevices > 1)
        {
            checkCudaErrors(cudaEventRecord(deviceData[dev].event));
            // MJH: Hack on older driver versions to force kernel launches to flush!
            cudaStreamQuery(0);
        }

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    if (numDevices > 1)
    {
        for (unsigned int dev = 0; dev < numDevices; dev++)
        {
            checkCudaErrors(cudaEventSynchronize(deviceData[dev].event));
        }
    }

    if (bUsePBO)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(2, pgres, 0));
    }
}


// Explicit specializations needed to generate code
template void integrateNbodySystem<float>(DeviceData<float> *deviceData,
                                          cudaGraphicsResource **pgres,
                                          unsigned int currentRead,
                                          float deltaTime,
                                          float damping,
                                          unsigned int numBodies,
                                          unsigned int numDevices,
                                          int blockSize,
                                          bool bUsePBO);

template void integrateNbodySystem<double>(DeviceData<double> *deviceData,
                                           cudaGraphicsResource **pgres,
                                           unsigned int currentRead,
                                           float deltaTime,
                                           float damping,
                                           unsigned int numBodies,
                                           unsigned int numDevices,
                                           int blockSize,
                                           bool bUsePBO);
