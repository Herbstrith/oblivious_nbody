#include <math.h>

#include "nbody_kernel.cuh"

// calculate_forces kernel function ... here is where we start

__global__ void calculate_forces( int nbodies, float4 *bodies)
{
  triangle <<< 1, 1 >>> (0, nbodies, bodies);
}


__global__ void triangle(int n0, int n1, float4 *bodies)
{
  int dn = n1 - n0;
  if(dn > 1){
    int nm = n0 + dn/2;

    triangle <<< 1, 1 >>> (n0, n1, bodies);  //spawn new thread
    triangle <<< 1, 1 >>>(nm,n1,bodies);  //we might try creating a __device__ version so we dont spawn this one, and instead keep working on this thread
    rect <<< 1, 1 >>> (n0, nm, nm, n1, bodies);//spawn new thread
  }
}

// rectangle kernel function ( without coarsening )


__global__ void rect(int i0, int i1, int j0, int j1, float4 *bodies)
{
  int di = i1 - i0;
  int dj = j1 -j0;

  if(di > 1 && dj >1){
    int im = i0 + di/2;
    int jm = j0 + dj/2;

    rect <<< 1, 1 >>>(i0, im, j0, jm, bodies); //spawn new threads
    rect <<< 1, 1 >>>(im, i1, jm, j1, bodies); //we might try creating a __device__ version so we dont spawn this one, and instead keep working on this thread

    rect <<< 1, 1 >>>(i0, im, jm, j1, bodies); //spawn new threads
    rect <<< 1, 1 >>>(im, i1, j0, jm, bodies); //spawn new threads
    
  } else {
    if (di > 0 && dj >0){
      float3 resultingForce;
      resultingForce = calculate_force(resultingForce, bodies[i0], bodies[j0]);
      //add_force(&bodies[di],resultingForce);
      bodies[di].x += resultingForce.x;
      bodies[di].y += resultingForce.y;
      bodies[di].z += resultingForce.z;
      
      //add_force(&bodies[dj], -resultingForce);
      bodies[dj].x += -resultingForce.x;
      bodies[dj].y += -resultingForce.y;
      bodies[dj].z += -resultingForce.z;
    }
  }
}


//calculate_force: body x body interaction (from the gems3 algorithm : bodyBodyInteraction kernel function)

__device__ float3 calculate_force(float3 ai, float4 bi, float4 bj) {
  float3 r;

  // r_ij  [3 FLOPS]
  r.x = bi.x - bj.x;
  r.y = bi.y - bj.y;
  r.z = bi.z - bj.z;

  // distSqr = dot(r_ij, r_ij)  [6 FLOPS]
  float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;

  // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
  float distSixth = distSqr * distSqr * distSqr;
  float invDistCube = 1.0f / sqrtf(distSixth);
    
  // s = m_j * invDistCube [1 FLOP]
  float s = bj.w * invDistCube;

  // a_i =  a_i + s * r_ij [6 FLOPS]
  ai.x += r.x * s;
  ai.y += r.y * s;
  ai.z += r.z * s;

  return ai;
}



// add_force kernel function

__device__ void add_force(float4 *body, float3 force) {
  
  body->x += force.x;
  body->y += force.y;
  body->z += force.z;
}