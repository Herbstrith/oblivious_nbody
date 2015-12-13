#include <math.h>

#include "nbody_kernel.cuh"

// calculate_forces kernel function ... here is where we start

// rectangle kernel function ( without coarsening )

//calculate_force: body x body interaction (from the gems3 algorithm)

__device__ float3 
bodyBodyInteraction(float3 ai, float4 bi, float4 bj) {
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