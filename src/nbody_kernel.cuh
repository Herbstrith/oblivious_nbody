#ifndef __NBODYKERNEL_CUH__
#define __NBODYKERNEL_CUH__



__global__ void calculate_forces( int nbodies, float4 *bodies);

__global__ void triangle(int n0, int n1, float4 *bodies);

__global__ void rect(int i0, int i1, int j0, int j1, float4 *bodies);

__device__ float3 calculate_force(float3 ai, float4 bi, float4 bj);

__device__ void add_force(float4 *body, float3 force);



#endif  //NBODYKERNEL_CUH