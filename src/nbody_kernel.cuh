#ifndef __NBODYKERNEL_CUH__
#define __NBODYKERNEL_CUH__


__device__ float3 bodyBodyInteraction(float3 ai, float4 bi, float4 bj);

__device__ void add_force(float4 *body, float3 force);



#endif  //NBODYKERNEL_CUH