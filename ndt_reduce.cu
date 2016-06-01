/*************************************************************************
LIBRARY: NDT-NEW DEVELOPER TOOLS

FILE: ndt_reduce.cu
AUTHOR: Zehuan Wang 
DATA: 12/20/2012

Reduction tools
**************************************************************************
**************************************************************************
ROUTINES:
ndt_block_reduce_min
*************************************************************************/
#include "ndt.h"

#ifndef MIN
#define MIN(x,y) ((x)>(y)?(y):(x))
#endif

template<typename T>
__device__ T ndt_block_reduce_min(T *ndt_i_pt_array, const int ndt_i_i_length)
{
  int t = threadIdx.x;
  bool odd = false;
  extern __shared__ T array[];
  if(t < ndt_i_i_length)
  {
    
    array[t] = ndt_i_pt_array[t];
  }
  __syncthreads();
  int i = ndt_i_i_length;
  while(i > 1)
  {
    if((i>>1)*2 < i)
    {
      i = i>>1;
      i++;
      odd = true;
    }
    else
    {
      i = i>>1;
      odd = false;
    }
    if((t < i && odd == false) || t < i-1)
      array[t] = MIN(array[t+i],array[t]);
    __syncthreads();
  }
  
  return array[0];
}
