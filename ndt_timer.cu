/*************************************************************************
LIBRARY: NDT-NEW DEVELOPER TOOLS

FILE: ndt_timer.cu
AUTHOR: Zehuan Wang 
DATA: 12/20/2012

Timer tools
**************************************************************************
**************************************************************************
ROUTINES:
ndt_gpu_timer_start()
ndt_gpu_timer_end()
ndt_cpu_timer_start()
ndt_cpu_timer_end()
ndt_timer_print()
ndt_error_print()
*************************************************************************/
#include "ndt.h"

extern cudaError_t ndt_gpu_timer_start(cudaEvent_t *ndt_o_pcudaEvent_start, cudaEvent_t *ndt_o_pcudaEvent_end)
{
  cudaError_t ndt_err;
  cudaEventCreate(ndt_o_pcudaEvent_start);
  cudaEventCreate(ndt_o_pcudaEvent_end);
  ndt_err = cudaEventRecord(*ndt_o_pcudaEvent_start, 0);
  return ndt_err;
} 

extern cudaError_t ndt_gpu_timer_end(cudaEvent_t ndt_i_cudaEvent_start, cudaEvent_t ndt_i_cudaEvent_end, float* ndt_o_pf_time)
{
  cudaError_t ndt_err;
  cudaEventRecord(ndt_i_cudaEvent_end,0);
  cudaEventSynchronize(ndt_i_cudaEvent_end);
  cudaEventElapsedTime(ndt_o_pf_time,ndt_i_cudaEvent_start,ndt_i_cudaEvent_end);
  cudaEventDestroy( ndt_i_cudaEvent_start );
  ndt_err = cudaEventDestroy( ndt_i_cudaEvent_end );
  return ndt_err;
}


extern ndt_error ndt_cpu_timer_start(long long* ndt_o_pf_start)
{
  if(ndt_o_pf_start == NULL)
    return ndt_input_null_pointer;
  struct timeval start;
  gettimeofday(&start,NULL);
  *ndt_o_pf_start = start.tv_sec*1000000+start.tv_usec;
  return ndt_success;
}

extern ndt_error ndt_cpu_timer_end(long long ndt_i_f_start, float* ndt_o_pf_time)
{
  if(ndt_o_pf_time == NULL)
    return ndt_input_null_pointer;
  struct timeval end;
  gettimeofday(&end,NULL);
  *ndt_o_pf_time = (end.tv_sec*1000000+end.tv_usec - ndt_i_f_start)/1000.0;
  return ndt_success;
}

extern void ndt_timer_print(float ndt_i_pf_time)
{
  printf("The time elapsed is [%fms]\n",ndt_i_pf_time);
  return;
}

extern void ndt_error_print(ndt_error ndt_i_error)
{
  switch(ndt_i_error)
  {
  case ndt_success:
    printf("nv zh Success\n");
    break;
  case ndt_input_null_pointer:
    printf("nv zh input null pointer\n");
    break;
  default:
    printf("nv zh unknown error\n");
    break;
  }
  return;
}
