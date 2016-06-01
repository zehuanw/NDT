/*************************************************************************
LIBRARY: NDT-NEW DEVELOPER TOOLS

FILE: ndt.h
AUTHOR: Zehuan Wang 
DATA: 12/20/2012

Header file of NDT
**************************************************************************
**************************************************************************
ROUTINES:
ndt_gpu_timer_start()
ndt_gpu_timer_end()
ndt_cpu_timer_start()
ndt_cpu_timer_end()
ndt_timer_print()
ndt_error_print()

ndt_block_reduce_min()
*************************************************************************/
#include <time.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>

#define NDT_ERR {					\
  cudaDeviceSynchronize();\
    cudaError_t ndt_cudaError_crt = cudaGetLastError();\
    printf("%d,%s\n",__LINE__,cudaGetErrorString(ndt_cudaError_crt));\
}


enum ndt_error {ndt_success,ndt_input_null_pointer};

/************************************************************************
#cat: ndt_gpu_timer_start - used to init the cudaEvent start and end to
#cat: timing the kernel run on gpu.
#cat: before it we need to declare cudaEvent start and end. Then put the
#cat: pointer in to it.

Output:
ndt_o_pcudaEvent_start - pointer of the event start
ndt_o_pcudaEvent_end - pointer of the event end
Input:
--
Return:
error message of cuda function
 ***********************************************************************/
extern cudaError_t ndt_gpu_timer_start(cudaEvent_t *ndt_o_pcudaEvent_start, cudaEvent_t *ndt_o_pcudaEvent_end);

/***********************************************************************
#cat: ndt_gpu_timer_end - used to used to timing the kernel run on gpu.
#cat: before it we need to call ndt_gpu_timer_start to init the 
#cat: cudaEvent start and end. then put them as input of this function.
#cat: This function will output the time elapsed in ms.

Output:
ndt_o_pf_time - pointer to the record time.
Input:
ndt_i_cudaEvent_start - the event start
ndt_i_cudaEvent_end - the event end
Return:
error message of cuda function
 **********************************************************************/
extern cudaError_t ndt_gpu_timer_end(cudaEvent_t ndt_i_cudaEvent_start, cudaEvent_t ndt_i_cudaEvent_end, float* ndt_o_pf_time);


/***********************************************************************
#cat: ndt_cpu_timer_start - record the start time of the host code
#cat: we need to declare a float start before it and put the pointer into it

Output:
ndt_o_pf_start - pinter to the start time.
Input:
--
Return:
NDT_ERROR message
 **********************************************************************/
extern ndt_error ndt_cpu_timer_start(long long* ndt_o_pf_start);


/***********************************************************************
#cat: ndt_cpu_timer_end - record the end time of the host code
#cat: we need to call ndt_cpu_timer_start before it. And pass the
#cat: output into this function. We need to declare a float time before
#cat: to store the time elapsed in ms.

Output:
ndt_o_pf_time - time elapsed in ms
Input:
ndt_i_f_start - the start time.
Return:
NDT_ERROR message
 **********************************************************************/
extern ndt_error ndt_cpu_timer_end(long long ndt_i_f_start, float* ndt_o_pf_time);

/**********************************************************************
#cat: ndt_cpu_timer_print - print the time cost in ms.

Output:
--
Input:
ndt_i_pf_time - time elapsed in ms
 *********************************************************************/
extern void ndt_timer_print(float ndt_i_pf_time);


/**********************************************************************
#cat: ndt_cpu_timer_print - print the time cost in ms.

Output:
--
Input:
ndt_i_error - error message returned by ndt functions.
 *********************************************************************/
extern void ndt_error_print(ndt_error ndt_i_error);

/**********************************************************************
#cat: ndt_block_reduce_min - a device function to find the min value
#cat: in a array. Each block reduce a array. Need Enough Shared (
#cat: ndt_i_i_length*sizeof(T) and enough threads (blockDim == ndt_i_i_length)

Output:
--
Input:
ndt_i_pt_array - input array of different types
ndt_i_i_length - length of the input array

Return:
The min value
 *********************************************************************/
template<typename T>
__device__ T ndt_block_reduce_min(T *ndt_i_pt_array, const int ndt_i_i_length);


