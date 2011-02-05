// util.cu --- Part of the project OPLib 1.0, a high performance pricing library
// based on operator methods, higher level BLAS and multicore architectures 

// Author:     2009 Claudio Albanese
// Maintainer: Claudio Albanese <claudio@albanese.co.uk>
// Created:    April-July 2009
// Version:    1.0.0
// Credits:    The CUDA code for SGEMM4, SGEMV4 and SSQMM were inspired by 
//             Vasily Volkov's implementation of SGEMM
//			   We use several variations of the multi-threaded Mersenne Twister algorithm of 
//			   period 2203 due to Makoto Matsumoto.
//             The Monte Carlo routine in SMC includes code by Victor Podlozhnyuk 
//             included in the CUDA SDK.
//             CPU-side BLAS and random number generators link to primitives in the
//			   Intel Math Kernel Libraries. 

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; see the file COPYING.  If not, write to
// the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
// Boston, MA 02111-1307, USA.



#include "cuda.h"
#include "cublas.h"
#define BLOCK_DIM 16
#include <time.h>
#include <stdlib.h>
#include <stdio.h>


#ifdef _EMULATED
const char* sEmulator = "Emulator";
#endif


#ifdef LINUX
	#include <sys/time.h>
	#define __declspec(x)
	#define __stdcall
	static double start_time;
	static double end_time;
#else
	#include <windows.h>
	static LARGE_INTEGER start_time;
	static LARGE_INTEGER end_time;
#endif

static int status;


#ifdef LINUX
extern "C" __declspec( dllexport ) void opcuda_create_timer()
{
	timeval tp;
	gettimeofday(&tp, NULL); 
	start_time = (double) tp.tv_sec + (double) tp.tv_usec / 1000000.0;	
}

extern "C" __declspec( dllexport ) void opcuda_reset_timer()
{
	timeval tp;
	gettimeofday(&tp, NULL); 
	start_time = (double) tp.tv_sec + (double) tp.tv_usec / 1000000.0;
}

extern "C" __declspec( dllexport ) void opcuda_stop_timer()
{
	timeval tp;
	gettimeofday(&tp, NULL); 
	end_time = (double) tp.tv_sec + (double) tp.tv_usec / 1000000.0;
}

extern "C" __declspec( dllexport ) double opcuda_get_timer_value()
{
	return (end_time - start_time);
}

#else

extern "C" __declspec( dllexport ) void opcuda_create_timer()
{
   status = QueryPerformanceCounter(&start_time);
}

extern "C" __declspec( dllexport ) void opcuda_reset_timer()
{
   status = QueryPerformanceCounter(&start_time);
}

extern "C" __declspec( dllexport ) void opcuda_stop_timer()
{
   status = QueryPerformanceCounter(&end_time);
}

extern "C" __declspec( dllexport ) double opcuda_get_timer_value()
{
   LARGE_INTEGER frequency;
   QueryPerformanceFrequency(&frequency);
   return ((double) end_time.QuadPart - (double) start_time.QuadPart) / (double)frequency.QuadPart;
}

#endif







extern __host__ cublasStatus CUBLASAPI cublasInit (void);

extern "C" __declspec( dllexport ) int opcuda_cublas_init(){
	status = cublasInit();
	return status;
}




extern "C" __declspec( dllexport ) void opcuda_thread_synchronize(){
	cudaThreadSynchronize();
}



extern __host__ cudaError_t CUDARTAPI cudaSetDevice(int device);

extern "C" __declspec( dllexport ) void opcuda_set_device( int dev ){
	cudaSetDevice(dev );
}




extern __host__ cudaError_t CUDARTAPI cudaGetDevice(int *device);

extern "C" __declspec( dllexport ) int opcuda_get_device(void)
{
	int dev;
	cudaGetDevice(&dev);
	return dev;
}


extern "C" __declspec( dllexport ) int opcuda_device_get_count(void)
{
	int count;
	cudaGetDeviceCount(&count);
	return count;
}

extern "C" __declspec( dllexport ) void opcuda_device_get_name(char *name, int len, int dev)
{
#ifdef _EMULATED
	strncpy(name, sEmulator, len);
#else
//	cuDeviceGetName(name, len, dev);
#endif
}

extern "C" __declspec( dllexport ) void opcuda_device_total_memory(unsigned *bytes, int dev)
{
#ifdef _EMULATED
	*bytes = 0xffffffff;
#else
//	cuDeviceTotalMem(bytes, dev);
#endif
}


extern "C" __declspec( dllexport ) int opcuda_memcpy_h2d(unsigned dptr, 
														unsigned long hptr, unsigned sz)
{
	void * hp = (void *) hptr;
	status = cudaMemcpy((void *) dptr, (void *) hp, sz, cudaMemcpyHostToDevice);
	return status;
}

extern "C" __declspec( dllexport ) int opcuda_memcpy_d2h(unsigned dptr, 
														unsigned long hptr, unsigned sz)
{
	void * hp = (void *) hptr;
	status = cudaMemcpy((void *) hp, (void *) dptr, sz, cudaMemcpyDeviceToHost);
	return status;
}

extern "C" __declspec( dllexport ) int opcuda_memcpy_d2d(unsigned dptr2, unsigned dptr1, unsigned sz){
  status = cudaMemcpy((void *) dptr2, (void *) dptr1, sz, cudaMemcpyDeviceToDevice);
  return status;
}


extern "C" __declspec( dllexport ) unsigned int opcuda_get_status()
{
	return status;
}

extern "C" __declspec( dllexport ) unsigned int opcuda_mem_alloc(unsigned sz)
{
	void* dptr;
	status = cudaMalloc( & dptr, sz);
	return (unsigned ) (unsigned long) dptr;
}


extern "C" __declspec( dllexport ) void opcuda_mem_free_device(unsigned ptr)
{
    cudaThreadSynchronize();
    status = cudaFree((void *)ptr);
}

extern "C" __declspec( dllexport ) unsigned long opcuda_mem_alloc_host(unsigned sz)
{
	unsigned int hptr;
	void *buf;
	status = cudaMallocHost( & buf, sz);
	hptr = (unsigned long) buf;
	return hptr;
}

extern "C" __declspec( dllexport ) void opcuda_mem_free_host(unsigned long hptr)
{
	void *hp = (void *)hptr;
    cudaFreeHost(hp);
}



extern "C" __declspec( dllexport ) int opcuda_device_count(void)
{
	int count;
	cudaGetDeviceCount(&count);
	return count;
}


extern "C" __declspec( dllexport )  int opcuda_multi_processor_count(int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    return deviceProp.multiProcessorCount;
}

extern "C" __declspec( dllexport )  int opcuda_max_threads_per_block(int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    return deviceProp.maxThreadsPerBlock;
}


#define NTHREADS 4096

__global__ void global_scopy(int n, float *x, int incx, float *y, int incy)
{
		const int tid = blockDim.x * blockIdx.x + threadIdx.x;
		int j;

		for(int i=0; i<n; i+= NTHREADS)
		{		
				j = i+tid;
				if(j<n) y[j*incy] = x[j*incx];
		} 		

}

extern "C" __declspec( dllexport )  void opcuda_scopy(int n, 
					unsigned int xptr, int incx, unsigned int yptr, int incy)
{
		float * x = (float*)xptr;		
		float * y = (float*)yptr;
		global_scopy<<< 32, 128 >>>(n, x, incx, y, incy);
}


extern "C" __declspec( dllexport )  int opcuda_shutdown()
{
		return cublasShutdown ();
}

