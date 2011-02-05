#include "cuda.h"
#include "cublas.h"
#define GRIDDIM 32
#define BLOCKDIM 1

#ifdef LINUX
#define __declspec(x)
#endif

__global__ void global_scopy2(unsigned int * uint_arg)
{

	const unsigned int * argptr = (unsigned int *) uint_arg[blockIdx.x];	

	const int nargs = 3;
	__shared__ unsigned int u[3];
	__shared__ float c[16];

	if(threadIdx.x< nargs) u[threadIdx.x] = argptr[threadIdx.x];
  __syncthreads();
	
	float *destination = (float *) u[0];	
	float *source = (float *) u[1];
	int n = (int) u[2];
		
	if(threadIdx.x<n) c	[threadIdx.x] = source[threadIdx.x]; __syncthreads();
	if(threadIdx.x<n) destination	[threadIdx.x] = c[threadIdx.x]; __syncthreads();
	n-=16; if(n<0) return;
	source += 16; destination +=16;
			 
	if(threadIdx.x<n) c[threadIdx.x] = source[threadIdx.x]; __syncthreads();
	if(threadIdx.x<n) destination	[threadIdx.x] = c[threadIdx.x]; __syncthreads();
	n-=16; if(n<0) return;
	source += 16; destination +=16;
			 
	if(threadIdx.x<n) c	[threadIdx.x] = source[threadIdx.x]; __syncthreads();
	if(threadIdx.x<n) destination	[threadIdx.x] = c[threadIdx.x]; __syncthreads();
	n-=16; if(n<0) return;
	source += 16; destination +=16;

	if(threadIdx.x<n) c	[threadIdx.x] = source[threadIdx.x]; __syncthreads();
	if(threadIdx.x<n) destination	[threadIdx.x] = c[threadIdx.x]; __syncthreads();
	n-=16; if(n<0) return;
	source += 16; destination +=16;

	if(threadIdx.x<n) c	[threadIdx.x] = source[threadIdx.x]; __syncthreads();
	if(threadIdx.x<n) destination	[threadIdx.x] = c[threadIdx.x]; __syncthreads();
	n-=16; if(n<0) return;
	source += 16; destination +=16;

	if(threadIdx.x<n) c	[threadIdx.x] = source[threadIdx.x]; __syncthreads();
	if(threadIdx.x<n) destination	[threadIdx.x] = c[threadIdx.x]; __syncthreads();
	n-=16; if(n<0) return;
	source += 16; destination +=16;

	if(threadIdx.x<n) c	[threadIdx.x] = source[threadIdx.x]; __syncthreads();
	if(threadIdx.x<n) destination	[threadIdx.x] = c[threadIdx.x]; __syncthreads();
	n-=16; if(n<0) return;
	source += 16; destination +=16;

	if(threadIdx.x<n) c	[threadIdx.x] = source[threadIdx.x]; __syncthreads();
	if(threadIdx.x<n) destination	[threadIdx.x] = c[threadIdx.x]; __syncthreads();
	n-=16; if(n<0) return;
	source += 16; destination +=16;

}



extern "C" __declspec( dllexport ) void opcuda_scopy2(unsigned int nblocks, unsigned int uint_arg_ptr){

    // setup execution parameters
		dim3 grid(nblocks, 1 ), threads(16, 1 );

    global_scopy2<<< grid, threads >>>((unsigned int *) uint_arg_ptr);

}





__global__ void global_scopy1(float *destination, float *source, unsigned int remainder)
{
		
	__shared__ float c[16];

	destination += blockIdx.x * 128;
	source += blockIdx.x * 128;
	int n = 128;
	if(blockIdx.x == blockDim.x-1) n = remainder;
	
		
	if(threadIdx.x<n) c	[threadIdx.x] = source[threadIdx.x]; __syncthreads();
	if(threadIdx.x<n) destination	[threadIdx.x] = c[threadIdx.x]; __syncthreads();
	n-=16; if(n<0) return;
	source += 16; destination +=16;
			 
	if(threadIdx.x<n) c[threadIdx.x] = source[threadIdx.x]; __syncthreads();
	if(threadIdx.x<n) destination	[threadIdx.x] = c[threadIdx.x]; __syncthreads();
	n-=16; if(n<0) return;
	source += 16; destination +=16;
			 
	if(threadIdx.x<n) c	[threadIdx.x] = source[threadIdx.x]; __syncthreads();
	if(threadIdx.x<n) destination	[threadIdx.x] = c[threadIdx.x]; __syncthreads();
	n-=16; if(n<0) return;
	source += 16; destination +=16;

	if(threadIdx.x<n) c	[threadIdx.x] = source[threadIdx.x]; __syncthreads();
	if(threadIdx.x<n) destination	[threadIdx.x] = c[threadIdx.x]; __syncthreads();
	n-=16; if(n<0) return;
	source += 16; destination +=16;

	if(threadIdx.x<n) c	[threadIdx.x] = source[threadIdx.x]; __syncthreads();
	if(threadIdx.x<n) destination	[threadIdx.x] = c[threadIdx.x]; __syncthreads();
	n-=16; if(n<0) return;
	source += 16; destination +=16;

	if(threadIdx.x<n) c	[threadIdx.x] = source[threadIdx.x]; __syncthreads();
	if(threadIdx.x<n) destination	[threadIdx.x] = c[threadIdx.x]; __syncthreads();
	n-=16; if(n<0) return;
	source += 16; destination +=16;

	if(threadIdx.x<n) c	[threadIdx.x] = source[threadIdx.x]; __syncthreads();
	if(threadIdx.x<n) destination	[threadIdx.x] = c[threadIdx.x]; __syncthreads();
	n-=16; if(n<0) return;
	source += 16; destination +=16;

	if(threadIdx.x<n) c	[threadIdx.x] = source[threadIdx.x]; __syncthreads();
	if(threadIdx.x<n) destination	[threadIdx.x] = c[threadIdx.x]; __syncthreads();
	n-=16; if(n<0) return;
	source += 16; destination +=16;

}


extern "C" __declspec( dllexport ) void opcuda_scopy1(unsigned destination, unsigned source, unsigned n){

    int nblocks = ((n + 127)/128);
    unsigned int remainder = n - 128 * (nblocks-1);
    global_scopy1<<< nblocks, 16 >>>((float *) destination, (float *) source, remainder);

}

