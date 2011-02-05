// speak.cu --- Part of the project OPLib 1.0, a high performance pricing library
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


#ifdef LINUX
#define __declspec(x)
#define __stdcall
#endif


#include "cuda.h"
#include "cublas.h"
#define BLOCK_DIM 128


static __global__ void global_benchmark_register_peak(float * x1, float *y1, float * x2, float *y2) {
													
		__shared__ float sx1[BLOCK_DIM];
		__shared__ float sy1[BLOCK_DIM];
		__shared__ float sx2[BLOCK_DIM];
		__shared__ float sy2[BLOCK_DIM];
		float rx1[8], ry1[8], rx2[8], ry2[8];
		int tid = threadIdx.x;
		int offset = blockIdx.x * blockDim.x;
		
		sx1[tid] = x1[offset + tid];
		__syncthreads();	
		sy1[tid] = y1[offset + tid];
		__syncthreads();		
		sx2[tid] = x2[offset + tid];
		__syncthreads();		
		sy2[tid] = y2[offset + tid];
		__syncthreads();

		for(int i=0; i<8; i++)
		{
			rx1[i] = sx1[tid];
			ry1[i] = sy1[tid];
			rx2[i] = sx2[tid];
			ry2[i] = sy2[tid];
		}

		float temp[8];
		#pragma unroll
		for(int i = 0; i<300; i++)
		{
			temp[0] = rx1[0];
			temp[1] = rx1[1];
			temp[2] = rx1[2];
			temp[3] = rx1[3];
			temp[4] = rx1[4];
			temp[5] = rx1[5];
			temp[6] = rx1[6];
			temp[7] = rx1[7];
			
			rx1[0] = rx1[0] * rx2[0] + ry1[0] * ry2[0];			
			rx1[1] = rx1[1] * rx2[1] + ry1[1] * ry2[1];			
			rx1[2] = rx1[2] * rx2[2] + ry1[2] * ry2[2];			
			rx1[3] = rx1[3] * rx2[3] + ry1[3] * ry2[3];			
			rx1[4] = rx1[4] * rx2[4] + ry1[4] * ry2[4];			
			rx1[5] = rx1[5] * rx2[5] + ry1[5] * ry2[5];			
			rx1[6] = rx1[6] * rx2[6] + ry1[6] * ry2[6];			
			rx1[7] = rx1[7] * rx2[7] + ry1[7] * ry2[7];			

			ry1[0] = ry1[0] * rx2[0] - temp[0] * ry2[0];
			ry1[1] = ry1[1] * rx2[1] - temp[1] * ry2[1];
			ry1[2] = ry1[2] * rx2[2] - temp[2] * ry2[2];
			ry1[3] = ry1[3] * rx2[3] - temp[3] * ry2[3];
			ry1[4] = ry1[4] * rx2[4] - temp[4] * ry2[4];
			ry1[5] = ry1[5] * rx2[5] - temp[5] * ry2[5];
			ry1[6] = ry1[6] * rx2[6] - temp[6] * ry2[6];
			ry1[7] = ry1[7] * rx2[7] - temp[7] * ry2[7];
		}

		sx1[tid] = rx1[0] + rx1[1] + rx1[2] + rx1[3] + rx1[4] + rx1[5] + rx1[6] + rx1[7];
		sy1[tid] = ry1[0] + ry1[1] + ry1[2] + ry1[3] + ry1[4] + ry1[5] + ry1[6] + ry1[7];
		sx2[tid] = rx2[0] + rx2[1] + rx2[2] + rx2[3] + rx2[4] + rx2[5] + rx2[6] + rx2[7];
		sy2[tid] = ry2[0] + ry2[1] + ry2[2] + ry2[3] + ry2[4] + ry2[5] + ry2[6] + ry2[7];
		
		x1[offset + tid] = sx1[tid];
		__syncthreads();		
		y1[offset +tid] = sy1[tid];
		__syncthreads();
		x2[offset +tid] = sx2[tid];
		__syncthreads();		
		y2[offset +tid] = sy2[tid];
		__syncthreads();
		
}



	extern "C" __declspec( dllexport ) void opcuda_benchmark_register_peak(unsigned x1ptr, unsigned y1ptr, unsigned x2ptr, unsigned y2ptr)
	{												
		global_benchmark_register_peak<<< 64, BLOCK_DIM >>>((float *) x1ptr, (float *) y1ptr, (float *) x2ptr, (float *) y2ptr);																															
	}





static __global__ void global_benchmark_shared_peak(float * x1, float *y1, float * x2, float *y2) {
													
		__shared__ float sx1[BLOCK_DIM];
		__shared__ float sy1[BLOCK_DIM];
		__shared__ float sx2[BLOCK_DIM];
		float rx1[8], ry1[8];
		int tid = threadIdx.x;
		int offset = blockIdx.x * blockDim.x;
		
		sx1[tid] = x1[offset + tid];
		__syncthreads();	
		sy1[tid] = y1[offset + tid];
		__syncthreads();		

		for(int i=0; i<8; i++)
		{
			rx1[i] = sx1[tid];
			ry1[i] = sy1[tid];
		}

		#pragma unroll
		for(int i = 0; i<300; i++)
		{
		
			rx1[0] = rx1[0] * sx2[0] + ry1[0] ;			
			rx1[1] = rx1[1] * sx2[1] + ry1[1] ;			
			rx1[2] = rx1[2] * sx2[2] + ry1[2] ;			
			rx1[3] = rx1[3] * sx2[3] + ry1[3] ;			
			rx1[4] = rx1[4] * sx2[4] + ry1[4] ;			
			rx1[5] = rx1[5] * sx2[5] + ry1[5] ;			
			rx1[6] = rx1[6] * sx2[6] + ry1[6] ;			
			rx1[7] = rx1[7] * sx2[7] + ry1[7] ;			

			ry1[0] = ry1[0] * sx2[0] - rx1[0];
			ry1[1] = ry1[1] * sx2[1] - rx1[1];
			ry1[2] = ry1[2] * sx2[2] - rx1[2];
			ry1[3] = ry1[3] * sx2[3] - rx1[3];
			ry1[4] = ry1[4] * sx2[4] - rx1[4];
			ry1[5] = ry1[5] * sx2[5] - rx1[5];
			ry1[6] = ry1[6] * sx2[6] - rx1[6];
			ry1[7] = ry1[7] * sx2[7] - rx1[7];
		}

		sx1[tid] = rx1[0] + rx1[1] + rx1[2] + rx1[3] + rx1[4] + rx1[5] + rx1[6] + rx1[7];
		sy1[tid] = ry1[0] + ry1[1] + ry1[2] + ry1[3] + ry1[4] + ry1[5] + ry1[6] + ry1[7];
		sx2[tid] = sx2[0] + sx2[1] + sx2[2] + sx2[3] + sx2[4] + sx2[5] + sx2[6] + sx2[7];
		
		x1[offset + tid] = sx1[tid];
		__syncthreads();		
		y1[offset +tid] = sy1[tid];
		__syncthreads();
		x2[offset +tid] = sx2[tid];
		__syncthreads();		
		
}



	extern "C" __declspec( dllexport ) void opcuda_benchmark_shared_peak(unsigned x1ptr, unsigned y1ptr, unsigned x2ptr, unsigned y2ptr)
	{												
		global_benchmark_shared_peak<<< 64, BLOCK_DIM >>>((float *) x1ptr, (float *) y1ptr, (float *) x2ptr, (float *) y2ptr);																															
	}

