// stranspose.cu --- Part of the project OPLib 1.0, a high performance pricing library
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



#ifdef LINUX
#define __declspec(x)
#define __stdcall
#endif


__global__ void transpose(float* a, float* b, const unsigned int ma, 
				   const unsigned int na, const unsigned int lda, 
				   const unsigned int ldb )
				   {
				   
	__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
	unsigned int i = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int j = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((i < ma) && (j < na))
	{
		block[threadIdx.y][threadIdx.x] = a[i + lda*j];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	i = blockIdx.x * BLOCK_DIM + threadIdx.x;
	j = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((i < ma) && (j < na))
	{
		b[j+ldb*i] = block[threadIdx.y][threadIdx.x];
	}
}

extern "C" __declspec( dllexport ) void transpose(int aptr, int bptr, int ma, 
				   int na, int lda, int ldb ){
				   
// setup execution parameters
    dim3 grid(1+ ma / BLOCK_DIM, 1 + na / BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
	
	float * d_a = (float *) aptr;
	float * d_b = (float *) bptr;
	
    transpose<<< grid, threads >>>(d_a, d_b, ma, na, lda, ldb );
    //cudaThreadSynchronize();
    
}