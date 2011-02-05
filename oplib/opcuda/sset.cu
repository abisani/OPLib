// sset.cu --- Part of the project OPLib 1.0, a high performance pricing library
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

#define NTHREADS 4096


__global__ void global_ssetall(float *x, int n, float c, int incx)
{
		const int tid = blockDim.x * blockIdx.x + threadIdx.x;
		int j;

		for(int i=0; i<n; i+= NTHREADS)
		{		
				j = i+tid;
				if(j<n) x[j*incx] = c;
		} 		

}

extern "C" __declspec( dllexport )  void opcuda_ssetall(unsigned int xPtr, int n, float c, int incx)
{
		
	global_ssetall<<< 32, 128 >>>((float *)xPtr, n, c, incx);

}

__global__ void global_ssetone(float *x, int n, float c, int i)
{
		const int tid = blockDim.x * blockIdx.x + threadIdx.x;

		if(tid==i) x[i] = c;

}

extern "C" __declspec( dllexport )  void opcuda_ssetone(unsigned int xPtr, int n, float c, int i)
{
		
	global_ssetone<<< 1, 128 >>>((float *)xPtr, n, c, i);

}
