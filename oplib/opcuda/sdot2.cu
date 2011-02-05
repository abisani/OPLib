// sdot2.cu --- Part of the project OPLib 1.0, a high performance pricing library
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
#define GRIDDIM 32
#define BLOCKDIM 1



#ifdef LINUX
#define __declspec(x)
#define __stdcall
#endif


__global__ void global_sdot2(float* a, float *b, float *c, int nh, int d)
{

		int hh;
		float tot;

		for(int h=0; h<nh; h+= GRIDDIM)
		{		
				hh = h + blockIdx.x;			
				tot = 0;
				if(hh<nh)
				{				
						c[hh] = 0;				
						for(int y=0; y<d; y++)
						{		
								tot += a[hh+nh*y] * b[hh+nh*y];
						}
						c[hh] = (float) tot;
				}
		} 			
 }



extern "C" __declspec( dllexport ) void opcuda_sdot2(unsigned int aptr, 
  											unsigned int  bptr, unsigned int cptr, int m, int n){

    // setup execution parameters
		dim3 grid(32, 1 ), threads(128, 1 );

		float * a = (float *) aptr;
		float * b = (float *) bptr;
		float * c = (float *) cptr;
    global_sdot2<<< GRIDDIM, BLOCKDIM >>>(a, b, c, m, n);

}
