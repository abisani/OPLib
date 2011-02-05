// sek.cu --- Part of the project OPLib 1.0, a high performance pricing library
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



static __global__ void global_sek(int d, const float *gen, float dt, float *tpker)
{
		
	const int iny1 = threadIdx.x;
	const int iny2 = threadIdx.y;
	const int iby1 = blockIdx.x * 16;
	const int iby2 = blockIdx.y * 16;
	const int y1 = iby1 + iny1;
	const int y2 = iby2 + iny2;
	
	int id = 0;	
	
	if(y1<d && y2 <d)
	{
			int yy = y1 + d*y2;
			if(y1==y2) id = 1;	
			tpker[yy] = (id + dt * gen[yy]);			
	}

}	


static __global__ void global_sekd(int d, const float *gen, float dt, float *tpker, float *r)
{
		
	const int iny1 = threadIdx.x;
	const int iny2 = threadIdx.y;
	const int iby1 = blockIdx.x * 16;
	const int iby2 = blockIdx.y * 16;
	const int y1 = iby1 + iny1;
	const int y2 = iby2 + iny2;
	
	int id = 0;	
	
	if(y1<d && y2<d)
	{
			int yy = y1 + d*y2;
			if(y1==y2) id = 1;	
			tpker[yy] = (id + dt * gen[yy]) / (1 + dt * r[y1]);			
	}

}	


static __global__ void global_sekd_lambda(int d, const float *gen, float dt, float *tpker, float *r, float *lambda)
{
		
	const int iny1 = threadIdx.x;
	const int iny2 = threadIdx.y;
	const int iby1 = blockIdx.x * 16;
	const int iby2 = blockIdx.y * 16;
	const int y1 = iby1 + iny1;
	const int y2 = iby2 + iny2;
	
	int id = 0;	
	
	if(y1<d && y2<d)
	{
			int yy = y1 + d*y2;
			if(y1==y2) id = 1;	
			tpker[yy] = (id + dt * gen[yy]) / (1 + dt * r[y1] * *lambda);			
	}

}	


extern "C" __declspec( dllexport )  void opcuda_sek(int d, unsigned int genPtr, float dt, unsigned int tpkerPtr)
{	

	int n = (d+15) / 16;
	dim3 grid(n, n);
	dim3 threads(16, 16);

	global_sek<<<grid, threads>>>(d, (float *) genPtr, dt, (float *) tpkerPtr);
  
}	

extern "C" __declspec( dllexport )  void opcuda_sekd(int d, unsigned int genPtr, float dt, unsigned int tpkerPtr, unsigned int rPtr)
{	

	int n = (d+15) / 16;
	dim3 grid(n, n);
	dim3 threads(16, 16);

	global_sekd<<<grid, threads>>>(d, (float *) genPtr, dt, (float *) tpkerPtr, (float *) rPtr);
  
}	


extern "C" __declspec( dllexport )  void opcuda_sekd_lambda(int d, unsigned int genPtr, float dt, unsigned int tpkerPtr, unsigned int rPtr, unsigned int lambdaPtr)
{	

	int n = (d+15) / 16;
	dim3 grid(n, n);
	dim3 threads(16, 16);

	global_sekd_lambda<<<grid, threads>>>(d, (float *) genPtr, dt, (float *) tpkerPtr, (float *) rPtr, (float *) lambdaPtr);
  
}	
