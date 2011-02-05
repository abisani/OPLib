// sglvg.cu --- Part of the project OPLib 1.0, a high performance pricing library
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




__global__ void global_sglvg(float* gen_yyi, int ni, int d,  
								float *invmat, float *SDrift_yi, float *SVol_yi){

						const int tid = blockDim.x * blockIdx.x + threadIdx.x;
						const int i = tid / d;
						const int y0 = tid - i * d;
						int y1;
						float sum;
						float xi;
						float rhs[2];
						bool condition;
					
						if ((i < ni) && (y0 < d))
						{
							for(y1=0; y1<d; y1++)
							{				
					           __syncthreads();
								gen_yyi[i * d * d + (y0 + d * y1)] = 0;
							}

							if (! (y0 == 0 || y0 == d - 1))
							{
								condition = (y0 > 0 && y0 < d - 1);

								__syncthreads();
								rhs[0] = SDrift_yi[y0+d*i];
								rhs[0] *= condition;

								__syncthreads();
								rhs[1] = SVol_yi[y0+d*i];
								rhs[1] *= (condition*rhs[1]);

								sum = 0;
								y1 = y0 - 1;
								if (y0 > 0)
								{
									xi = 0;
									for (int i1 = 0; i1 < 2; i1++) 
													xi += invmat[4 * y0 + 0 + 2 * i1] * rhs[i1];
									gen_yyi[i * d * d + (y0 + d * y1)] = xi;
									sum += xi;
								}


								y1 = y0 + 1;
								if (y0 < d - 1)
								{
									xi = 0;
									for (int i1 = 0; i1 < 2; i1++) xi += invmat[4 * y0 + 1 + 2 * i1] * rhs[i1];
									gen_yyi[i * d * d + (y0 + d * y1)] = xi; 
									sum += xi;
								}

								gen_yyi[i * d * d + (y0 + d * y0)] = -sum;

								}
      		}
}




extern "C" __declspec( dllexport ) void opcuda_sglvg(
                unsigned gen_yyi_ptr, int ni, int d, 
                unsigned invmat_ptr, unsigned parptr){

	float* gen_yyi = (float *) gen_yyi_ptr;
	float* invmat = (float *)invmat_ptr;
	float* SDrift_yi = (float *) parptr;
	float* SVol_yi = (float *) parptr + ni * d;

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int nprocessors = deviceProp.multiProcessorCount;
	int maxthreads = deviceProp.maxThreadsPerBlock;
	int nthreads_per_block = maxthreads;
	int nblocks = 2 * nprocessors;

	int number_of_blocks = max(nblocks, (int) ceil((float)ni * d / nthreads_per_block));  								
								
	global_sglvg<<< nthreads_per_block, number_of_blocks >>>
								(gen_yyi, ni, d, invmat, SDrift_yi, SVol_yi);

}

extern __shared__ float diag[];

__global__ void global_sglvep(float* gen_yyq, int nq, int d, 
								float *DeltaT_q, int *niter_q){

						const int q = blockIdx.x;
						const int y0 = threadIdx.x/d;				
						const int y1 = threadIdx.x%d;				
						float DeltaT = DeltaT_q[q];
						
						float *dt	= diag + nq;				
												
				    if(y0==y1)
				    {
								diag[y0] = gen_yyq[y0 + d * y0 + d * d * q];
						}
			      __syncthreads();
								
				    if(y0==y1 && y0==0)
				    {				    
								float maxdiag = 0;
								for(int y=0; y<d; y++)
										if(-diag[y0]>maxdiag) maxdiag = -diag[y0];

								dt[0] = 0.5 / maxdiag;
								if (dt[0] > 1.0 / 365.0) dt[0] = 1.0 / 365.0;
								niter_q[q] = (int) ceil(log(DeltaT / dt[0]) / log(2.0));
						}
			      __syncthreads();

						gen_yyq[q * d * d + (y0 + d * y1)] *= dt[0];
			      __syncthreads();

				    if(y0==y1)
				    {
								gen_yyq[y0 + d * y0 + d * q * q] += 1;
						}							    
		}




extern "C" __declspec( dllexport ) void opcuda_sglvep
                (unsigned genptr, int nq, int d, unsigned Deltaqptr, 
                 unsigned niterqptr){

	float* gen_yyq = (float *) genptr;
	float* DeltaT_q = (float *) Deltaqptr;
	int *niter_q = (int *) niterqptr;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int nprocessors = deviceProp.multiProcessorCount;
  int maxthreads = deviceProp.maxThreadsPerBlock;
  int nthreads_per_block = maxthreads;
  int nblocks = 2 * nprocessors;
  int number_of_blocks = max(nblocks, (int) ceil((float)nq * d / nthreads_per_block));  								
								
  global_sglvep<<< nthreads_per_block, number_of_blocks >>>(gen_yyq, nq, d, DeltaT_q, niter_q);

}



