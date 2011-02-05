// svsvg.cu --- Part of the project OPLib 1.0, a high performance pricing library
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



__global__ void global_sgsvg(float* gen_yy_i, int ni, int nx, int nr, 
								float *invm, float * xval_y, 
								float *SDrift_yi, float *SVol_yi, 
								float *VolDrift_yi, float *VolVol_yi, 
								float *Jumpsz_minus_yi, float *Jumpsz_plus_yi){
								
            const int tid = blockDim.x * blockIdx.x + threadIdx.x;
            const int d = nx * nr;
            const int i = tid / d;
            const int y0 = tid - i * d;
            const int r0 = y0 / nx;
            const int x0 = y0 - r0 * nx;
            int x1, r1, y1;
            float xi;
            float rhs[4];
            bool condition;

            rhs[0] = 0;

            condition = (x0 > 0 && x0 < nx - 1);
            rhs[1] = SVol_yi[y0 + d * i];
            rhs[1] *= (condition * rhs[1]);

            rhs[2] = VolDrift_yi[y0 + d * i];

            condition = (r0 > 0 && r0 < nr - 1);
            rhs[3] = VolVol_yi[y0 + d * i];
            rhs[3] *= condition * rhs[3];

            x1 = x0 - 1;
            r1 = r0;
            y1 = x1 + nx * r1;
            if (x0 > 0)
            {
                xi = 0;
                for (int i1 = 0; i1 < 4; i1++) xi += invm[16 * y0 + 0 + 4 * i1] * rhs[i1];
                gen_yy_i[i * d * d + (y0 + d * y1)] = xi;
            }

            x1 = x0 + 1;
            r1 = r0;
            y1 = x1 + nx * r1;
            if (x0 < nx - 1)
            {
                xi = 0;
                for (int i1 = 0; i1 < 4; i1++) xi += invm[16 * y0 + 1 + 4 * i1] * rhs[i1];
                gen_yy_i[i * d * d + (y0 + d * y1)] = xi;
            }

            x1 = x0;
            r1 = r0 - 1;
            y1 = x1 + nx * r1;
            if (r0 > 0)
            {
                xi = 0;
                for (int i1 = 0; i1 < 4; i1++) xi += invm[16 * y0 + 2 + 4 * i1] * rhs[i1];
                gen_yy_i[i * d * d + (y0 + d * y1)] = xi;
            }

            x1 = x0;
            r1 = r0 + 1;
            y1 = x1 + nx * r1;
            if (r0 < nr - 1)
            {
                xi = 0;
                for (int i1 = 0; i1 < 4; i1++) xi += invm[16 * y0 + 3 + 4 * i1] * rhs[i1];
                gen_yy_i[i * d * d + (y0 + d * y1)] = xi;
            }

            //add jumps

            float jumpsz_minus = Jumpsz_minus_yi[i * d + y0] * xval_y[y0];
            float jumpsz_plus = Jumpsz_minus_yi[i * d + y0] * xval_y[y0];

            for (int r1 = r0 + 1; r1 < nr; r1++)
            {
                float prob = 0;
                float sum0 = 0;
                for (int x1 = 0; x1 < nx; x1++)
                {
                    int y1 = x1 + nx * r1;
                    prob += gen_yy_i[i * d * d + (y0 + d * y1)];
                }

                //prob is the total probability for a volatility transition
                if (prob > 0)
                {
                    for (int x1 = 0; x1 < x0 - 1; x1++)
                    {
                        int y1 = x1 + nx * r1;
                        if (jumpsz_minus > 0)
                        {
                            sum0 += exp((xval_y[y1] - xval_y[y0]) / jumpsz_minus);
                        }
                    }

                    for (int x1 = x0 + 1; x1 < nx; x1++)
                    {
                        int y1 = x1 + nx * r1;
                        if (jumpsz_plus > 0)
                        {
                            sum0 += exp(-(xval_y[y1] - xval_y[y0]) / jumpsz_plus);
                        }
                    }

                    if (sum0 > 0)
                    {
                        float ratio = prob / sum0;
                        for (int x1 = 0; x1 < x0 - 1; x1++)
                        {
                            int y1 = x1 + nx * r1;
                            gen_yy_i[i * d * d + (y0 + d * y1)] = ratio * exp((xval_y[y1] - xval_y[y0]) / jumpsz_minus);
                        }
                        for (int x1 = x0 + 1; x1 < nx; x1++)
                        {
                            int y1 = x1 + nx * r1;
                            gen_yy_i[i * d * d + (y0 + d * y1)] = ratio * exp(-(xval_y[y1] - xval_y[y0]) / jumpsz_plus);
                        }
                    }  //end if(sum0>0)
                }//end if (prob0>0)
            }//end    for (int r1 = r0 + 1; r1 < grid.nr; r1++)


            //fix up sum rules for drift and probability conservation


            float sum0 = 0;
            float drift = 0;
            for (int y1 = 0; y1 < d; y1++)
            {
                if (y0 != y1)
                {
                    if (gen_yy_i[i * d * d + (y0 + d * y1)] < 0)
                    {
                        gen_yy_i[i * d * d + (y0 + d * y1)] = 0;
                    }
                    else
                    {
                        drift += gen_yy_i[i * d * d + (y0 + d * y1)] * (xval_y[y1] - xval_y[y0]);
                        sum0 += gen_yy_i[i * d * d + (y0 + d * y1)];
                    }
                }
            }

            float drift0 = SDrift_yi[y0 + d * i];

            if (drift > drift0)
            {
                if (x0 > 0)
                {
                    int y1 = x0 - 1 + nx * r0;
                    float ratio = (drift - drift0) / (xval_y[y1] - xval_y[y0]);
                    gen_yy_i[i * d * d + (y0 + d * y1)] += ratio;
                    sum0 += ratio;
                }
            }
            else
            {
                if (x0 < nx - 1)
                {
                    int y1 = x0 + 1 + nx * r0;
                    float ratio = (drift0 - drift) / (xval_y[y1] - xval_y[y0]);
                    gen_yy_i[i * d * d + (y0 + d * y1)] += ratio;
                    sum0 += ratio;
                }
            }

            gen_yy_i[i * d * d + (y0 + d * y0)] = -sum0;

        }// end function





extern "C" __declspec( dllexport ) void opcuda_sgsvg(unsigned gen_yyi_ptr, int ni, int nx, int nr, unsigned invmat_ptr, unsigned parptr){

	int d = nx*nr;
	if(d%32!=0) return; //assume d is a multiple of 32
	
	
	float* gen_yyi = (float *) gen_yyi_ptr;
	float* invmat = (float *)invmat_ptr;
	float * xval_y = (float *) parptr;
	float* SDrift_yi = (float *) parptr + d;
	float* SVol_yi = (float *) parptr + d + ni * d;
	float* VolDrift_yi  = (float *) parptr + d + 2 * ni * d;
	float* VolVol_yi = (float *) parptr + d + 3 * ni * d;
	float* Jumpsz_minus_yi = (float *) parptr + d + 4 * ni * d;
	float* Jumpsz_plus_yi = (float *) parptr + d + 5 * ni * d;

						
  global_sgsvg<<< ni * d / 32, 32 >>>
					(gen_yyi, ni, nx, nr, invmat, xval_y, SDrift_yi, SVol_yi, 
								VolDrift_yi, VolVol_yi, Jumpsz_minus_yi, Jumpsz_plus_yi);

}

extern __shared__ float diag[];

__global__ void global_sgsvep(float* gen_yyq, int nq, int d, 
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




extern "C" __declspec( dllexport ) void opcuda_sgsvep
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
								
  global_sgsvep<<< nthreads_per_block, number_of_blocks >>>
																		(gen_yyq, nq, d, DeltaT_q, niter_q);

}



