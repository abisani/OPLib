// smc.cu --- Part of the project OPLib 1.0, a high performance pricing library
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


#include "MersenneTwister.h"
#include <cuda.h>

#define MAX_NUMBER_OF_FACTORS 100  // must be an even number
#define CHOLM_SZ MAX_NUMBER_OF_FACTORS*MAX_NUMBER_OF_FACTORS


#ifdef LINUX
#define __declspec(x)
#endif

__constant__ float cholm[CHOLM_SZ];
__device__ static mt_struct_stripped ds_MT[MT_RNG_COUNT];
static mt_struct_stripped h_MT[MT_RNG_COUNT];


#define PI 3.14159265358979f
__device__ void BoxMuller(float& u1, float& u2){
    float   r = sqrtf(-2.0f * logf(u1));
    float phi = 2 * PI * u2;
    u1 = r * __cosf(phi);
    u2 = r * __sinf(phi);
}

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
__device__ inline float cndGPU(float d)
		{
    
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
        K = 1.0f / (1.0f + 0.2316419f * fabsf(d));

    float
        cnd = RSQRT2PI * __expf(- 0.5f * d * d) * 
        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if(d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}



//Load twister configurations

extern "C" __declspec( dllexport ) int opcuda_mc_load_mt_gpu(char * MT_stream, long sz){

	char * h_MT_ptr = (char *) MT_stream;
	if(sz != sizeof(h_MT)) return 1;
	for(int i = 0; i<sizeof(h_MT); i++)
	{
		h_MT_ptr[i] = MT_stream[i];
	}
	return 0;
}

extern "C" __declspec( dllexport ) int opcuda_mc_nrng()
{
		return MT_RNG_COUNT;		
}

extern "C" __declspec( dllexport ) int opcuda_mc_status_sz()
{
		return MT_NN*MT_RNG_COUNT;	
}


////////////////////////////////////////////////////////////////////////////////
// Write MT_RNG_COUNT vertical lanes of NPerRng random numbers to *d_Random.
// For coalesced global writes MT_RNG_COUNT should be a multiple of warp size.
// Initial states for each generator are the same, since the states are
// initialized from the global seed. In order to improve distribution properties
// on small NPerRng we supply dedicated (local) seeds to each twister.
////////////////////////////////////////////////////////////////////////////////


__global__ void mc_setseed_device(unsigned int* mt_rs)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int iRng = tid;
    const int nRng = blockDim.x * gridDim.x;
          
	mt_struct_stripped config = ds_MT[iRng];

    //Initialize state variable
    mt_rs[tid + nRng * 0] = config.seed;

    for(int iState = 1; iState < MT_NN; iState++)
    mt_rs[tid + nRng * iState] = 
       (1812433253U * (mt_rs[tid + nRng * (iState - 1)] ^ (mt_rs[tid + nRng *(iState - 1)] >> 30)) + iState) & MT_WMASK;  

}


//Initialize/seed twister for current GPU context
extern "C" __declspec( dllexport ) void opcuda_mc_setseed(unsigned long int host_seedptr, unsigned int mtptr){
    int i;
    //Need to be thread-safe
    mt_struct_stripped *MT = (mt_struct_stripped *)malloc(MT_RNG_COUNT * sizeof(mt_struct_stripped));

	 unsigned int * seed = (unsigned int *) host_seedptr;
    
    for(i = 0; i < MT_RNG_COUNT; i++)
    {
        MT[i]      = h_MT[i];
        MT[i].seed = seed[i];
    }

	cudaMemcpyToSymbol(ds_MT, MT, sizeof(h_MT));
    
	mc_setseed_device<<<64, 64>>>((unsigned int*) mtptr);    
    
	free(MT);

}



__global__ void mc1f_device1024(unsigned int* mt_rs, short y0, 
								short *y_sk, int nscenPerRng, const int nk, const int d, 
								float *ctpker_m_yy, int *m_k){


	const int iRng = blockDim.x * blockIdx.x + threadIdx.x;
	const int nRng = blockDim.x * gridDim.x; 		
				         
    int iState, iState1, iStateM;
    unsigned int mti, mti1, mtiM, x;		

	short yprevious;
	float rand;

    unsigned int mt[MT_NN];
    unsigned *rmt =  &mt_rs[iRng];

	//coalesced read of status vector
	#pragma unroll      
	for(iState = 0; iState < MT_NN; iState++) 
		{  
			mt[iState] = *rmt; rmt += nRng;  
		}
	__syncthreads();

    //Load bit-vector Mersenne Twister parameters   
    mt_struct_stripped config = ds_MT[iRng];

    iState = 0;
    mti1 = mt[0];
    int scen = 0;
    int k = 0;

	yprevious = y0;

	for(; ;){					
			//Mersenne Twister
			iState1 = iState + 1;
			iStateM = iState + MT_MM;
			if(iState1 >= MT_NN) iState1 -= MT_NN;
			if(iStateM >= MT_NN) iStateM -= MT_NN;
			mti  = mti1;
			mti1 = mt[iState1];
			mtiM = mt[iStateM];

			x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
			x =  mtiM ^ (x >> 1) ^ ((x & 1) ? config.matrix_a : 0);
			mt[iState] = x;
			iState = iState1;

			//Tempering transformation
			x ^= (x >> MT_SHIFT0);
			x ^= (x << MT_SHIFTB) & config.mask_b;
			x ^= (x << MT_SHIFTC) & config.mask_c;
			x ^= (x >> MT_SHIFT1);
            				 
			//convert to (0, 1] float
			rand = ((float)x + 1.0f) / 4294967296.0f;					
					        
			short *ry = y_sk + iRng + nRng * scen + nRng * nscenPerRng * k;

			int ub, lb;		
			int m = m_k[k];
			float *rker = &ctpker_m_yy[m*d*d+yprevious];					
			ub = d-1; 					
			lb = 0; 
														
			int ymid;
    		short y;

			#pragma unroll
			for(int iter = 0; iter<10; iter++)
				{
					ymid = (ub + lb)/2;
				    if(rand < rker[d * ymid])	ub = ymid;		//this is the bottleneck
				    else lb = ymid;
				}
					
    		y = ub;
    		//uncomment the following for debug checks
			//if(ub > lb +1) y = -1;							//this will trigger an exception				    											
			//if(rker[d * ub] < rand) y = - 1000 - ub;			//this will trigger an exception				    											
			//if(rker[d * lb] > rand && lb>0) y = -2000 - lb;	//this will trigger an exception				    											
			__syncthreads();					
			*ry = y; 
			ry += nRng * nscenPerRng; 
			yprevious = y;

          					
			if(++k >= nk)
			{
				k = 0;
				yprevious = y0;
	     		if(++scen >= nscenPerRng) break;
			}					  
    }		//end for(;;)


	//save status vector of random number generator
    rmt =  &mt_rs[iRng];
	__syncthreads();
	
	#pragma unroll      
	for(iState = 0; iState < MT_NN; iState++)
		{  
			*rmt = mt[iState]; rmt += nRng;
		}  
}




__global__ void mc1f_device512(unsigned int* mt_rs, short y0, 
								short *y_sk, int nscenPerRng, const int nk, const int d, 
								float *ctpker_m_yy, int *m_k){


	const int iRng = blockDim.x * blockIdx.x + threadIdx.x;
	const int nRng = blockDim.x * gridDim.x; 		
				         
    int iState, iState1, iStateM;
    unsigned int mti, mti1, mtiM, x;		

	short yprevious;
	float rand;

    unsigned int mt[MT_NN];
    unsigned *rmt =  &mt_rs[iRng];

	//coalesced read of status vector
	#pragma unroll      
	for(iState = 0; iState < MT_NN; iState++) 
		{  
			mt[iState] = *rmt; rmt += nRng;  
		}
	__syncthreads();

    //Load bit-vector Mersenne Twister parameters   
    mt_struct_stripped config = ds_MT[iRng];

    iState = 0;
    mti1 = mt[0];
    int scen = 0;
    int k = 0;

	yprevious = y0;

	for(; ;){					
			//Mersenne Twister
			iState1 = iState + 1;
			iStateM = iState + MT_MM;
			if(iState1 >= MT_NN) iState1 -= MT_NN;
			if(iStateM >= MT_NN) iStateM -= MT_NN;
			mti  = mti1;
			mti1 = mt[iState1];
			mtiM = mt[iStateM];

			x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
			x =  mtiM ^ (x >> 1) ^ ((x & 1) ? config.matrix_a : 0);
			mt[iState] = x;
			iState = iState1;

			//Tempering transformation
			x ^= (x >> MT_SHIFT0);
			x ^= (x << MT_SHIFTB) & config.mask_b;
			x ^= (x << MT_SHIFTC) & config.mask_c;
			x ^= (x >> MT_SHIFT1);
            				 
			//convert to (0, 1] float
			rand = ((float)x + 1.0f) / 4294967296.0f;					
					        
			short *ry = y_sk + iRng + nRng * scen + nRng * nscenPerRng * k;

			int ub, lb;		
			int m = m_k[k];
			float *rker = &ctpker_m_yy[m*d*d+yprevious];					
			ub = d-1; 					
			lb = 0; 
														
			int ymid;
    		short y;
			
			#pragma unroll
			for(int iter = 0; iter<9; iter++)
				{
					ymid = (ub + lb)/2;
				    if(rand < rker[d * ymid])	ub = ymid;		//this read is the bottleneck
				    else lb = ymid;
				}
					
    		y = ub;
    		//uncomment the following for debug checks
			//if(ub > lb +1) y = -1;							//this will trigger an exception				    											
			//if(rker[d * ub] < rand) y = - 1000 - ub;			//this will trigger an exception				    											
			//if(rker[d * lb] > rand && lb>0) y = -2000 - lb;	//this will trigger an exception				    											
			__syncthreads();					
			*ry = y; 
			ry += nRng * nscenPerRng; 
			yprevious = y;

          					
			if(++k >= nk)
			{
				k = 0;
				yprevious = y0;
	     		if(++scen >= nscenPerRng) break;
			}					  
    }		//end for(;;)


	//save status vector of random number generator
    rmt =  &mt_rs[iRng];
	__syncthreads();
	
	#pragma unroll      
	for(iState = 0; iState < MT_NN; iState++)
		{  
			*rmt = mt[iState]; rmt += nRng;
		}  
}



__global__ void mc1f_device256(unsigned int* mt_rs, short y0, 
								short *y_sk, int nscenPerRng, const int nk, const int d, 
								float *ctpker_m_yy, int *m_k){


	const int iRng = blockDim.x * blockIdx.x + threadIdx.x;
	const int nRng = blockDim.x * gridDim.x; 		
				         
    int iState, iState1, iStateM;
    unsigned int mti, mti1, mtiM, x;		

	short yprevious;
	float rand;

    unsigned int mt[MT_NN];
    unsigned *rmt =  &mt_rs[iRng];

	//coalesced read of status vector
	#pragma unroll      
	for(iState = 0; iState < MT_NN; iState++) 
		{  
			mt[iState] = *rmt; rmt += nRng;  
		}
	__syncthreads();

    //Load bit-vector Mersenne Twister parameters   
    mt_struct_stripped config = ds_MT[iRng];

    iState = 0;
    mti1 = mt[0];
    int scen = 0;
    int k = 0;

	yprevious = y0;

	for(; ;){					
			//Mersenne Twister
			iState1 = iState + 1;
			iStateM = iState + MT_MM;
			if(iState1 >= MT_NN) iState1 -= MT_NN;
			if(iStateM >= MT_NN) iStateM -= MT_NN;
			mti  = mti1;
			mti1 = mt[iState1];
			mtiM = mt[iStateM];

			x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
			x =  mtiM ^ (x >> 1) ^ ((x & 1) ? config.matrix_a : 0);
			mt[iState] = x;
			iState = iState1;

			//Tempering transformation
			x ^= (x >> MT_SHIFT0);
			x ^= (x << MT_SHIFTB) & config.mask_b;
			x ^= (x << MT_SHIFTC) & config.mask_c;
			x ^= (x >> MT_SHIFT1);
            				 
			//convert to (0, 1] float
			rand = ((float)x + 1.0f) / 4294967296.0f;					
					        
			short *ry = y_sk + iRng + nRng * scen + nRng * nscenPerRng * k;

			int ub, lb;		
			int m = m_k[k];
			float *rker = &ctpker_m_yy[m*d*d+yprevious];					
			ub = d-1; 					
			lb = 0; 
														
			int ymid;
    		short y;

			float ctpker;
			#pragma unroll
			for(int iter = 0; iter<8; iter++)
				{
					ymid = (ub + lb)/2;
					ctpker = rker[d * ymid];				//this is the bottleneck
				    if(rand < ctpker)	ub = ymid;		
				    else lb = ymid;
				}
					
    		y = ub;
    		//uncomment the following for debug checks
			//if(ub > lb +1) y = -1;							//this will trigger an exception				    											
			//if(rker[d * ub] < rand) y = - 1000 - ub;			//this will trigger an exception				    											
			//if(rker[d * lb] > rand && lb>0) y = -2000 - lb;	//this will trigger an exception				    											
			__syncthreads();					
			*ry = y; 
			ry += nRng * nscenPerRng; 
			yprevious = y;

          					
			if(++k >= nk)
			{
				k = 0;
				yprevious = y0;
	     		if(++scen >= nscenPerRng) break;
			}					  
    }		//end for(;;)


	//save status vector of random number generator
    rmt =  &mt_rs[iRng];
	__syncthreads();
	
	#pragma unroll      
	for(iState = 0; iState < MT_NN; iState++)
		{  
			*rmt = mt[iState]; rmt += nRng;
		}  
}



__global__ void mc1f_device128(unsigned int* mt_rs, short y0, 
								short *y_sk, int nscenPerRng, const int nk, const int d, 
								float *ctpker_m_yy, int *m_k){


	const int iRng = blockDim.x * blockIdx.x + threadIdx.x;
	const int nRng = blockDim.x * gridDim.x; 		
				         
    int iState, iState1, iStateM;
    unsigned int mti, mti1, mtiM, x;		

	short yprevious;
	float rand;

    unsigned int mt[MT_NN];
    unsigned *rmt =  &mt_rs[iRng];

	//coalesced read of status vector
	#pragma unroll      
	for(iState = 0; iState < MT_NN; iState++) 
		{  
			mt[iState] = *rmt; rmt += nRng;  
		}
	__syncthreads();

    //Load bit-vector Mersenne Twister parameters   
    mt_struct_stripped config = ds_MT[iRng];

    iState = 0;
    mti1 = mt[0];
    int scen = 0;
    int k = 0;

	yprevious = y0;

	for(; ;){					
			//Mersenne Twister
			iState1 = iState + 1;
			iStateM = iState + MT_MM;
			if(iState1 >= MT_NN) iState1 -= MT_NN;
			if(iStateM >= MT_NN) iStateM -= MT_NN;
			mti  = mti1;
			mti1 = mt[iState1];
			mtiM = mt[iStateM];

			x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
			x =  mtiM ^ (x >> 1) ^ ((x & 1) ? config.matrix_a : 0);
			mt[iState] = x;
			iState = iState1;

			//Tempering transformation
			x ^= (x >> MT_SHIFT0);
			x ^= (x << MT_SHIFTB) & config.mask_b;
			x ^= (x << MT_SHIFTC) & config.mask_c;
			x ^= (x >> MT_SHIFT1);
            				 
			//convert to (0, 1] float
			rand = ((float)x + 1.0f) / 4294967296.0f;					
					        
			short *ry = y_sk + iRng + nRng * scen + nRng * nscenPerRng * k;

			int ub, lb;		
			int m = m_k[k];
			float *rker = &ctpker_m_yy[m*d*d+yprevious];					
			ub = d-1; 					
			lb = 0; 
														
			int ymid;
    		short y;
			float ctpker;
			#pragma unroll
			for(int iter = 0; iter<7; iter++)
				{
					ymid = (ub + lb)/2;
					ctpker = rker[d * ymid];				//this is the bottleneck
				    if(rand < ctpker)	ub = ymid;		
				    else lb = ymid;
				}
					
    		y = ub;
    		//uncomment the following for debug checks
			//if(ub > lb +1) y = -1;							//this will trigger an exception				    											
			//if(rker[d * ub] < rand) y = - 1000 - ub;			//this will trigger an exception				    											
			//if(rker[d * lb] > rand && lb>0) y = -2000 - lb;	//this will trigger an exception				    											
			__syncthreads();					
			*ry = y; 
			ry += nRng * nscenPerRng; 
			yprevious = y;

          					
			if(++k >= nk)
			{
				k = 0;
				yprevious = y0;
	     		if(++scen >= nscenPerRng) break;
			}					  
    }		//end for(;;)


	//save status vector of random number generator
    rmt =  &mt_rs[iRng];
	__syncthreads();
	
	#pragma unroll      
	for(iState = 0; iState < MT_NN; iState++)
		{  
			*rmt = mt[iState]; rmt += nRng;
		}  
}





extern "C" __declspec( dllexport ) int opcuda_mc1f(unsigned int mtptr, 
													int y0, unsigned int y_sk, int nscen, 
													int nk, int d, const unsigned int ctpker_m_yy,
													const unsigned int m_k, const unsigned long int yhostptr)
{
		
		int n_per_rng = nscen/MT_RNG_COUNT;
		
		int size = nscen * nk * sizeof(short);
		
		int status = 0;
		
		if(d>=1024) return 0;
								
		if(d<=1024 && d>512)
		{
			mc1f_device1024<<<32, 128, nk*sizeof(int)>>> ((unsigned int *) mtptr, y0, 
															(short *) y_sk, n_per_rng, nk, 
															d, (float *) ctpker_m_yy, (int *) m_k); 
		}
						
		if(d<=512 && d>256)
		{
			mc1f_device512<<<32, 128, nk*sizeof(int)>>> ((unsigned int *) mtptr, y0, 
															(short *) y_sk, n_per_rng, nk, 
															d, (float *) ctpker_m_yy, (int *) m_k); 
		}

		if(d<=256 && d>128)
		{
			mc1f_device256<<<32, 128, nk*sizeof(int)>>> ((unsigned int *) mtptr, y0, 
															(short *) y_sk, n_per_rng, nk, 
															d, (float *) ctpker_m_yy, (int *) m_k); 
		}
						
		if(d<=128)
		{
			mc1f_device128<<<32, 128, nk*sizeof(int)>>> ((unsigned int *) mtptr, y0, 
															(short *) y_sk, n_per_rng, nk, 
															d, (float *) ctpker_m_yy, (int *) m_k); 
		}
						
						
        status = cudaMemcpy((void *) yhostptr, (void *) y_sk, size, cudaMemcpyDeviceToHost);
        return status;

}







__global__ void mc1f_mt_benchmark(unsigned int* mt_rs, float * unif_s, int nscenPerRng){

	const int iRng = blockDim.x * blockIdx.x + threadIdx.x;
	const int nRng = blockDim.x * gridDim.x; 		
				         
    int iState, iState1, iStateM;
    unsigned int mti, mti1, mtiM, x;		

    unsigned int mt[MT_NN];
    unsigned *rmt =  &mt_rs[iRng];

	//coalesced read of status vector
	#pragma unroll      
	for(iState = 0; iState < MT_NN; iState++) 
		{  
			mt[iState] = *rmt; rmt += nRng;  
		}
	__syncthreads();

    //Load bit-vector Mersenne Twister parameters   
    mt_struct_stripped config = ds_MT[iRng];

    iState = 0;
    mti1 = mt[0];

	for(int scen = 0; scen < nscenPerRng; scen+= 1){					
	
			//Mersenne Twister
			iState1 = iState + 1;
			iStateM = iState + MT_MM;
			if(iState1 >= MT_NN) iState1 -= MT_NN;
			if(iStateM >= MT_NN) iStateM -= MT_NN;
			mti  = mti1;
			mti1 = mt[iState1];
			mtiM = mt[iStateM];

			x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
			x =  mtiM ^ (x >> 1) ^ ((x & 1) ? config.matrix_a : 0);
			mt[iState] = x;
			iState = iState1;

			//Tempering transformation
			x ^= (x >> MT_SHIFT0);
			x ^= (x << MT_SHIFTB) & config.mask_b;
			x ^= (x << MT_SHIFTC) & config.mask_c;
			x ^= (x >> MT_SHIFT1);
            				 
			//convert to [0, 1) float
			float rand = (float)x / 4294967296.0f;					      					
			__syncthreads();
			unif_s[iRng + nRng * scen] = rand;					      					

    }		
    //end for(;;)

	//save status vector of random number generator
    rmt =  &mt_rs[iRng];
	__syncthreads();
	
	#pragma unroll      
	for(iState = 0; iState < MT_NN; iState++)
		{  
			*rmt = mt[iState]; rmt += nRng;
		}  
}


extern "C" __declspec( dllexport ) int opcuda_mt_benchmark(unsigned int mtptr, unsigned int unif_ptr, int nscen)
{	
		int n_per_rng = nscen/MT_RNG_COUNT;		
		mc1f_mt_benchmark <<<32, 128>>> ((unsigned int *) mtptr, (float *) unif_ptr, n_per_rng); 
        return 0;
}
