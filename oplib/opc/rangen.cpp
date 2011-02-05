// rangen.cpp --- Part of the project OPLib 1.0, a high performance pricing library
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


#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"


#ifdef LINUX
#define __declspec(x)
#define __stdcall
#endif

#ifdef DEBUG
		#undef DEBUG
#endif

static	VSLStreamStatePtr* rangen_stream_ptr_th;
static	int rangen_status;
static int nth;


extern "C" __declspec( dllexport ) void opc_rangen_init_seed(unsigned int* seed_th, int nth_)
{

	nth = nth_;
	rangen_stream_ptr_th = new VSLStreamStatePtr[nth]; 
	for(int th = 0; th<nth; th++)
	{
		rangen_stream_ptr_th[th] = new VSLStreamStatePtr(); 
		vslNewStream(&rangen_stream_ptr_th[th], VSL_BRNG_MT2203+th, seed_th[th]);
	}

}


extern "C" __declspec( dllexport ) int opc_rangen_getstatus(void)
{
   return rangen_status;
}


extern "C" __declspec( dllexport ) void opc_rangen_finalize(void)
{
	for(int th = 0; th<nth; th++)
	{
		rangen_status = vslDeleteStream(&rangen_stream_ptr_th[th]);
		if(rangen_status>0) break;
	}
};


extern "C" __declspec( dllexport ) void  opc_gennor(double* r, int n, double mean, double sigma, int th)
{
	rangen_status = vdRngGaussian(VSL_METHOD_DGAUSSIAN_ICDF, rangen_stream_ptr_th[th], n, r, mean, sigma);
}


extern "C" __declspec( dllexport ) void  opc_genunif(double* r, int n, double a, double b, int th)
{
	rangen_status = vdRngUniform(VSL_METHOD_DUNIFORM_STD, rangen_stream_ptr_th[th], n, r, a, b);
}



extern "C" __declspec( dllexport ) int opc_dmc1(double* ctpker_yy_m, int* hash_ys_m, 
					    	int d, int nscen_per_batch, int nk, 
				      		int y0, unsigned int* unif_s, short* y_sk, int* m_k, 
			      			unsigned long payoff_eval_ptr, int th, int b) 
{

	void (__stdcall * payoff_eval)(short* y_sk, int th, int batch) = 
				(void (__stdcall * )(short* y_sk, int th, int batch)) payoff_eval_ptr;

	VSLStreamStatePtr rangen_stream_ptr = rangen_stream_ptr_th[th];

	int niter = (int)(log((double)(d)) / log(2.)) + 1;

	double rand;
	const double max_random = 4294967296.0; 

	for (int k = 0; k < nk; k++)
	{
		rangen_status = viRngUniformBits(VSL_METHOD_IUNIFORMBITS_STD, rangen_stream_ptr, nscen_per_batch, unif_s);
		if(rangen_status!= 0) return rangen_status;

		int m = m_k[k];
		int yprevious = y0;
		short * y_s = &y_sk[nscen_per_batch * k];
		int * hash_ys = &hash_ys_m[d * d * m];
		double * ctpker_yy = &ctpker_yy_m[d * d * m];

		for (int scen = 0; scen < nscen_per_batch; scen++)
		{
	        rand = (double) unif_s[scen]/max_random;
			if(k>0) yprevious = y_s[scen - nscen_per_batch];
			int s = (int)(rand * d);
			int hash = hash_ys[yprevious + d * s];
			short ub = hash >> 16;
			short lb = hash - (ub << 16);

			if(lb==0) if(rand<ctpker_yy[yprevious]) ub = 0;

			if (ub - lb > 1)
			{ 
				double *ctp = &ctpker_yy[yprevious];
				int ymid; 
				for (int iter = 0; iter < niter; iter++)
	            {
		            ymid = (ub + lb) / 2;
				    double ctpker; 
					ctpker = ctp[d * ymid]; 
					if (rand < ctpker) ub = ymid;
				    else lb = ymid;
					if (ub - lb <= 1) break;
		       }
			}
			y_s[scen] = ub;
		}               
	}
	payoff_eval(y_sk, th, b);
	return 0;
}



extern "C" __declspec( dllexport ) int opc_mt_benchmark(int nscen_per_batch, 
														float* unif_s, 
														int th) 
{

	VSLStreamStatePtr rangen_stream_ptr = rangen_stream_ptr_th[th];

	unsigned int * iunif_s = (unsigned int *) unif_s;
	const float max_random = 4294967296.0;

	rangen_status = viRngUniformBits(VSL_METHOD_IUNIFORMBITS_STD, rangen_stream_ptr, nscen_per_batch, iunif_s);

	for(int s = 0; s<nscen_per_batch; s++)
	{
		unif_s[s] = ((float) iunif_s[s])/ max_random;
	}

	return 0;
}








