// sck.cu --- Part of the project OPLib 1.0, a high performance pricing library
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


#define NTHREADS 4096



#ifdef LINUX
#define __declspec(x)
#define __stdcall
#endif


__global__ void global_sck(const int d, const int nm, float* tpker_yy_m, float* ctpker_yy_m)
{

	  const int tid = blockDim.x * blockIdx.x + threadIdx.x; 
		int d2 = d*d;

		for(int i=0; i < nm * d; i+= NTHREADS)
		{		
				int row, m;
				
				m = (i + tid)/d;
				row = i + tid - m * d;
				
  			if(i + tid < nm * d)
  			{			
						float * tpker = tpker_yy_m + m * d2 + row;
						float * ctpker = ctpker_yy_m + m * d2 + row;
						float rtpker;
						float rctpker;
						float rctpker2;

						rctpker = tpker[0];	
						ctpker[0] = rctpker;
						int y2;
		
						for(y2 = d; y2 < d2; y2 += d )
						{											
								rtpker = tpker [y2];	
								rctpker += rtpker;			
								ctpker[y2] = rctpker;		
						}

						for(y2 = 0; y2 < d2; y2 += d)
						{			
								rctpker2 = ctpker [y2];	
								rctpker2 /= rctpker;
								rctpker2 = min(rctpker2, 1.);   		
								ctpker[y2] = rctpker2;		
						}
				}
		}
};



extern "C" __declspec( dllexport ) void opcuda_sck(const int d, const int nm, const unsigned int tpker_yy_m, const unsigned int ctpker_yy_m)
{	

    global_sck<<< 32, 128 >>>(d, nm, (float *) tpker_yy_m, (float *) ctpker_yy_m);   

}
	
