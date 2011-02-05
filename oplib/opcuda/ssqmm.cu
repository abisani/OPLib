// ssqmm --- Part of the project OPLib 1.0, a high performance pricing library
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


#define NCOLS 4


#ifdef LINUX
#define __declspec(x)
#define __stdcall
#endif




__device__ void rank1_update( const float a, const float *b, float *c )
{
	c[0] += a*b[0];
	c[1] += a*b[1];
	c[2] += a*b[2];
	c[3] += a*b[3];
	c[4] += a*b[4];
	c[5] += a*b[5];
	c[6] += a*b[6];
	c[7] += a*b[7];
	c[8] += a*b[8];
	c[9] += a*b[9];
	c[10] += a*b[10];
	c[11] += a*b[11];
	c[12] += a*b[12];
	c[13] += a*b[13];
	c[14] += a*b[14];
	c[15] += a*b[15];
}

__device__ void rankk_update( int k, const float *A0, int lda, const float *b, int ldb, float *c )
{
    if( k <= 0 ) return;
    const float *A = A0;  

    int i = 0;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c );
}


__device__ void store_block2( int num, float *c, float *C0, int ldc )
{
    if( num <= 0 ) return;
    int i = 0; 
    float *C = C0;
    
    C[0] = c[i++]; if( i >= num ) return; C += ldc;  
    C[0] = c[i++]; if( i >= num ) return; C += ldc;  
    C[0] = c[i++]; if( i >= num ) return; C += ldc;  
    C[0] = c[i++]; if( i >= num ) return; C += ldc;  
      
    C[0] = c[i++]; if( i >= num ) return; C += ldc;  
    C[0] = c[i++]; if( i >= num ) return; C += ldc;  
    C[0] = c[i++]; if( i >= num ) return; C += ldc;  
    C[0] = c[i++]; if( i >= num ) return; C += ldc;  

    C[0] = c[i++]; if( i >= num ) return; C += ldc;  
    C[0] = c[i++]; if( i >= num ) return; C += ldc;  
    C[0] = c[i++]; if( i >= num ) return; C += ldc;  
    C[0] = c[i++]; if( i >= num ) return; C += ldc;  

    C[0] = c[i++]; if( i >= num ) return; C += ldc;  
    C[0] = c[i++]; if( i >= num ) return; C += ldc;  
    C[0] = c[i++]; if( i >= num ) return; C += ldc;  
    C[0] = c[i++];
}


static __global__ void global_ssqmm(const int nb, const int d, const int ni, const unsigned A_i, const unsigned B_i, const unsigned C_i)
{

	const int i = blockIdx.x / nb;
	const int inx = threadIdx.x;
	const int iny = threadIdx.y;
	const int ibx = (blockIdx.x % nb) * 64;
	const int iby = blockIdx.y * 16;
	const int row = ibx + inx + iny*16;
	const int lda = d;
	const int ldb = d;
	const int ldc = d;
	const int m = d;
	const int n = d;
	int k = d;
	
	const unsigned * Au_i = (unsigned *) A_i;
	const unsigned * Bu_i = (unsigned *) B_i;
	const unsigned * Cu_i = (unsigned *) C_i;
		
	float * A = (float *)(Au_i[i]);
	float * B = (float *)(Bu_i[i]);
	float * C = (float *)(Cu_i[i]);

	A += ibx + inx + iny * 16;
	B += inx + (iby + iny) * ldb;
	C += ibx + inx + iny * 16 + iby * ldc;
	
	float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
	__shared__ float b[16][17];
	for( ; k > 0; k -= 16 )
	{
	
#pragma unroll
		for(int j = 0; j < 16; j += 4 )	b[inx][iny+j]  = B[j*ldb];
		__syncthreads();
    if( k < 16 )  break;

#pragma unroll
	    for(int j = 0; j < 16; j++, A += lda )    rank1_update( A[0], &b[j][0], c ); 
	    __syncthreads();
  		B += 16;
	};

    rankk_update( k, A, lda, &b[0][0], 17, c );

    if( row >= m ) return;
    
    store_block2( n - iby, c, C, ldc);
	
	};



extern "C" void __declspec( dllexport ) opcuda_ssqmm(int d, int ni, unsigned A_i, unsigned B_i, unsigned C_i)
{	
  const int nb = ((d+63)/64);   
 	dim3 grid( ni * nb, (d+15)/16), threads( 16, 4 );
  global_ssqmm<<<grid, threads>>>(nb, d, ni, A_i,  B_i, C_i);   
}	


