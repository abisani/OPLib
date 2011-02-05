// sgemv4.cu --- Part of the project OPLib 1.0, a high performance pricing library
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


static __global__ void global_sgemv4(unsigned int * argptr_bid)
{

__shared__ unsigned int arg[10];

	const unsigned int * argptr = (unsigned int *) argptr_bid[blockIdx.x];	
	
	if(threadIdx.x<=9) arg[threadIdx.x] = argptr[threadIdx.x];
    __syncthreads();
	
	const int blockIdx_x = arg[1];
	const int blockIdx_y = arg[2];
	const int d = arg[3];
	const int nz = arg[4];
	float * A = (float *)(arg[6]);
	float * B = (float *)(arg[7]);
	const int * col0_v = (int *)(arg[8]);
	const int * col1_v = (int *)(arg[9]);
	

	const int ibx = blockIdx_x * 64;
	const int iby = blockIdx_y * 16;
	const int row = ibx + threadIdx.x + threadIdx.y * 16;
	const int twonzldc = 2 * nz * d;
	int k = d;

	A += row;
	float *C = B + row; 
	B += threadIdx.x; 
  
	float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

	__shared__ int scol0_v[16];		
	__shared__ int scol1_v[16];		

	if(threadIdx.y == 0) scol0_v[threadIdx.x] = col0_v[iby + threadIdx.x];
	if(threadIdx.y == 1) scol1_v[threadIdx.x] = col1_v[iby + threadIdx.x];
	__syncthreads();

		    
	__shared__ float b[16][17];
	
	for( ; k > 0; k -= 16 )
	{
		b[threadIdx.x][threadIdx.y]  = B[scol0_v[threadIdx.y]];
		b[threadIdx.x][threadIdx.y + 4]  = B[scol0_v[threadIdx.y + 4]];
		b[threadIdx.x][threadIdx.y + 8]  = B[scol0_v[threadIdx.y + 8]];
		b[threadIdx.x][threadIdx.y + 12] = B[scol0_v[threadIdx.y + 12]];
	__syncthreads();
		
		if( k < 16 )  break;

		#pragma unroll
	    for( int i = 0; i < 16; i++, A += d )  rank1_update( A[0], &b[i][0], c ); 
	    __syncthreads();
		B += 16;
	};

    rankk_update( k, A, d, &b[0][0], 17, c );

    if( row >= d ) return;

	int col1;
	col1 = scol1_v[0]; if(col1 >= twonzldc )  return; C[col1] = c[0]; 
	col1 = scol1_v[1]; if(col1 >= twonzldc )  return; C[col1] = c[1]; 
    col1 = scol1_v[2]; if(col1 >= twonzldc )  return; C[col1] = c[2]; 
    col1 = scol1_v[3]; if(col1 >= twonzldc )  return; C[col1] = c[3]; 
    col1 = scol1_v[4]; if(col1 >= twonzldc )  return; C[col1] = c[4]; 
    col1 = scol1_v[5]; if(col1 >= twonzldc )  return; C[col1] = c[5]; 
    col1 = scol1_v[6]; if(col1 >= twonzldc )  return; C[col1] = c[6]; 
    col1 = scol1_v[7]; if(col1 >= twonzldc )  return; C[col1] = c[7]; 
    col1 = scol1_v[8]; if(col1 >= twonzldc )  return; C[col1] = c[8]; 
    col1 = scol1_v[9]; if(col1 >= twonzldc )  return; C[col1] = c[9]; 
    col1 = scol1_v[10]; if(col1 >= twonzldc ) return; C[col1] = c[10]; 
    col1 = scol1_v[11]; if(col1 >= twonzldc ) return; C[col1] = c[11]; 
    col1 = scol1_v[12]; if(col1 >= twonzldc ) return; C[col1] = c[12]; 
    col1 = scol1_v[13]; if(col1 >= twonzldc ) return; C[col1] = c[13]; 
    col1 = scol1_v[14]; if(col1 >= twonzldc ) return; C[col1] = c[14]; 
    col1 = scol1_v[15]; if(col1 >= twonzldc ) return; C[col1] = c[15]; 
    
	};



extern "C" void __declspec( dllexport ) opcuda_sgemv4(int nblocks, unsigned argptr_bid)
{	

  dim3 grid(nblocks, 1), threads( 16, 4 );
  global_sgemv4<<<grid, threads>>>((unsigned int *) argptr_bid);   

}	



