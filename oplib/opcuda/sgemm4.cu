#define NCOLS 4

#ifdef LINUX
#define __declspec(x)
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


static __global__ void global_sgemm4(unsigned int * argptr_bid)
{

	__shared__ unsigned int arg[9];

	const unsigned int * argptr = (unsigned int *) argptr_bid[blockIdx.x];	
	
	if(threadIdx.x<=8) arg[threadIdx.x] = argptr[threadIdx.x];
    __syncthreads();
	

	const int blockIdx_x = arg[1];
	const int blockIdx_y = arg[2];
	const int m = arg[3];
	const int n = arg[4];
	int k = arg[5];
	float * A = (float *)(arg[6]);
	float * B = (float *)(arg[7]);
	float * C = (float *)(arg[8]);

	
	const int inx = threadIdx.x;
	const int iny = threadIdx.y;
	const int ibx = blockIdx_x * 64;
	const int iby = blockIdx_y * 16;
	const int row = ibx + inx + iny*16;


	const int lda = m;
	const int ldb = k;
	const int ldc = m;
	

	A += row;
	B += inx + (iby + iny) * ldb;
	C += row + iby * ldc;
	
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



extern "C" void __declspec( dllexport ) opcuda_sgemm4(int nblocks, unsigned argptr_bid)
{	

  dim3 grid(nblocks, 1), threads( 16, 4 );
  global_sgemm4<<<grid, threads>>>( (unsigned int *) argptr_bid);   

}	


