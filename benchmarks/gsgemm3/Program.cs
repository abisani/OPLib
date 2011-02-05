// CPayoff.cs --- Part of the project OPLib 1.0, a high performance pricing library
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




using System;
using System.Collections.Generic;
using System.Text;
using OPModel;
using OPModel.Types;


namespace GPU_SGEMM3
{
    class Program
    {
        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_device_get_count();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_shutdown();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_cublas_init();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_set_device(uint device_number);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private uint opcuda_mem_alloc(uint sz);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_memcpy_h2d(uint dptr, IntPtr hptr, uint sz);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe protected void opcuda_sgemm(int m, int n, int k, float alpha, uint APtr, int lda, uint BPtr, int ldb, float beta, uint CPtr, int ldc);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_thread_synchronize();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_mem_free_device(uint dptr);


        static void Main(string[] args)
        {
            int ndev = opcuda_device_get_count();

            for (uint dev = 0; dev < ndev; dev++)
                run_benchmark_gpu_sgemm(dev);

            Console.Read();
        }

        
        static public void run_benchmark_gpu_sgemm(uint dev)
        {

            Console.WriteLine("running sgemm3 benchmark on device " + dev);

            Random rg = new Random();
            int d = 96 * 6;
            int fsz = 3 * d * d;
            float[] host_float_buf = new float[fsz];

            for (int a = 0; a < host_float_buf.Length; a++)
                host_float_buf[a] = (float)(0.01 * rg.Next(-1000, 1000));

            int status = opcuda_cublas_init();
            if (status != 0) throw new ExecutionEngineException();
            opcuda_set_device(dev);

            uint device_float_buf_ptr = opcuda_mem_alloc((uint)(host_float_buf.Length * sizeof(float)));
            uint aptr = device_float_buf_ptr;
            uint bptr = (uint)(device_float_buf_ptr + d * d * sizeof(float));
            uint cptr = (uint)(device_float_buf_ptr + 2 * d * d * sizeof(float));

            unsafe
            {
                fixed (float* bufp = &host_float_buf[0])
                {
                    status = opcuda_memcpy_h2d(device_float_buf_ptr, (IntPtr)bufp, (uint)(fsz * sizeof(float)));
                    if (status != 0) throw new System.Exception();
                }
            }

            CStopWatch sw = new CStopWatch();
            sw.Reset();

            int niter = 100;

            for (int iter = 0; iter < niter; iter++)
            {
                opcuda_sgemm(d, d, d, 1, aptr, d, bptr, d, 0, cptr, d);
            }

            opcuda_thread_synchronize();
            double time1 = sw.Peek();

            double nflops = 2d * niter * (double)d * (double)d * (double)d;
            double gigaflops_per_second = nflops / (1000000000d * time1);

            opcuda_mem_free_device(device_float_buf_ptr);
            status = opcuda_shutdown();
            if (status != 0) throw new ExecutionEngineException();

            Console.WriteLine("performance: " + String.Format("{0:0.0}", gigaflops_per_second) + " GF/sec");

        }

    }
}
