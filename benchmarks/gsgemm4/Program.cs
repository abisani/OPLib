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


namespace GPU_SGEMM4
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

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_memcpy_d2h(uint dptr, IntPtr hptr, uint sz);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_sgemm4(int nblocks, uint argptr_bid);

        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private void opc_sgemm(float* a, float* b, float* c, int lda, int ldb, int ldc, int m, int n, int k);



        static void Main(string[] args)
        {
            int ndev = opcuda_device_get_count();

            for (uint dev = 0; dev < ndev; dev++)
                run_benchmark_gpu_sgemm4(dev);

            Console.Read();
        }


        static public void run_benchmark_gpu_sgemm4(uint dev)
        {

            Console.WriteLine("running sgemm4, device " + dev);

            int ni = 40;
            Random rg = new Random();
            int[] m_i = new int[ni];
            int[] n_i = new int[ni];
            int[] k_i = new int[ni];
            int d = 96 * 6;

            int fsz = 0;
            for (int i = 0; i < ni; i++)
            {
                m_i[i] = d;
                n_i[i] = d; // rg.Next(1, 25);
                k_i[i] = d;
            }
            for (int i = 0; i < ni; i++)
            {
                //n_i[i] = (int)((n_i[i] * 400.0) / isum);
                fsz += m_i[i] * k_i[i] + k_i[i] * n_i[i] + m_i[i] * n_i[i];
            }

            float[] host_float_buf = new float[fsz];
            for (int a = 0; a < host_float_buf.Length; a++)
                host_float_buf[a] = (float)(0.01 * rg.Next(-1000, 1000));

            int status = opcuda_cublas_init();
            opcuda_set_device(dev);

            if (status != 0) throw new ExecutionEngineException();
            uint device_float_buf_ptr = opcuda_mem_alloc((uint)(host_float_buf.Length * sizeof(float)));

            unsafe
            {
                fixed (float* bufp = &host_float_buf[0])
                {
                    status = opcuda_memcpy_h2d(device_float_buf_ptr, (IntPtr)bufp, (uint)(fsz * sizeof(float)));
                    if (status != 0) throw new System.Exception();
                }
            }

            int nblocks = 0;
            for (int i = 0; i < ni; i++)
            {
                int blockDim_x, blockDim_y;
                blockDim_x = (m_i[i] + 63) / 64;
                blockDim_y = (n_i[i] + 15) / 16;
                nblocks += blockDim_x * blockDim_y;
            }

            int bid = 0;
            int[] i_bid = new int[nblocks];
            int[] host_blockIdx_x_bid = new int[nblocks];
            int[] host_blockIdx_y_bid = new int[nblocks];

            for (int i = 0; i < ni; i++)
            {
                int blockDim_x, blockDim_y;
                blockDim_x = (m_i[i] + 63) / 64;
                blockDim_y = (n_i[i] + 15) / 16;
                for (int bx = 0; bx < blockDim_x; bx++)
                {
                    for (int by = 0; by < blockDim_y; by++)
                    {
                        i_bid[bid] = i;
                        host_blockIdx_x_bid[bid] = bx;
                        host_blockIdx_y_bid[bid] = by;
                        bid += 1;
                    }
                }
            }

            long ptr = (long)device_float_buf_ptr;
            uint[] A_i = new uint[ni];
            uint[] B_i = new uint[ni];
            uint[] C_i = new uint[ni];

            for (int i = 0; i < ni; i++)
            {
                A_i[i] = (uint)ptr;
                ptr += m_i[i] * k_i[i] * sizeof(float);

                B_i[i] = (uint)ptr;
                ptr += k_i[i] * n_i[i] * sizeof(float);

                C_i[i] = (uint)ptr;
                ptr += m_i[i] * n_i[i] * sizeof(float);
            }

            int isz = nblocks * (1 + 9);
            uint device_int_buf_ptr = opcuda_mem_alloc((uint)(isz * sizeof(int)));
            uint[] host_int_buf = new uint[isz];
            int nargs = 9;

            for (bid = 0; bid < nblocks; bid++)
            {
                host_int_buf[bid] = (uint)(device_int_buf_ptr + (nblocks + nargs * bid) * sizeof(uint));
                int offset = nblocks + nargs * bid;
                int i = i_bid[bid];
                host_int_buf[offset + 0] = (uint)i; //const int i = c[0];
                host_int_buf[offset + 1] = (uint)host_blockIdx_x_bid[bid]; //const int blockIdx_x = c[1];
                host_int_buf[offset + 2] = (uint)host_blockIdx_y_bid[bid]; //const int blockIdx_y = c[2];
                host_int_buf[offset + 3] = (uint)m_i[i]; //const int m = c[3];
                host_int_buf[offset + 4] = (uint)n_i[i]; //const int n = c[4];
                host_int_buf[offset + 5] = (uint)k_i[i]; //int k = c[5];
                host_int_buf[offset + 6] = A_i[i]; //float* A = (float*)(c[6]);
                host_int_buf[offset + 7] = B_i[i]; //float* B = (float*)(c[7]);
                host_int_buf[offset + 8] = C_i[i]; //float* C = (float*)(c[8]);
            }

            float[] host_float_buf2 = new float[fsz];
            unsafe
            {
                fixed (float* bufp = &host_float_buf[0])
                {
                    status = opcuda_memcpy_h2d(device_float_buf_ptr, (IntPtr)bufp, (uint)(fsz * sizeof(float)));
                    if (status != 0) throw new System.Exception();
                }
                fixed (uint* bufp = &host_int_buf[0])
                {
                    status = opcuda_memcpy_h2d(device_int_buf_ptr, (IntPtr)bufp, (uint)(isz * sizeof(float)));
                    if (status != 0) throw new System.Exception();
                }
                fixed (float* bufp = &host_float_buf2[0])
                {
                    status = opcuda_memcpy_d2h(device_float_buf_ptr, (IntPtr)bufp, (uint)(fsz * sizeof(float)));
                    if (status != 0) throw new System.Exception();
                }
            }

            opcuda_sgemm4(nblocks, device_int_buf_ptr);

            unsafe
            {
                fixed (float* buf2p = &host_float_buf2[0])
                {
                    status = opcuda_memcpy_d2h(device_float_buf_ptr, (IntPtr)buf2p, (uint)(fsz * sizeof(float)));
                    if (status != 0) throw new System.Exception();

                    fixed (float* bufp = &host_float_buf[0])
                    {
                        int ptr2 = 0;
                        float errore, maxerror1, maxerror2;
                        maxerror1 = 0;

                        for (int i = 0; i < ni; i++)
                        {
                            float* Ap = bufp + ptr2;
                            float* A2p = buf2p + ptr2;
                            ptr2 += m_i[i] * k_i[i];

                            float* Bp = bufp + ptr2;
                            float* B2p = buf2p + ptr2;
                            ptr2 += k_i[i] * n_i[i];

                            float* Cp = bufp + ptr2;
                            float* C2p = buf2p + ptr2;
                            ptr2 += m_i[i] * n_i[i];

                            maxerror2 = 0;
                            for (int j = 0; j < m_i[i]; j++)
                            {
                                for (int k = 0; k < k_i[i]; k++)
                                {
                                    errore = Math.Abs(Ap[j + m_i[i] * k] - A2p[j + m_i[i] * k]);
                                    if (errore > maxerror2) maxerror2 = errore;
                                }
                            }
                            if (maxerror2 > Math.Pow(10, -6)) throw new System.Exception();

                            maxerror2 = 0;
                            for (int j = 0; j < k_i[i]; j++)
                            {
                                for (int k = 0; k < n_i[i]; k++)
                                {
                                    errore = Math.Abs(Bp[j + k_i[i] * k] - B2p[j + k_i[i] * k]);
                                    if (errore > maxerror2) maxerror2 = errore;
                                }
                            }
                            if (maxerror2 > Math.Pow(10, -6)) throw new System.Exception();

                            opc_sgemm(Ap, Bp, Cp, m_i[i], k_i[i], m_i[i], m_i[i], n_i[i], k_i[i]);
                            for (int j = 0; j < m_i[i]; j++)
                            {
                                for (int k = 0; k < n_i[i]; k++)
                                {
                                    errore = Math.Abs(Cp[j + m_i[i] * k] - C2p[j + m_i[i] * k]) / (1 + Math.Abs(Cp[j + m_i[i] * k]));
                                    if (errore > maxerror1)
                                    {
                                        maxerror1 = errore;
                                     }
                                }
                            }
                        }
                        if (maxerror1 > 2 * Math.Pow(10, -3)) throw new System.Exception();
                    }
                }
            }

            CStopWatch sw = new CStopWatch();
            sw.Reset();
            int niter = 10;
            
            for (int iter = 0; iter < niter; iter++)
            {
                opcuda_sgemm4(nblocks, device_int_buf_ptr);
            }

            opcuda_thread_synchronize();
            double time1 = sw.Peek();

            double nflops = 0;
            for (int i = 0; i < ni; i++)
            {
                nflops += 2 * k_i[i] * m_i[i] * n_i[i];
            }
            nflops *= niter;
            double gigaflops_per_second = nflops / (1000000000d * time1);

            opcuda_mem_free_device(device_int_buf_ptr);
            opcuda_mem_free_device(device_float_buf_ptr);
            status = opcuda_shutdown();
            if (status != 0) throw new ExecutionEngineException();

            Console.WriteLine("performance: " + String.Format("{0:0.0}", gigaflops_per_second) + " GF/sec");

        }

    }
}
