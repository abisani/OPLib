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


namespace GPU_SGEMV4
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
        static extern unsafe private int opcuda_memcpy_d2h(uint dptr, IntPtr hptr, uint sz);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_sgemv3(int d, int nz, int ncol, uint A, uint B, uint col0ptr, uint col1ptr);

        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private void opc_sgemv(int m, int n, float alpha, float* a, int lda, float* x, int incx, float beta, float* y, int incy);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_sgemv4(int nblocks, uint argptr_bid);


        static void Main(string[] args)
        {
            int ndev = opcuda_device_get_count();

            for (uint dev = 0; dev < ndev; dev++)
                run_benchmark_gpu_sgemv4(dev);

            Console.Read();
        }


        static public void run_benchmark_gpu_sgemv4(uint dev)
        {

            Console.WriteLine("running sgemv4, device " + dev);

            int ni = 20;
            Random rg = new Random();
            int d = 96 * 6;
            int nz = 1600;
            int[] ncol_i = new int[ni];
            int[][] col0_i_c = new int[ni][];
            int[][] col1_i_c = new int[ni][];
            int[] col0_c = new int[2 * nz];
            int[] col1_c = new int[2 * nz];

            bool[] todo_v = new bool[2 * ni * nz];

            for (int i = 0; i < ni; i++)
            {
                ncol_i[i] = 0;
                for (int iter = 0; iter < 60; iter++)
                {
                    if (rg.Next(2) == 1)
                    {
                        int v0 = 0, v1 = 0;
                        int x = rg.Next(2 * nz);
                        for (int k = 0; k <= x; k++)
                        {
                            v0 += 1;
                            if (v0 >= 2 * nz) v0 -= 2 * nz;
                            while (todo_v[v0])
                            {
                                v0 += 1;
                                if (v0 >= 2 * nz) v0 -= 2 * nz;
                            }
                        }
                        if (todo_v[v0]) throw new System.Exception();
                        todo_v[v0] = true;
                        x = rg.Next(2 * nz);
                        for (int k = 0; k <= x; k++)
                        {
                            v1 += 1;
                            if (v1 >= 2 * nz) v1 -= 2 * nz;
                            while (todo_v[v1])
                            {
                                v1 += 1;
                                if (v1 >= 2 * nz) v1 -= 2 * nz;
                            }
                        }
                        if (todo_v[v1]) throw new System.Exception();
                        todo_v[v1] = true;
                        col0_c[ncol_i[i]] = v0 * d;
                        col1_c[ncol_i[i]] = v1 * d;
                        ncol_i[i] += 1;
                    }
                }

                col0_i_c[i] = new int[16 * ((ncol_i[i] + 15) / 16)];
                col1_i_c[i] = new int[16 * ((ncol_i[i] + 15) / 16)];

                int c;
                for (c = 0; c < ncol_i[i]; c++)
                {
                    col0_i_c[i][c] = col0_c[c];
                    col1_i_c[i][c] = col1_c[c];
                }

                for (; c < col0_i_c[i].Length; c++)
                {
                    col0_i_c[i][c] = 2 * nz * d;
                    col1_i_c[i][c] = 2 * nz * d;
                }
            }


            int fsz = ni * d * d + 2 * d * nz + 1;

            float[] host_float_buf = new float[fsz];

            for (int a = 0; a < host_float_buf.Length; a++)
                host_float_buf[a] = (float)(0.01 * rg.Next(-1000, 1000));

            int status = opcuda_cublas_init();
            if (status != 0) throw new ExecutionEngineException();
            opcuda_set_device(dev);

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
                blockDim_x = (d + 63) / 64;
                blockDim_y = (ncol_i[i] + 15) / 16;
                nblocks += blockDim_x * blockDim_y;
            }

            int bid = 0;
            int[] i_bid = new int[nblocks];
            int[] host_blockIdx_x_bid = new int[nblocks];
            int[] host_blockIdx_y_bid = new int[nblocks];

            for (int i = 0; i < ni; i++)
            {
                int blockDim_x, blockDim_y;
                blockDim_x = (d + 63) / 64;
                blockDim_y = (ncol_i[i] + 15) / 16;
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
            uint Z;

            Z = (uint)ptr;
            ptr += 2 * nz * d * sizeof(float);

            for (int i = 0; i < ni; i++)
            {
                A_i[i] = (uint)ptr;
                ptr += d * d * sizeof(float);
            }

            int nargs = 10;
            int isz = (nargs + 1) * nblocks;
            for (int i = 0; i < ni; i++)
            {
                isz += 16 * 2 * ((ncol_i[i] + 15) / 16);
            }
            uint device_int_buf_ptr = opcuda_mem_alloc((uint)(isz * sizeof(int)));
            uint[] col0_v_ptr_i = new uint[ni];
            uint[] col1_v_ptr_i = new uint[ni];

            long colptr = device_int_buf_ptr + (nargs + 1) * nblocks * sizeof(uint);
            for (int i = 0; i < ni; i++)
            {
                col0_v_ptr_i[i] = (uint)colptr;
                colptr += 16 * ((ncol_i[i] + 15) / 16) * sizeof(uint);
                col1_v_ptr_i[i] = (uint)colptr;
                colptr += 16 * ((ncol_i[i] + 15) / 16) * sizeof(uint);
            }

            uint[] host_int_buf = new uint[isz];
            for (bid = 0; bid < nblocks; bid++)
            {
                int offset = nblocks + nargs * bid;
                host_int_buf[bid] = (uint)(device_int_buf_ptr + offset * sizeof(uint));
                int i = i_bid[bid];
                host_int_buf[offset + 0] = (uint)i;                             //const int i = c[0];
                host_int_buf[offset + 1] = (uint)host_blockIdx_x_bid[bid];      //const int blockIdx_x = c[1];
                host_int_buf[offset + 2] = (uint)host_blockIdx_y_bid[bid];      //const int blockIdx_y = c[2];
                host_int_buf[offset + 3] = (uint)d;                             //const int m = c[3];
                host_int_buf[offset + 4] = (uint)nz;                            //const int n = c[4];
                host_int_buf[offset + 5] = (uint)ncol_i[i];                     //int k = c[5];
                host_int_buf[offset + 6] = A_i[i];                               //float* A = (float*)(c[6]);
                host_int_buf[offset + 7] = Z;                                    //float* B = (float*)(c[7]);
                host_int_buf[offset + 8] = col0_v_ptr_i[i];                               //float* C = (float*)(c[8]);
                host_int_buf[offset + 9] = col1_v_ptr_i[i];                               //float* C = (float*)(c[8]);
            }

            int coffset = nblocks + nargs * nblocks;
            for (int i = 0; i < ni; i++)
            {
                int c = 0;
                int bufsz = 16 * ((ncol_i[i] + 15) / 16);

                for (c = 0; c < ncol_i[i]; c++)
                {
                    host_int_buf[coffset + c] = (uint)col0_i_c[i][c];
                }

                for (; c < bufsz; c++)
                {
                    host_int_buf[coffset + c] = (uint)(2 * nz * d);
                }

                coffset += bufsz;
                for (c = 0; c < ncol_i[i]; c++)
                {
                    host_int_buf[coffset + c] = (uint)col1_i_c[i][c];
                }

                for (; c < bufsz; c++)
                {
                    host_int_buf[coffset + c] = (uint)(2 * nz * d);
                }
                coffset += bufsz;
            }
            float[] host_float_buf2 = new float[fsz];
            uint[] host_int_buf2 = new uint[isz];

            unsafe
            {
                fixed (float* bufp = &host_float_buf[0])
                {
                    status = opcuda_memcpy_h2d(device_float_buf_ptr, (IntPtr)bufp, (uint)(fsz * sizeof(float)));
                    if (status != 0) throw new System.Exception();
                }
                fixed (uint* bufp = &host_int_buf[0])
                {
                    status = opcuda_memcpy_h2d(device_int_buf_ptr, (IntPtr)bufp, (uint)(isz * sizeof(uint)));
                    if (status != 0) throw new System.Exception();
                }
                fixed (float* bufp = &host_float_buf2[0])
                {
                    status = opcuda_memcpy_d2h(device_float_buf_ptr, (IntPtr)bufp, (uint)(fsz * sizeof(float)));
                    if (status != 0) throw new System.Exception();
                }
                fixed (uint* bufp = &host_int_buf2[0])
                {
                    status = opcuda_memcpy_d2h(device_int_buf_ptr, (IntPtr)bufp, (uint)(isz * sizeof(uint)));
                    if (status != 0) throw new System.Exception();
                }
            }


            opcuda_sgemv3(d, nz, ncol_i[0], A_i[0], Z, col0_v_ptr_i[0], col1_v_ptr_i[0]);

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
                        float* Zp = bufp;
                        float* Z2p = buf2p;
                        ptr2 += 2 * d * nz;
                        maxerror2 = 0.0000001f;
                        for (int j = 0; j < d; j++)
                        {
                            for (int v = 0; v < 2 * nz; v++)
                            {
                                if (!todo_v[v])
                                {
                                    errore = Math.Abs(Zp[j + d * v] - Z2p[j + d * v]);
                                    if (errore > maxerror2)
                                        maxerror2 = errore;
                                }
                            }
                        }
                        if (maxerror2 > Math.Pow(10, -6)) throw new System.Exception();
                        int i = 0;
                        float* Ap = bufp + ptr2;
                        float* A2p = buf2p + ptr2;
                        ptr2 += d * d;
                        maxerror2 = 0;
                        for (int j = 0; j < d; j++)
                        {
                            for (int k = 0; k < d; k++)
                            {
                                errore = Math.Abs(Ap[j + d * k] - A2p[j + d * k]);
                                if (errore > maxerror2) maxerror2 = errore;
                            }
                        }
                        if (maxerror2 > Math.Pow(10, -6)) throw new System.Exception();

                        for (int c = 0; c < ncol_i[i]; c++)
                        {
                            opc_sgemv(d, d, 1, Ap, d, Zp + col0_i_c[i][c], 1, 0, Zp + col1_i_c[i][c], 1);
                        }

                        for (int j = 0; j < d; j++)
                        {
                            for (int k = 0; k < nz; k++)
                            {
                                errore = Math.Abs(Zp[j + d * k] - Z2p[j + d * k]) / (1 + Math.Abs(Zp[j + d * k]));
                                if (errore > maxerror1)
                                {
                                    maxerror1 = errore;
                                }
                            }
                        }
                        if (maxerror1 > Math.Pow(10, -3)) throw new System.Exception();
                    }
                }
            }


            unsafe
            {
                fixed (float* bufp = &host_float_buf[0])
                {
                    status = opcuda_memcpy_h2d(device_float_buf_ptr, (IntPtr)bufp, (uint)(fsz * sizeof(float)));
                    if (status != 0) throw new System.Exception();
                }
                fixed (uint* bufp = &host_int_buf[0])
                {
                    status = opcuda_memcpy_h2d(device_int_buf_ptr, (IntPtr)bufp, (uint)(isz * sizeof(uint)));
                    if (status != 0) throw new System.Exception();
                }
            }


            opcuda_sgemv4(nblocks, device_int_buf_ptr);

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

                        float* Zp = bufp;
                        float* Z2p = buf2p;
                        ptr2 += 2 * d * nz;

                        maxerror2 = 0.0000001f;
                        for (int j = 0; j < d; j++)
                        {
                            for (int v = 0; v < 2 * nz; v++)
                            {
                                if (!todo_v[v])
                                {
                                    errore = Math.Abs(Zp[j + d * v] - Z2p[j + d * v]);
                                    if (errore > maxerror2)
                                        maxerror2 = errore;
                                }
                            }
                        }
                        if (maxerror2 > Math.Pow(10, -6)) throw new System.Exception();

                        for (int i = 0; i < ni; i++)
                        {
                            float* Ap = bufp + ptr2;
                            float* A2p = buf2p + ptr2;
                            ptr2 += d * d;

                            maxerror2 = 0;
                            for (int j = 0; j < d; j++)
                            {
                                for (int k = 0; k < d; k++)
                                {
                                    errore = Math.Abs(Ap[j + d * k] - A2p[j + d * k]);
                                    if (errore > maxerror2) maxerror2 = errore;
                                }
                            }
                            if (maxerror2 > Math.Pow(10, -6)) throw new System.Exception();

                            for (int c = 0; c < ncol_i[i]; c++)
                            {
                                opc_sgemv(d, d, 1, Ap, d, Zp + col0_i_c[i][c], 1, 0, Zp + col1_i_c[i][c], 1);
                            }

                        }

                        for (int j = 0; j < d; j++)
                        {
                            for (int k = 0; k < nz; k++)
                            {
                                errore = Math.Abs(Zp[j + d * k] - Z2p[j + d * k]) / (1 + Math.Abs(Zp[j + d * k]));
                                if (errore > maxerror1)
                                {
                                    maxerror1 = errore;
                                }
                            }
                        }

                        if (maxerror1 > Math.Pow(10, -3)) throw new System.Exception();
                    }
                }
            }

            CStopWatch sw = new CStopWatch();
            sw.Reset();

            int niter = 100;

            for (int iter = 0; iter < niter; iter++)
            {
                opcuda_sgemv4(nblocks, device_int_buf_ptr);
            }

            opcuda_thread_synchronize();
            double time1 = sw.Peek();

            double nflops = 0;
            for (int i = 0; i < ni; i++)
            {
                nflops += 2 * d * d * ncol_i[i];
            }
            nflops *= niter;
            double gigaflops_per_second = nflops / (1000000000d * time1);

            status = opcuda_shutdown();
            if (status != 0) throw new ExecutionEngineException();

            Console.WriteLine("performance: " + String.Format("{0:0.0}", gigaflops_per_second) + " GF/sec");

        }

    }
}
