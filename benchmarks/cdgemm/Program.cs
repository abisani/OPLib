// CProgram.cs --- Part of the project OPLib 1.0, a high performance pricing library
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
using OPModel.Types;


namespace CSV
{
    class Program
    {

        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private void opc_dgemm(double* a, double* b, double* c, int lda, int ldb, int ldc, int m, int n, int k);

        [System.Runtime.InteropServices.DllImport("test")]
        static extern unsafe private int opc_test();

		static void Main(string[] args)
        {

            Console.Write("benchmarking dgemm on the cpu");

            Random rg = new Random();
            int d = 96 * 6;
            int fsz = 3 * d * d;
            double[] host_double_buf = new double[fsz];

            for (int a = 0; a < host_double_buf.Length; a++)
                host_double_buf[a] = (double)(0.01 * rg.Next(-1000, 1000));

           OPModel.CStopWatch sw = new OPModel.CStopWatch();
           sw.Reset();

            int niter = 100;

            unsafe
            {
                fixed (double* ap = &host_double_buf[0])
                {
                    fixed (double* bp = &host_double_buf[d * d])
                    {
                        fixed (double* cp = &host_double_buf[2 * d * d])
                        {
                            for (int iter = 0; iter < niter; iter++)
                            {
                                opc_dgemm(ap, bp, cp, d, d, d, d, d, d);
								opc_test();
                            }
                        }
                    }
                }
            }

            double time1 = sw.Peek();

            double nflops = 2.0 * (double) niter * (double)d * (double)d * (double)d;
            double gigaflops_per_second = nflops/ (1000000000d * time1);

            Console.Write(Environment.NewLine + "performance: " + String.Format("{0:0.0}", gigaflops_per_second) + " GF/sec");
			Console.Read();
        }
    }
}
