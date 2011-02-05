// CModelHostDouble.cs --- Part of the project OPLib 1.0, a high performance pricing library
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
using System.Threading;
using OPModel;

namespace OPModel
{

    abstract public partial class CModel
    {

        protected double[] _host_d_gen_yy_i;
        public double[] host_d_lambda_i;

        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private int opc_dmc1(
                                double* ctpker_yy_m, int* hash_ys_m,
                                int d, int nscen_per_batch, int nk,
                                int y0, uint* unif_scen_th, short* y_sk_th, int* m_k,
                                IntPtr payoff_eval_ptr, int th, int b);

        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private void opc_mersenne_twister_benchmark(int nscen_per_batch, int nbatches, int nk, double* unif_sk_th, int nth);

        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private void opc_dgemm(double* a, double* b, double* c, int lda, int ldb, int ldc, int m, int n, int k);

        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private void opc_dcopy(int n, double* destination, double* source);



        abstract public void host_d_gen();

        public double host_d_tpk(int y1, int y2, int m)
        {
            return mcbuf.host_d_tpk_yy_m[y1 + grid.d * y2 + grid.d * grid.d * m];
        }

        public void host_d_sdot2(double[] a_hy, double[] b_hy, ref double[] c_h, int nh, int d)
        {
            if (a_hy.Length != nh * d) throw new System.Exception();
            if (b_hy.Length != nh * d) throw new System.Exception();
            if (c_h.Length != nh) throw new System.Exception();

            for (int h = 0; h < nh; h++)
            {
                c_h[h] = 0;
                for (int y = 0; y < d; y++)
                {
                    c_h[h] += a_hy[h + nh * y] * b_hy[h + nh * y];
                }
            }
        }


        public void host_d_mc_ek()
        {
            int d = grid.d;
            double[] host_xtpk_yy_j = mcbuf.host_d_xtpk_yy_j;
            double[] dt_j = mcplan.host_d_dt_j;

            CArray.alloc(ref mcbuf.host_d_xtpk_yy_j, d * d * mcplan.nj);

            for (int j = 0; j < mcplan.nj; j++)
            {
                int i;
                i = mcplan.i_j[j];

                for (int y1 = 0; y1 < d; y1++)
                {
                    for (int y2 = 0; y2 < d; y2++)
                    {
                        if (y1 != y2)
                        {
                            host_xtpk_yy_j[(y1 + d * y2) + d * d * j] =
                                dt_j[j] * _host_d_gen_yy_i[i * d * d + y1 + d * y2];
                        }
                        else
                        {
                            host_xtpk_yy_j[(y1 + d * y2) + d * d * j] =
                                1 + dt_j[j] * _host_d_gen_yy_i[i * d * d + y1 + d * y2];
                        }
                    }
                }


            }
        }



        void host_d_mc_fex()
        {
            host_d_mc_ek();
            for (int r1 = 0; r1 < mcplan.nr; r1++) host_d_mc_sq(r1);
            host_d_mc_copy();
        }

        void host_d_mc_sq(int r1)
        {

            int d = grid.d;
            CArray.alloc(ref mcbuf.host_d_xtpkbuf_yy_idx, Math.Max(mcplan.nm, mcplan.nj) * d * d);

            for (int j = 0; j < mcplan.nj; j++)
            {
                if (mcplan.niter_j[j] > r1)
                {
                    if (r1 % 2 == 1)
                    {
                        unsafe
                        {
                            fixed (double* tpkbufp = &mcbuf.host_d_xtpkbuf_yy_idx[j * d * d])
                            {
                                fixed (double* tpkp = &mcbuf.host_d_xtpk_yy_j[j * d * d])
                                {
                                    opc_dgemm(tpkbufp, tpkbufp, tpkp, d, d, d, d, d, d);
                                    _cpu_nflops += 2f * d * d * d;
                                }
                            }
                        }
                    }
                    else
                    {
                        unsafe
                        {
                            fixed (double* tpkbufp = &mcbuf.host_d_xtpkbuf_yy_idx[j * d * d])
                            {
                                fixed (double* tpkp = &mcbuf.host_d_xtpk_yy_j[j * d * d])
                                {
                                    opc_dgemm(tpkp, tpkp, tpkbufp, d, d, d, d, d, d);
                                    _cpu_nflops += 2f * d * d * d;
                                }
                            }
                        }
                    }
                }
            }
        }




        void host_d_mc_copy()
        {
            int d = grid.d;

            for (int j = 0; j < mcplan.nj; j++)
            {
                if (mcplan.niter_j[j] % 2 == 1)
                {
                    unsafe
                    {
                        fixed (double* tpkbufp = &mcbuf.host_d_xtpkbuf_yy_idx[j * d * d])
                        {
                            fixed (double* tpkp = &mcbuf.host_d_xtpk_yy_j[j * d * d])
                            {
                                opc_dcopy(d * d, tpkp, tpkbufp);
                            }
                        }
                    }
                }
            }
        }


        void host_d_mc_minit()
        {

            int d = grid.d;
            CArray.alloc(ref mcbuf.host_d_tpk_yy_m, mcplan.nm * d * d);

            for (int m = 0; m < mcplan.nm; m++)
            {
                int j;
                j = mcplan.jfactor_ms[m][0];
                {
                    unsafe
                    {
                        fixed (double* xtpkjp = &mcbuf.host_d_xtpk_yy_j[j * d * d])
                        {
                            fixed (double* tpkmp = &mcbuf.host_d_tpk_yy_m[m * d * d])
                            {
                                opc_dcopy(d * d, tpkmp, xtpkjp);
                            }
                        }
                    }
                }
            }
        }



        void host_d_mc_mk(int s)
        {

            int d = grid.d;

            for (int m = 0; m < mcplan.nm; m++)
            {
                int j;
                if (s < mcplan.jfactor_ms[m].Length)
                {
                    j = mcplan.jfactor_ms[m][s];
                    {
                        if (s % 2 == 1)
                        {
                            unsafe
                            {
                                fixed (double* tpkmp1 = &mcbuf.host_d_xtpkbuf_yy_idx[m * d * d])
                                {
                                    fixed (double* xtpkjp = &mcbuf.host_d_xtpk_yy_j[j * d * d])
                                    {
                                        fixed (double* tpkmp2 = &mcbuf.host_d_tpk_yy_m[m * d * d])
                                        {
                                            opc_dgemm(tpkmp1, xtpkjp, tpkmp2, d, d, d, d, d, d);
                                            _cpu_nflops += 2f * d * d * d;
                                        }
                                    }
                                }
                            }
                        }

                        else
                        {
                            unsafe
                            {
                                fixed (double* tpkmp1 = &mcbuf.host_d_tpk_yy_m[m * d * d])
                                {
                                    fixed (double* tpkjp = &mcbuf.host_d_xtpk_yy_j[j * d * d])
                                    {
                                        fixed (double* tpkmp2 = &mcbuf.host_d_xtpkbuf_yy_idx[m * d * d])
                                        {
                                            opc_dgemm(tpkmp1, tpkjp, tpkmp2, d, d, d, d, d, d);
                                            _cpu_nflops += 2f * d * d * d;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }







        void host_d_mc_mcopy()
        {
            int d = grid.d;

            for (int m = 0; m < mcplan.nm; m++)
            {
                if (mcplan.jfactor_ms[m].Length % 2 == 0)
                {
                    unsafe
                    {
                        fixed (double* tpkmp1 = &mcbuf.host_d_xtpkbuf_yy_idx[m * d * d])
                        {
                            fixed (double* tpkmp2 = &mcbuf.host_d_tpk_yy_m[m * d * d])
                            {
                                opc_dcopy(d * d, tpkmp2, tpkmp1);
                            }
                        }
                    }
                }
            }
        }



        public void host_d_run_mersenne_twister_benchmark(int nth, int nbatches, int nscen_per_batch, int nk, double[] host_d_unif_sk_th)
        {
            if (nbatches % nth != 0) throw new System.Exception();
            if (host_d_unif_sk_th.Length < nth * nscen_per_batch * nk) throw new System.Exception();
            unsafe
            {
                fixed (double* unif_sk_th = &host_d_unif_sk_th[0])
                {
                    opc_mersenne_twister_benchmark(nscen_per_batch, nbatches, nk, unif_sk_th, nth);
                }
            }
        }

        public void host_d_mc_run1f(double[] payoff_a, CMCEvaluator Evaluator)
        {

            OPModel.Types.CJobQueue Queue = new OPModel.Types.CJobQueue();
            object[] p_b = new object[mcplan._nbatches];
            for (int b = 0; b < mcplan._nbatches; b++)
            {
                host_d_mc_run1f_func_input input = new host_d_mc_run1f_func_input();
                input.batch = b;
                input.Evaluator = Evaluator;
                p_b[b] = input;

            }
            Queue.Exec(host_d_mc_run1f_func, null, p_b, mcplan.nth);
        }



        public Object host_d_mc_run1f_func(Object p, int th)
        {
            host_d_mc_run1f_func_input input = (host_d_mc_run1f_func_input)p;

            unsafe
            {
                fixed (double* ctpkerp = &mcbuf.host_d_ctpk_yy_m[0])
                {
                    fixed (int* hashp = &mcbuf._host_hash_ys_m[0])
                    {
                        fixed (uint* unif_scenp = &mcbuf.host_unif_scen_th[mcplan._nscen_per_batch * th])
                        {
                            fixed (short* y_skp = &mcbuf.host_y_sk_th[mcplan._nscen_per_batch * mcplan.nk * th])
                            {
                                fixed (int* mkp = &mcplan.m_k[0])
                                {
                                    int status = opc_dmc1(ctpkerp, hashp, grid.d, mcplan._nscen_per_batch,
                                                mcplan.nk, grid.y0, unif_scenp,
                                                y_skp, mkp,
                                                System.Runtime.InteropServices.Marshal.GetFunctionPointerForDelegate(input.Evaluator.eval),
                                                th, input.batch);
                                    if (status != 0) throw new System.Exception();
                                }
                            }
                        }
                    }
                }
            }
            return null;
        }

        public struct host_d_mc_run1f_func_input
        {
            public CMCEvaluator Evaluator;
            public int batch;
        }

        public struct host_d_mc_run1f_func_return
        {
        }


        public void host_d_setall(ref double[] buf, double c)
        {
            for (int i = 0; i < buf.Length; i++) buf[i] = c;
        }

        public void host_d_copy(ref double[] destination, double[] source)
        {
            CArray.alloc(ref destination, source.Length);
            source.CopyTo(destination, 0);
        }

        public void host_d_mc_init_seed(int seed)
        {
            Types.CRangen.mtinit(mcplan.nth, seed);
        }

        public void host_d_mc_init()
        {
            Types.CRangen.mtinit(mcplan.nth);
        }

        private void host_d_mc_hash()
        {
            int d = grid.d;
            CArray.alloc(ref mcbuf.host_d_ctpk_yy_m, d * d * mcplan.nm);
            CArray.alloc(ref mcbuf._host_hash_ys_m, mcplan.nm * d * d);

            double[] host_tpk_yy_m = mcbuf.host_d_tpk_yy_m;
            double[] host_ctpk_yy_m = mcbuf.host_d_ctpk_yy_m;
            int[] host_hash_ys_m = mcbuf._host_hash_ys_m;

            for (int m = 0; m < mcplan.nm; m++)
            {
                double rker, rcker;
                for (int y1 = 0; y1 < d; y1++)
                {
                    rcker = host_tpk_yy_m[(y1 + d * 0) + d * d * m];
                    host_ctpk_yy_m[(y1 + d * 0) + d * d * m] = rcker;
                    for (int y2 = 1; y2 < d; y2++)
                    {
                        rcker += host_tpk_yy_m[(y1 + d * y2) + d * d * m];
                        host_ctpk_yy_m[(y1 + d * y2) + d * d * m] = rcker;
                    }

                    for (int y2 = 1; y2 < d; y2++)
                    {
                        rker = host_ctpk_yy_m[(y1 + d * y2) + d * d * m];
                        rker /= rcker;
                        rker = Math.Min(rker, 1d);
                        host_ctpk_yy_m[(y1 + d * y2) + d * d * m] = rker;
                    }

                    int lb = 0, ub = 0;

                    for (int s = 0; s < d; s++)
                    {
                        for (; lb < d; lb++)
                        {
                            rcker = host_ctpk_yy_m[(y1 + d * lb) + d * d * m];
                            if (rcker > (float)s / (float)d) break;
                        }

                        if (lb > 0) lb -= 1;

                        for (; ub < d; ub++)
                        {
                            rcker = host_ctpk_yy_m[(y1 + d * ub) + d * d * m];
                            if (rcker > ((float)(s + 1) / (float)d)) break;
                        }

                        if (ub == d) ub -= 1;
                        if (host_ctpk_yy_m[(y1 + d * lb) + d * d * m] >= 1) ub = lb;
                        host_hash_ys_m[y1 + d * s + d * d * m] = lb + (ub << 16);
                    }
                }
            }
        }

    }
}
