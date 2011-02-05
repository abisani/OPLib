// CBenchmarks.cs --- Part of the project OPLib 1.0, a high performance pricing library
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
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.Diagnostics;
using System.Drawing.Imaging;
using System.Drawing.Printing;
using OPModel;

namespace OPBench
{
    public partial class OPBench : Form
    {

        Label benchmark_cpu_sgemm;
        Label benchmark_cpu_dgemm;
        Label[] benchmark_gpu_sgemm_dev;
        Label[] benchmark_gpu_sgemm4_dev;
        Label[] benchmark_gpu_sgemv4_dev;
        Label[] benchmark_gpu_sgemv2_dev;
        Label[] benchmark_gpu_mt_no_copy_dev;
        Label[] benchmark_gpu_mt_with_copy_dev;
        Label[] benchmark_gpu_sglv1f_blas_dev;
        Label[] benchmark_gpu_sglv1f_mc_dev;
        Label[] benchmark_gpu_sgsv1f_blas_dev;
        Label[] benchmark_gpu_sgsv1f_mc_dev;
        Label benchmark_cpu_dclv1f_blas;
        Label benchmark_cpu_dcsv1f_blas;
        Label benchmark_cpu_dclv1f_mc;
        Label benchmark_cpu_dcsv1f_mc;
        Label benchmark_cpu_mt;
        Label[] benchmark_shared_peak_dev;
        Label[] benchmark_register_peak_dev;

        public class SBenchmarks
        {
            public double cpu_sgemm_performance;
            public double cpu_dgemm_performance;
            public double[] gpu_sgemm_performance_dev;
            public double[] gpu_sgemm4_performance_dev;
            public double[] gpu_sgemv4_performance_dev;
            public double[] gpu_sgemv2_performance_dev;
            public double[] gpu_mt_no_copy_performance_dev;
            public double[] gpu_mt_with_copy_performance_dev;
            public double[] gpu_sglv1f_blas_performance_dev;
            public double[] gpu_sglv1f_mc_performance_dev;
            public double[] gpu_sgsv1f_blas_performance_dev;
            public double[] gpu_sgsv1f_mc_performance_dev;
            public double cpu_dclv1f_blas_performance;
            public double cpu_dcsv1f_blas_performance;
            public double cpu_dclv1f_mc_performance;
            public double cpu_dcsv1f_mc_performance;
            public double cpu_mt_performance;
            public double[] register_peak_performance_dev;
            public double[] shared_peak_performance_dev;
        }

        SBenchmarks benchmarks;

        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private int opc_mt_benchmark(int nscen_per_batch, float* unif_s, int th); 

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_cublas_init();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_thread_synchronize();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private uint opcuda_mem_alloc(uint sz);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe protected void opcuda_sgemm(int m, int n, int k, float alpha, uint APtr, int lda, uint BPtr, int ldb, float beta, uint CPtr, int ldc);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe protected void opcuda_sgemv(int m, int n, float alpha, uint Aptr, int lda,
                                                                uint xptr, int incx, float beta, uint yptr, int incy);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_memcpy_h2d(uint dptr, IntPtr hptr, uint sz);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_memcpy_d2h(uint dptr, IntPtr hptr, uint sz);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_mem_free_device(uint dptr);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_mc_load_mt_gpu();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_mc_setseed(IntPtr host_seedptr, uint mtptr);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_mt_benchmark(uint mtptr, uint unif_ptr, int nscen);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_mc_status_sz();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_mc_nrng();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_shutdown();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_sgemm4(int nblocks, uint argptr_bid);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_sgemv4(int nblocks, uint argptr_bid);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_benchmark_register_peak(uint x1ptr, uint y1ptr, uint x2ptr, uint y2ptr);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_benchmark_shared_peak(uint x1ptr, uint y1ptr, uint x2ptr, uint y2ptr);

        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private void opc_sgemm(float* a, float* b, float* c, int lda, int ldb, int ldc, int m, int n, int k);

        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private void opc_dgemm(double* a, double* b, double* c, int lda, int ldb, int ldc, int m, int n, int k);

        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private void opc_sgemv(int m, int n, float alpha, float* a, int lda, float* x, int incx, float beta, float* y, int incy);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_set_device(uint device_number);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_sgemv3(int d, int nz, int ncol, uint A, uint B, uint col0ptr, uint col1ptr);





        public void run_benchmark_gpu_sgemm4_finalize(IAsyncResult asyncResult)
        {
            //Must call EndInvoke() to prevent resource leaks
            System.Runtime.Remoting.Messaging.AsyncResult
            result = (System.Runtime.Remoting.Messaging.AsyncResult)asyncResult;
            delegate_func_0 doWork = (delegate_func_0)result.AsyncDelegate;
            doWork.EndInvoke(asyncResult);

            ISynchronizeInvoke synchronizer = B_GPU_SGEMM4;
            if (synchronizer.InvokeRequired == false)
            {
                main_run_benchmark_gpu_sgemm4_finalize();
                return;
            }
            delegate_func_0 delegate_run_benchmark_gpu_sgemm4_finalize = new delegate_func_0(main_run_benchmark_gpu_sgemm4_finalize);
            try
            {
                synchronizer.Invoke(delegate_run_benchmark_gpu_sgemm4_finalize, new object[] { });
            }
            catch
            { }
        }


        public void run_benchmark_gpu_sgemv4_finalize(IAsyncResult asyncResult)
        {
            //Must call EndInvoke() to prevent resource leaks
            System.Runtime.Remoting.Messaging.AsyncResult
            result = (System.Runtime.Remoting.Messaging.AsyncResult)asyncResult;
            delegate_func_0 doWork = (delegate_func_0)result.AsyncDelegate;
            doWork.EndInvoke(asyncResult);

            ISynchronizeInvoke synchronizer = B_GPU_SGEMV4;
            if (synchronizer.InvokeRequired == false)
            {
                main_run_benchmark_gpu_sgemv4_finalize();
                return;
            }
            delegate_func_0 delegate_run_benchmark_gpu_sgemv4_finalize = new delegate_func_0(main_run_benchmark_gpu_sgemv4_finalize);
            try
            {
                synchronizer.Invoke(delegate_run_benchmark_gpu_sgemv4_finalize, new object[] { });
            }
            catch
            { }
        }



        public void run_benchmark_gpu_sglv1f_finalize(IAsyncResult asyncResult)
        {
            //Must call EndInvoke() to prevent resource leaks
            System.Runtime.Remoting.Messaging.AsyncResult
            result = (System.Runtime.Remoting.Messaging.AsyncResult)asyncResult;
            delegate_func_0 doWork = (delegate_func_0)result.AsyncDelegate;
            doWork.EndInvoke(asyncResult);

            ISynchronizeInvoke synchronizer = B_GPU_SGLV;
            if (synchronizer.InvokeRequired == false)
            {
                main_run_benchmark_gpu_sglv1f_finalize();
                return;
            }
            delegate_func_0 delegate_run_benchmark_gpu_sglv1f_finalize = new delegate_func_0(main_run_benchmark_gpu_sglv1f_finalize);
            try
            {
                synchronizer.Invoke(delegate_run_benchmark_gpu_sglv1f_finalize, new object[] { });
            }
            catch
            { }
        }



        public void run_benchmark_gpu_mt_finalize(IAsyncResult asyncResult)
        {
            //Must call EndInvoke() to prevent resource leaks
            System.Runtime.Remoting.Messaging.AsyncResult
            result = (System.Runtime.Remoting.Messaging.AsyncResult)asyncResult;
            delegate_func_0 doWork = (delegate_func_0)result.AsyncDelegate;
            doWork.EndInvoke(asyncResult);

            ISynchronizeInvoke synchronizer = B_GPU_SGLV;
            if (synchronizer.InvokeRequired == false)
            {
                main_run_benchmark_gpu_mt_finalize();
                return;
            }
            delegate_func_0 delegate_run_benchmark_gpu_mt_finalize = new delegate_func_0(main_run_benchmark_gpu_mt_finalize);
            try
            {
                synchronizer.Invoke(delegate_run_benchmark_gpu_mt_finalize, new object[] { });
            }
            catch
            { }
        }

        
        public void run_benchmark_gpu_sgsv1f_finalize(IAsyncResult asyncResult)
        {
            //Must call EndInvoke() to prevent resource leaks
            System.Runtime.Remoting.Messaging.AsyncResult
            result = (System.Runtime.Remoting.Messaging.AsyncResult)asyncResult;
            delegate_func_0 doWork = (delegate_func_0)result.AsyncDelegate;
            doWork.EndInvoke(asyncResult);

            ISynchronizeInvoke synchronizer = B_GPU_SGSV;
            if (synchronizer.InvokeRequired == false)
            {
                main_run_benchmark_gpu_sgsv1f_finalize();
                return;
            }
            delegate_func_0 delegate_run_benchmark_gpu_sgsv1f_finalize = new delegate_func_0(main_run_benchmark_gpu_sgsv1f_finalize);
            try
            {
                synchronizer.Invoke(delegate_run_benchmark_gpu_sgsv1f_finalize, new object[] { });
            }
            catch
            { }
        }


        public void run_benchmark_cpu_dclv1f_finalize(IAsyncResult asyncResult)
        {
            //Must call EndInvoke() to prevent resource leaks
            System.Runtime.Remoting.Messaging.AsyncResult
            result = (System.Runtime.Remoting.Messaging.AsyncResult)asyncResult;
            delegate_func_0 doWork = (delegate_func_0)result.AsyncDelegate;
            doWork.EndInvoke(asyncResult);
            ISynchronizeInvoke synchronizer = B_CPU_DCLV1F;
            if (synchronizer.InvokeRequired == false)
            {
                main_run_benchmark_cpu_dclv1f_finalize();
                return;
            }
            delegate_func_0 delegate_run_benchmark_cpu_dclv1f_finalize = new delegate_func_0(main_run_benchmark_cpu_dclv1f_finalize);
            try
            {
                synchronizer.Invoke(delegate_run_benchmark_cpu_dclv1f_finalize, new object[] { });
            }
            catch
            { }
        }


        public void run_benchmark_cpu_mt_finalize(IAsyncResult asyncResult)
        {
            //Must call EndInvoke() to prevent resource leaks
            System.Runtime.Remoting.Messaging.AsyncResult
            result = (System.Runtime.Remoting.Messaging.AsyncResult)asyncResult;
            delegate_func_0 doWork = (delegate_func_0)result.AsyncDelegate;
            doWork.EndInvoke(asyncResult);
            ISynchronizeInvoke synchronizer = B_CPU_MT;
            if (synchronizer.InvokeRequired == false)
            {
                main_run_benchmark_cpu_mt_finalize();
                return;
            }
            delegate_func_0 delegate_run_benchmark_cpu_mt_finalize = new delegate_func_0(main_run_benchmark_cpu_mt_finalize);
            try
            {
                synchronizer.Invoke(delegate_run_benchmark_cpu_mt_finalize, new object[] { });
            }
            catch
            { }
        }

        public void run_benchmark_cpu_dcsv1f_finalize(IAsyncResult asyncResult)
        {
            //Must call EndInvoke() to prevent resource leaks
            System.Runtime.Remoting.Messaging.AsyncResult
            result = (System.Runtime.Remoting.Messaging.AsyncResult)asyncResult;
            delegate_func_0 doWork = (delegate_func_0)result.AsyncDelegate;
            doWork.EndInvoke(asyncResult);
            ISynchronizeInvoke synchronizer = B_CPU_DCSV1F;
            if (synchronizer.InvokeRequired == false)
            {
                main_run_benchmark_cpu_dcsv1f_finalize();
                return;
            }
            delegate_func_0 delegate_run_benchmark_cpu_dcsv1f_finalize = new delegate_func_0(main_run_benchmark_cpu_dcsv1f_finalize);
            try
            {
                synchronizer.Invoke(delegate_run_benchmark_cpu_dcsv1f_finalize, new object[] { });
            }
            catch
            { }
        }



        public void run_benchmark_gpu_sgemm_finalize(IAsyncResult asyncResult)
        {
            //Must call EndInvoke() to prevent resource leaks
            System.Runtime.Remoting.Messaging.AsyncResult
            result = (System.Runtime.Remoting.Messaging.AsyncResult)asyncResult;
            delegate_func_0 doWork = (delegate_func_0)result.AsyncDelegate;
            doWork.EndInvoke(asyncResult);

            ISynchronizeInvoke synchronizer = B_GPU_SGEMM;
            if (synchronizer.InvokeRequired == false)
            {
                main_run_benchmark_gpu_sgemm_finalize();
                return;
            }
            delegate_func_0 delegate_run_benchmark_gpu_sgemm_finalize = new delegate_func_0(main_run_benchmark_gpu_sgemm_finalize);
            try
            {
                synchronizer.Invoke(delegate_run_benchmark_gpu_sgemm_finalize, new object[] { });
            }
            catch
            { }

        }


        public void run_benchmark_gpu_sgemv2_finalize(IAsyncResult asyncResult)
        {
            //Must call EndInvoke() to prevent resource leaks
            System.Runtime.Remoting.Messaging.AsyncResult
            result = (System.Runtime.Remoting.Messaging.AsyncResult)asyncResult;
            delegate_func_0 doWork = (delegate_func_0)result.AsyncDelegate;
            doWork.EndInvoke(asyncResult);

            ISynchronizeInvoke synchronizer = B_GPU_SGEMV2;
            if (synchronizer.InvokeRequired == false)
            {
                main_run_benchmark_gpu_sgemv2_finalize();
                return;
            }
            delegate_func_0 delegate_run_benchmark_gpu_sgemv2_finalize = new delegate_func_0(main_run_benchmark_gpu_sgemv2_finalize);
            try
            {
                synchronizer.Invoke(delegate_run_benchmark_gpu_sgemv2_finalize, new object[] { });
            }
            catch
            { }

        }


        public void run_benchmark_cpu_sgemm_finalize(IAsyncResult asyncResult)
        {
            //Must call EndInvoke() to prevent resource leaks
            System.Runtime.Remoting.Messaging.AsyncResult
            result = (System.Runtime.Remoting.Messaging.AsyncResult)asyncResult;
            delegate_func_0 doWork = (delegate_func_0)result.AsyncDelegate;
            doWork.EndInvoke(asyncResult);

            ISynchronizeInvoke synchronizer = B_CPU_SGEMM;
            if (synchronizer.InvokeRequired == false)
            {
                main_run_benchmark_cpu_sgemm_finalize();
                return;
            }
            delegate_func_0 delegate_run_benchmark_cpu_sgemm_finalize = new delegate_func_0(main_run_benchmark_cpu_sgemm_finalize);
            try
            {
                synchronizer.Invoke(delegate_run_benchmark_cpu_sgemm_finalize, new object[] { });
            }
            catch
            { }

        }


        public void run_benchmark_cpu_dgemm_finalize(IAsyncResult asyncResult)
        {
            //Must call EndInvoke() to prevent resource leaks
            System.Runtime.Remoting.Messaging.AsyncResult
            result = (System.Runtime.Remoting.Messaging.AsyncResult)asyncResult;
            delegate_func_0 doWork = (delegate_func_0)result.AsyncDelegate;
            doWork.EndInvoke(asyncResult);

            ISynchronizeInvoke synchronizer = B_CPU_DGEMM;
            if (synchronizer.InvokeRequired == false)
            {
                main_run_benchmark_cpu_dgemm_finalize();
                return;
            }
            delegate_func_0 delegate_run_benchmark_cpu_dgemm_finalize = new delegate_func_0(main_run_benchmark_cpu_dgemm_finalize);
            try
            {
                synchronizer.Invoke(delegate_run_benchmark_cpu_dgemm_finalize, new object[] { });
            }
            catch
            { }

        }



        public void run_benchmark_gpu_all_finalize(IAsyncResult asyncResult)
        {
            //Must call EndInvoke() to prevent resource leaks
            System.Runtime.Remoting.Messaging.AsyncResult
            result = (System.Runtime.Remoting.Messaging.AsyncResult)asyncResult;
            delegate_func_0 doWork = (delegate_func_0)result.AsyncDelegate;
            doWork.EndInvoke(asyncResult);

            ISynchronizeInvoke synchronizer = B_GPU_RUN_ALL;
            if (synchronizer.InvokeRequired == false)
            {
                main_run_benchmark_cpu_sgemm_finalize();
                return;
            }
            delegate_func_0 delegate_run_benchmark_gpu_all_finalize = new delegate_func_0(main_run_benchmark_gpu_all_finalize);
            try
            {
                synchronizer.Invoke(delegate_run_benchmark_gpu_all_finalize, new object[] { });
            }
            catch
            { }

        }



        public void run_benchmark_cpu_all_finalize(IAsyncResult asyncResult)
        {
            //Must call EndInvoke() to prevent resource leaks
            System.Runtime.Remoting.Messaging.AsyncResult
            result = (System.Runtime.Remoting.Messaging.AsyncResult)asyncResult;
            delegate_func_0 doWork = (delegate_func_0)result.AsyncDelegate;
            doWork.EndInvoke(asyncResult);

            ISynchronizeInvoke synchronizer = B_CPU_RUN_ALL;
            if (synchronizer.InvokeRequired == false)
            {
                main_run_benchmark_cpu_sgemm_finalize();
                return;
            }
            delegate_func_0 delegate_run_benchmark_cpu_all_finalize = new delegate_func_0(main_run_benchmark_cpu_all_finalize);
            try
            {
                synchronizer.Invoke(delegate_run_benchmark_cpu_all_finalize, new object[] { });
            }
            catch
            { }
        }


        public void run_benchmark_register_peak_finalize(IAsyncResult asyncResult)
        {
            //Must call EndInvoke() to prevent resource leaks
            System.Runtime.Remoting.Messaging.AsyncResult
            result = (System.Runtime.Remoting.Messaging.AsyncResult)asyncResult;
            delegate_func_0 doWork = (delegate_func_0)result.AsyncDelegate;
            doWork.EndInvoke(asyncResult);

            ISynchronizeInvoke synchronizer = this.B_GPU_RegisterPeak;
            if (synchronizer.InvokeRequired == false)
            {
                main_run_benchmark_register_peak_finalize();
                return;
            }
            delegate_func_0 delegate_run_benchmark_basic_finalize = new delegate_func_0(main_run_benchmark_register_peak_finalize);
            try
            {
                synchronizer.Invoke(delegate_run_benchmark_basic_finalize, new object[] { });
            }
            catch
            { }

        }


        public void run_benchmark_shared_peak_finalize(IAsyncResult asyncResult)
        {
            //Must call EndInvoke() to prevent resource leaks
            System.Runtime.Remoting.Messaging.AsyncResult
            result = (System.Runtime.Remoting.Messaging.AsyncResult)asyncResult;
            delegate_func_0 doWork = (delegate_func_0)result.AsyncDelegate;
            doWork.EndInvoke(asyncResult);

            ISynchronizeInvoke synchronizer = B_GPU_SharedPeak;
            if (synchronizer.InvokeRequired == false)
            {
                main_run_benchmark_shared_peak_finalize();
                return;
            }
            delegate_func_0 delegate_run_benchmark_basic_finalize = new delegate_func_0(main_run_benchmark_shared_peak_finalize);
            try
            {
                synchronizer.Invoke(delegate_run_benchmark_basic_finalize, new object[] { });
            }
            catch
            { }

        }

        public void run_benchmark_cpu_dclv1f()
        {

            log.Add("running dclv1f on the cpu");
            int ni = 10;
            double S0 = 100;
            int nscen_per_batch = 4096 * 25;
            int nbatches = 120;
            TimeSpan dt0 = TimeSpan.FromDays(1);
            EFloatingPointUnit fpu = EFloatingPointUnit.host;
            EFloatingPointPrecision fpp = EFloatingPointPrecision.bit64;
            DateTime today = DateTime.Today;
            DateTime[] t_k = new DateTime[40];
            for (int k = 0; k < 40; k++) t_k[k] = today.AddDays(7 * (k + 1));
            DateTime[] t_i = new DateTime[ni];
            t_i[0] = today.AddDays(30); t_i[1] = today.AddDays(60);
            t_i[2] = today.AddDays(90); t_i[3] = today.AddDays(120);
            t_i[4] = today.AddDays(150); t_i[5] = today.AddDays(180);
            t_i[6] = today.AddDays(210); t_i[7] = today.AddDays(240);
            t_i[8] = today.AddDays(270); t_i[9] = today.AddDays(300);
            double[] xpivot_p = new double[7];
            double[] xgridspacing_p = new double[7];
            xpivot_p[0] = 1; xgridspacing_p[0] = 5;
            xpivot_p[1] = 70; xgridspacing_p[1] = 2.5;
            xpivot_p[2] = 90; xgridspacing_p[2] = 1;
            xpivot_p[3] = 110; xgridspacing_p[3] = 2.5;
            xpivot_p[4] = 140; xgridspacing_p[4] = 5;
            xpivot_p[5] = 200; xgridspacing_p[5] = 7.5;
            xpivot_p[6] = 300; xgridspacing_p[6] = 10;
            int nx = 128;

            OPModel.Types.CDevice device = new OPModel.Types.CDevice(fpp, fpu, 0);
            OPModel.Types.S1DGrid grid = new OPModel.Types.S1DGrid(device, nx, today, t_i, dt0, S0, xpivot_p, xgridspacing_p);

            double beta_i;
            double[] ir_i = new double[ni];
            double[] df_i = new double[ni];
            double[][] SDrift_i_y = new double[ni][];
            double[][] SVol_i_y = new double[ni][];
            for (int i = 0; i < ni; i++)
            {
                ir_i[i] = 0.05;
                if (i == 0)
                    df_i[0] = Math.Exp(-ir_i[i] * (t_i[0] - grid.today).Days / 365.25);
                else
                    df_i[i] = df_i[i - 1] * Math.Exp(-ir_i[i] * (t_i[i] - t_i[i - 1]).Days / 365.25);

                double Sigma0 = 0.25;
                beta_i = 1;
                SDrift_i_y[i] = new double[grid.d];
                SVol_i_y[i] = new double[grid.d];
                for (int y = 0; y < grid.d; y++)
                {
                    SDrift_i_y[i][y] = ir_i[i] * grid.host_d_xval(y);
                    SVol_i_y[i][y] = Sigma0 * grid.host_d_xval(grid.y0) * Math.Pow(grid.host_d_xval(y) / grid.host_d_xval(grid.y0), beta_i);
                }
            }

            CStopWatch sw = new CStopWatch();
            sw.Reset();
            CLVModel model = new CLVModel(grid, "DCLV1F");

            model.set_discount_curve(df_i);
            model.mkgen(SDrift_i_y, SVol_i_y);
            model.make_mc_plan(nscen_per_batch, nbatches, t_k);

            model.reset_flop_counter();
            double time = sw.Peek();
            sw.Reset();
            model.exe_mc_plan();
            time = sw.Peek();
            double nflops = model.cpu_nflops;

            double gigaflops_per_second = nflops / (1000000000d * time);

            if (benchmarks == null) benchmarks = new SBenchmarks();
            benchmarks.cpu_dclv1f_blas_performance = gigaflops_per_second;
            log.Add("blas performance: " + String.Format("{0:0.0}", gigaflops_per_second) + " GF/sec");

            CMCEvaluator evaluator = new emptyEvaluator();
            model.host_d_mc_init();
            double[] payoff_a = new double[nscen_per_batch * model.mcplan.nth];
            sw.Reset();
            unsafe
            {
                model.host_d_mc_run1f(payoff_a, evaluator);
            }

            time = sw.Peek();

            double nevals = (double)t_k.Length * (double)nbatches * (double)nscen_per_batch;
            double milion_evals_per_second = nevals / (1000000 * time);

            benchmarks.cpu_dclv1f_mc_performance = milion_evals_per_second;
            log.Add("mc performance: " + String.Format("{0:0.0}", milion_evals_per_second) + " milion eval/sec");

	
			opcuda_shutdown();
			
        }


        public void run_benchmark_cpu_dcsv1f()
        {

            log.Add("running dcsv1f on the cpu");

            int ni = 10;
            double S0 = 100;
            int nscen_per_batch = 4096 * 25;
            int nbatches = 100;
            TimeSpan dt0 = TimeSpan.FromDays(1);
            EFloatingPointUnit fpu = EFloatingPointUnit.host;
            EFloatingPointPrecision fpp = EFloatingPointPrecision.bit64;
            DateTime today = DateTime.Today;
            DateTime[] t_k = new DateTime[40];
            for (int k = 0; k < 40; k++) t_k[k] = today.AddDays(7 * (k + 1));
            DateTime[] t_i = new DateTime[ni];
            t_i[0] = today.AddDays(30); t_i[1] = today.AddDays(60);
            t_i[2] = today.AddDays(90); t_i[3] = today.AddDays(120);
            t_i[4] = today.AddDays(150); t_i[5] = today.AddDays(180);
            t_i[6] = today.AddDays(210); t_i[7] = today.AddDays(240);
            t_i[8] = today.AddDays(270); t_i[9] = today.AddDays(300);
            double[] xpivot_p = new double[7];
            double[] xgridspacing_p = new double[7];
            xpivot_p[0] = 1; xgridspacing_p[0] = 5;
            xpivot_p[1] = 70; xgridspacing_p[1] = 2.5;
            xpivot_p[2] = 90; xgridspacing_p[2] = 1;
            xpivot_p[3] = 110; xgridspacing_p[3] = 2.5;
            xpivot_p[4] = 140; xgridspacing_p[4] = 5;
            xpivot_p[5] = 200; xgridspacing_p[5] = 7.5;
            xpivot_p[6] = 300; xgridspacing_p[6] = 10;
            int nx = 64;
            int nr = 8;
            double[] rval_r = new double[nr];
            rval_r[0] = 0.55;
            rval_r[1] = 0.75;
            rval_r[2] = 0.90;
            rval_r[3] = 1.0;
            rval_r[4] = 1.1;
            rval_r[5] = 1.3;
            rval_r[6] = 1.6;
            rval_r[7] = 2.0;
            int r0 = 4;

            OPModel.Types.CDevice device = new OPModel.Types.CDevice(fpp, fpu, 0);
            OPModel.Types.S2DGrid grid = new OPModel.Types.S2DGrid(device, nx, nr, today, t_i, dt0, S0, xpivot_p, xgridspacing_p, rval_r, r0);
            double vol = .25;
            double lowbeta = 0.5;
            double highbeta = 0.5;
            double volvol = 0.5;
            double volmrr = 0.5;
            double volmrl = 3;
            double jumpsz_minus = -3.0;
            double jumpsz_plus = 0.0;

            double[,] taumatrix_ccol = new double[8, 4];
            for (int col = 0; col <= 2; col++)
            {
                taumatrix_ccol[0, col] = vol;
                taumatrix_ccol[1, col] = lowbeta;
                taumatrix_ccol[2, col] = highbeta;
                taumatrix_ccol[3, col] = volvol;
                taumatrix_ccol[4, col] = volmrr;
                taumatrix_ccol[5, col] = volmrl;
                taumatrix_ccol[6, col] = jumpsz_minus;
                taumatrix_ccol[7, col] = jumpsz_plus;
            }

            double[] ir_i = new double[ni];
            double[] df_i = new double[ni];
            for (int i = 0; i < ni; i++)
            {
                ir_i[i] = 0.05;
                if (i == 0) df_i[0] = Math.Exp(-ir_i[i] * (t_i[0] - grid.today).Days / 365.25);
                else df_i[i] = df_i[i - 1] * Math.Exp(-ir_i[i] * (t_i[i] - t_i[i - 1]).Days / 365.25);
            }

            CStopWatch sw = new CStopWatch();
            CSVModel model = new CSVModel(grid, "DCSV1F", df_i);
            model.mkgen(taumatrix_ccol, null);
            model.make_mc_plan(nscen_per_batch, nbatches, t_k);

            model.reset_flop_counter();
            double time = sw.Peek();
            sw.Reset();
            model.exe_mc_plan();
            time = sw.Peek();
            double nflops = model.cpu_nflops;

            double gigaflops_per_second = nflops / (1000000000d * time);
            if (benchmarks == null) benchmarks = new SBenchmarks();
            benchmarks.cpu_dcsv1f_blas_performance = gigaflops_per_second;
            log.Add("blas performance: " + String.Format("{0:0.0}", gigaflops_per_second) + " GF/sec");

            CMCEvaluator evaluator = new emptyEvaluator();
            double[] payoff_a = new double[nscen_per_batch * model.mcplan.nth];

            model.host_d_mc_init();
            sw.Reset();
            unsafe
            {
                model.host_d_mc_run1f(payoff_a, evaluator);
            }
            time = sw.Peek();

            double nevals = (double)t_k.Length * (double)nbatches * (double)nscen_per_batch;
            double milion_evals_per_second = nevals / (1000000 * time);
            benchmarks.cpu_dcsv1f_mc_performance = milion_evals_per_second;
            log.Add("mc performance: " + String.Format("{0:0.0}", milion_evals_per_second) + " milion eval/sec");

        }


        public void run_benchmark_cpu_mt()
        {

            log.Add("running the Mersenne Twister benchmark on the CPU");

            int nscen_per_batch = 4096 * 25;
            int nbatches = 2000;

            int nth = Environment.ProcessorCount;
            OPModel.Types.CRangen.mtinit(nth);

            float[] host_unif_scen_th = new float[nscen_per_batch * nth];

            OPModel.Types.CJobQueue Queue = new OPModel.Types.CJobQueue();
            object[] p_b = new object[nbatches];
            for (int b = 0; b < nbatches; b++)
            {
                host_d_mc_mt_func_input input = new host_d_mc_mt_func_input();
                input.nth = nth;
                input.batch = b;
                input.nscen_per_batch = nscen_per_batch;
                input.host_unif_scen_th = host_unif_scen_th;
                p_b[b] = input;
            }

            CStopWatch sw = new CStopWatch();
            sw.Reset();
            Queue.Exec(host_d_mc_mt_func, null, p_b, nth);
            double time = sw.Peek();

            double nevals = (double)nbatches * (double)nscen_per_batch;
            double milion_evals_per_second = nevals / (1000000 * time);

            if (benchmarks == null) benchmarks = new SBenchmarks();
            benchmarks.cpu_mt_performance = milion_evals_per_second;
            log.Add("mt performance: " + String.Format("{0:0.0}", milion_evals_per_second) + " milion eval/sec");
            host_unif_scen_th = null;
            System.GC.Collect();

        }

        public Object host_d_mc_mt_func(Object p, int th)
        {
            host_d_mc_mt_func_input input = (host_d_mc_mt_func_input)p;
            unsafe
            {
                fixed (float* unif_scenp = &input.host_unif_scen_th[input.nscen_per_batch * th])
                {
                    opc_mt_benchmark(input.nscen_per_batch, unif_scenp, th);
                }
            }
            return null;
        }

        public struct host_d_mc_mt_func_input
        {
            public int nth;
            public float[] host_unif_scen_th;
            public int batch;
            public int nscen_per_batch;
        }

        public struct host_d_mc_mt_func_return
        {
        }



        public void run_benchmark_gpu_sgsv1f()
        {
            for (uint dev = 0; dev < ndev; dev++) run_benchmark_gpu_sgsv1f(dev);
        }


        public void run_benchmark_gpu_sgsv1f(uint dev)
        {

            log.Add("running sgsv1f, device " + dev);

            int ni = 10;
            double S0 = 100;
            int nscen_per_batch = 4096; // 4096 * 25;
            int nbatches = 1; // 10;
            TimeSpan dt0 = TimeSpan.FromDays(1);
            EFloatingPointUnit fpu = EFloatingPointUnit.device;
            EFloatingPointPrecision fpp = EFloatingPointPrecision.bit32;
            DateTime today = DateTime.Today;
            DateTime[] t_k = new DateTime[40];
            for (int k = 0; k < 40; k++) t_k[k] = today.AddDays(7 * (k + 1));
            DateTime[] t_i = new DateTime[ni];
            t_i[0] = today.AddDays(30); t_i[1] = today.AddDays(60);
            t_i[2] = today.AddDays(90); t_i[3] = today.AddDays(120);
            t_i[4] = today.AddDays(150); t_i[5] = today.AddDays(180);
            t_i[6] = today.AddDays(210); t_i[7] = today.AddDays(240);
            t_i[8] = today.AddDays(270); t_i[9] = today.AddDays(300);
            double[] xpivot_p = new double[7];
            double[] xgridspacing_p = new double[7];
            xpivot_p[0] = 1; xgridspacing_p[0] = 5;
            xpivot_p[1] = 70; xgridspacing_p[1] = 2.5;
            xpivot_p[2] = 90; xgridspacing_p[2] = 1;
            xpivot_p[3] = 110; xgridspacing_p[3] = 2.5;
            xpivot_p[4] = 140; xgridspacing_p[4] = 5;
            xpivot_p[5] = 200; xgridspacing_p[5] = 7.5;
            xpivot_p[6] = 300; xgridspacing_p[6] = 10;
            int nx = 64;
            int nr = 8;
            double[] rval_r = new double[nr];
            rval_r[0] = 0.55;
            rval_r[1] = 0.75;
            rval_r[2] = 0.90;
            rval_r[3] = 1.0;
            rval_r[4] = 1.1;
            rval_r[5] = 1.3;
            rval_r[6] = 1.6;
            rval_r[7] = 2.0;
            int r0 = 4;

            OPModel.Types.CDevice device = new OPModel.Types.CDevice(fpp, fpu, dev);
            OPModel.Types.S2DGrid grid = new OPModel.Types.S2DGrid(device, nx, nr, today, t_i, dt0, S0, xpivot_p, xgridspacing_p, rval_r, r0);
            double vol = .25;
            double lowbeta = 0.5;
            double highbeta = 0.5;
            double volvol = 0.5;
            double volmrr = 0.5;
            double volmrl = 3;
            double jumpsz_minus = -3.0;
            double jumpsz_plus = 0.0;

            double[,] taumatrix_ccol = new double[8, 4];

            for (int col = 0; col <= 2; col++)
            {
                taumatrix_ccol[0, col] = vol;
                taumatrix_ccol[1, col] = lowbeta;
                taumatrix_ccol[2, col] = highbeta;
                taumatrix_ccol[3, col] = volvol;
                taumatrix_ccol[4, col] = volmrr;
                taumatrix_ccol[5, col] = volmrl;
                taumatrix_ccol[6, col] = jumpsz_minus;
                taumatrix_ccol[7, col] = jumpsz_plus;
            }


            double[] ir_i = new double[ni];
            double[] df_i = new double[ni];
            for (int i = 0; i < ni; i++)
            {
                ir_i[i] = 0.05;
                if (i == 0)
                    df_i[0] = Math.Exp(-ir_i[i] * (t_i[0] - grid.today).Days / 365.25);
                else
                    df_i[i] = df_i[i - 1] * Math.Exp(-ir_i[i] * (t_i[i] - t_i[i - 1]).Days / 365.25);
            }

            CSVModel model = new CSVModel(grid, "SGSV1F", df_i);

            model.mkgen(taumatrix_ccol, null);

            model.make_mc_plan(nscen_per_batch, nbatches, t_k);

            model.device_thread_synchronize();
            CStopWatch sw = new CStopWatch();
            sw.Reset();
            model.reset_flop_counter();

            model.exe_mc_plan();
            model.device_thread_synchronize();

            double time = sw.Peek();
            double nflops = model.gpu_nflops;

            double gigaflops_per_second = nflops / (1000000000d * time);
            log.Add("blas performance: " + String.Format("{0:0.0}", gigaflops_per_second) + " GF/sec");


            if (benchmarks == null) benchmarks = new SBenchmarks();
            if (benchmarks.gpu_sgsv1f_blas_performance_dev == null)
                benchmarks.gpu_sgsv1f_blas_performance_dev = new double[ndev];

            benchmarks.gpu_sgsv1f_blas_performance_dev[dev] = gigaflops_per_second;

            CMCEvaluator evaluator = new emptyEvaluator();
            double[] pdf_y = new double[grid.d];

            sw.Reset();
            if (model.device_mc_init() == 1)
            {
              MessageBox.Show("device_mc_init() failed", "OPBench", MessageBoxButtons.OK)    ;
              return;
            }
            unsafe
            {
                model.device_mc_run1f(pdf_y, evaluator);
            }
            model.device_thread_synchronize();
            time = sw.Peek();

            double nevals = (double)t_k.Length * (double)nbatches * (double)nscen_per_batch;
            double milion_evals_per_second = nevals / (1000000 * time);

            if (benchmarks == null) benchmarks = new SBenchmarks();
            if (benchmarks.gpu_sgsv1f_mc_performance_dev == null)
                benchmarks.gpu_sgsv1f_mc_performance_dev = new double[ndev];

            benchmarks.gpu_sgsv1f_mc_performance_dev[dev] = milion_evals_per_second;
            log.Add("mc performance: " + String.Format("{0:0.0}", milion_evals_per_second) + " milion eval/sec");

			opcuda_shutdown();

		}


        public void run_benchmark_gpu_mt()
        {
            for (uint dev = 0; dev < ndev; dev++)
                run_benchmark_gpu_mt(dev);
        }


        public void run_benchmark_gpu_sglv1f()
        {
            for (uint dev = 0; dev < ndev; dev++)
                run_benchmark_gpu_sglv1f(dev);
        }


        public void run_benchmark_gpu_mt(uint dev)
        {

            log.Add("running sglv1f, device " + dev);

            int nscen_per_batch = 4096 * 25;
            int nbatches = 20;

            int status = opcuda_cublas_init();
            if (status != 0) throw new ExecutionEngineException();
            opcuda_set_device(dev);

            opcuda_mc_load_mt_gpu();
            Random rand = new Random();
            int nrng = opcuda_mc_nrng();

            CArray host_seed_rg = new CArray(nrng, EType.int_t, EMemorySpace.host, null, "host_seed_rg");
            unsafe
            {
                int* seed_rg = (int*)host_seed_rg.hptr;
                for (int rg = 0; rg < nrng; rg++)
                {
                    seed_rg[rg] = (int)(rand.NextDouble() * int.MaxValue);
                }
            }

            CArray device_rgstatus = new CArray(opcuda_mc_status_sz(), EType.int_t, EMemorySpace.device, null, "mcbuf._device_rgstatus");
            unsafe
            {
                opcuda_mc_setseed(host_seed_rg.hptr, device_rgstatus.ptr);
            }

            CArray device_unif_s = new CArray(nscen_per_batch, EType.float_t, EMemorySpace.device, null, "device_unif_s");
            CArray host_unif_s = new CArray(nscen_per_batch, EType.float_t, EMemorySpace.host, null, "host_unif_s");

            CStopWatch sw = new CStopWatch();
            sw.Reset();
            unsafe
            {
                for (int b = 0; b < nbatches; b++)
                {
                    opcuda_mt_benchmark(device_rgstatus.ptr, device_unif_s.ptr, nscen_per_batch);
                    opcuda_memcpy_d2h(device_unif_s.ptr, host_unif_s.hptr, (uint)(sizeof(short) * host_unif_s.length));
                }
            }
            opcuda_thread_synchronize();
            double time = sw.Peek();

            double nevals = (double)nbatches * (double)nscen_per_batch;
            double milion_evals_per_second = nevals / (1000000 * time);
            status = opcuda_shutdown();
            if (status != 0) throw new ExecutionEngineException();
            if (benchmarks == null) benchmarks = new SBenchmarks();
            if (benchmarks.gpu_mt_with_copy_performance_dev == null)
                benchmarks.gpu_mt_with_copy_performance_dev = new double[ndev];
            benchmarks.gpu_mt_with_copy_performance_dev[dev] = milion_evals_per_second;
            log.Add("mc performance: " + String.Format("{0:0.0}", milion_evals_per_second) + " milion eval/sec");


            sw.Reset();
            unsafe
            {
                for (int b = 0; b < nbatches; b++)
                {
                    opcuda_mt_benchmark(device_rgstatus.ptr, device_unif_s.ptr, nscen_per_batch);
                }
            }
            opcuda_thread_synchronize();
            time = sw.Peek();

            nevals = (double)nbatches * (double)nscen_per_batch;
            milion_evals_per_second = nevals / (1000000 * time);
            status = opcuda_shutdown();
            if (status != 0) throw new ExecutionEngineException();
            if (benchmarks == null) benchmarks = new SBenchmarks();
            if (benchmarks.gpu_mt_no_copy_performance_dev == null)
                benchmarks.gpu_mt_no_copy_performance_dev = new double[ndev];
            benchmarks.gpu_mt_no_copy_performance_dev[dev] = milion_evals_per_second;
            log.Add("mc performance: " + String.Format("{0:0.0}", milion_evals_per_second) + " milion eval/sec");

			opcuda_shutdown();

		}




        public void run_benchmark_gpu_sglv1f(uint dev)
        {

            log.Add("running sglv1f, device " + dev);

            int ni = 10;
            double S0 = 100;
            int nscen_per_batch = 4096 * 25;
            int nbatches = 20;
            TimeSpan dt0 = TimeSpan.FromDays(1);
            EFloatingPointUnit fpu = EFloatingPointUnit.device;
            EFloatingPointPrecision fpp = EFloatingPointPrecision.bit32;
            DateTime today = DateTime.Today;
            DateTime[] t_k = new DateTime[40];
            for (int k = 0; k < 40; k++) t_k[k] = today.AddDays(7 * (k + 1));
            DateTime[] t_i = new DateTime[ni];
            t_i[0] = today.AddDays(30); t_i[1] = today.AddDays(60);
            t_i[2] = today.AddDays(90); t_i[3] = today.AddDays(120);
            t_i[4] = today.AddDays(150); t_i[5] = today.AddDays(180);
            t_i[6] = today.AddDays(210); t_i[7] = today.AddDays(240);
            t_i[8] = today.AddDays(270); t_i[9] = today.AddDays(300);
            double[] xpivot_p = new double[7];
            double[] xgridspacing_p = new double[7];
            xpivot_p[0] = 1; xgridspacing_p[0] = 5;
            xpivot_p[1] = 70; xgridspacing_p[1] = 2.5;
            xpivot_p[2] = 90; xgridspacing_p[2] = 1;
            xpivot_p[3] = 110; xgridspacing_p[3] = 2.5;
            xpivot_p[4] = 140; xgridspacing_p[4] = 5;
            xpivot_p[5] = 200; xgridspacing_p[5] = 7.5;
            xpivot_p[6] = 300; xgridspacing_p[6] = 10;
            int nx = 128;

            OPModel.Types.CDevice device = new OPModel.Types.CDevice(fpp, fpu, dev);
            OPModel.Types.S1DGrid grid = new OPModel.Types.S1DGrid(device, nx, today, t_i, dt0, S0, xpivot_p, xgridspacing_p);

            double beta_i;
            double[] ir_i = new double[ni];
            double[] df_i = new double[ni];
            double[][] SDrift_i_y = new double[ni][];
            double[][] SVol_i_y = new double[ni][];
            for (int i = 0; i < ni; i++)
            {
                ir_i[i] = 0.05;

                if (i == 0)
                    df_i[0] = Math.Exp(-ir_i[i] * (t_i[0] - grid.today).Days / 365.25);
                else
                    df_i[i] = df_i[i - 1] * Math.Exp(-ir_i[i] * (t_i[i] - t_i[i - 1]).Days / 365.25);

                double Sigma0 = 0.25;
                beta_i = 1;
                SDrift_i_y[i] = new double[grid.d];
                SVol_i_y[i] = new double[grid.d];
                for (int y = 0; y < grid.d; y++)
                {
                    SDrift_i_y[i][y] = ir_i[i] * grid.host_d_xval(y);
                    SVol_i_y[i][y] = Sigma0 * grid.host_d_xval(grid.y0) * Math.Pow(grid.host_d_xval(y) / grid.host_d_xval(grid.y0), beta_i);
                }
            }


            CStopWatch sw = new CStopWatch();
            sw.Reset();

            CLVModel model = new CLVModel(grid, "SGLV1F");
            model.set_discount_curve(df_i);
            model.mkgen(SDrift_i_y, SVol_i_y);
            model.make_mc_plan(nscen_per_batch, nbatches, t_k);
            model.reset_flop_counter();
            sw.Reset();
            model.exe_mc_plan();
            model.device_thread_synchronize();
            double time = sw.Peek();
            double nflops = model.gpu_nflops;
            double gigaflops_per_second = nflops / (1000000000d * time);

            if (benchmarks == null) benchmarks = new SBenchmarks();
            if (benchmarks.gpu_sglv1f_blas_performance_dev == null)
                benchmarks.gpu_sglv1f_blas_performance_dev = new double[ndev];

            benchmarks.gpu_sglv1f_blas_performance_dev[dev] = gigaflops_per_second;
            log.Add("blas performance: " + String.Format("{0:0.0}", gigaflops_per_second) + " GF/sec");

            CMCEvaluator evaluator = new emptyEvaluator();
            double[] pdf_y = new double[grid.d];
            if (model.device_mc_init() == 1)
            {
                MessageBox.Show("device_mc_init() failed", "OPBench", MessageBoxButtons.OK);
                return;
            }
            sw.Reset();
            unsafe
            {
                model.device_mc_run1f(pdf_y, evaluator);
            }
            model.device_thread_synchronize();
            time = sw.Peek();

            double nevals = (double)t_k.Length * (double)nbatches * (double)nscen_per_batch;
            double milion_evals_per_second = nevals / (1000000 * time);
            int status = opcuda_shutdown();
            if (status != 0) throw new ExecutionEngineException();
            if (benchmarks == null) benchmarks = new SBenchmarks();
            if (benchmarks.gpu_sglv1f_mc_performance_dev == null)
                benchmarks.gpu_sglv1f_mc_performance_dev = new double[ndev];
            benchmarks.gpu_sglv1f_mc_performance_dev[dev] = milion_evals_per_second;
            log.Add("mc performance: " + String.Format("{0:0.0}", milion_evals_per_second) + " milion eval/sec");

			opcuda_shutdown();
		
		
		}



        public void run_benchmark_gpu_sgemv4()
        {
            for (uint dev = 0; dev < ndev; dev++) 
                run_benchmark_gpu_sgemv4(dev);
        }


        public void run_benchmark_gpu_sgemv4(uint dev)
        {

            log.Add("running sgemv4, device " + dev);

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

            if (benchmarks == null) benchmarks = new SBenchmarks();
            if (benchmarks.gpu_sgemv4_performance_dev == null)
                benchmarks.gpu_sgemv4_performance_dev = new double[ndev];

            benchmarks.gpu_sgemv4_performance_dev[dev] = gigaflops_per_second;

            log.Add("performance: " + String.Format("{0:0.0}", gigaflops_per_second) + " GF/sec");

			opcuda_shutdown();

        }

        public void run_benchmark_gpu_sgemm4()
        {
            for (uint dev = 0; dev < ndev; dev++)
                run_benchmark_gpu_sgemm4(dev);
        }


        public void run_benchmark_gpu_sgemm4(uint dev)
        {

            log.Add("running sgemm4, device " + dev);

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

            if (benchmarks == null) benchmarks = new SBenchmarks();
            if (benchmarks.gpu_sgemm4_performance_dev == null)
                benchmarks.gpu_sgemm4_performance_dev = new double[ndev];

            benchmarks.gpu_sgemm4_performance_dev[dev] = gigaflops_per_second;

            log.Add("performance: " + String.Format("{0:0.0}", gigaflops_per_second) + " GF/sec");

			opcuda_shutdown();
        }


        public void run_benchmark_cpu_all()
        {
            run_benchmark_cpu_sgemm();
            run_benchmark_cpu_dgemm();
            run_benchmark_cpu_mt();
            run_benchmark_cpu_dclv1f();
            run_benchmark_cpu_dcsv1f();
        }

        public void run_benchmark_gpu_all()
        {
            run_benchmark_register_peak();
            run_benchmark_shared_peak();
            run_benchmark_gpu_mt();
            run_benchmark_gpu_sgemv2();
            run_benchmark_gpu_sgemm();
            run_benchmark_gpu_sgemm4();
            run_benchmark_gpu_sgemv4();
            run_benchmark_gpu_sglv1f();
            run_benchmark_gpu_sgsv1f();
        }

        public void run_benchmark_gpu_sgemv2()
        {
            for (uint dev = 0; dev < ndev; dev++)
                run_benchmark_gpu_sgemv2(dev);
        }



        public void run_benchmark_gpu_sgemv2(uint dev)
        {

            log.Add("running sgemv, device " + dev);

            Random rg = new Random();
            int d = 96 * 6;
            int fsz = d * d + 2 * d;
            float[] host_float_buf = new float[fsz];

            for (int a = 0; a < host_float_buf.Length; a++)
                host_float_buf[a] = (float)(0.01 * rg.Next(-1000, 1000));

            int status = opcuda_cublas_init();
            if (status != 0) throw new ExecutionEngineException();
            opcuda_set_device(dev);

            uint device_float_buf_ptr = opcuda_mem_alloc((uint)(host_float_buf.Length * sizeof(float)));
            uint aptr = device_float_buf_ptr;
            uint bptr = (uint)(device_float_buf_ptr + d * d * sizeof(float));
            uint cptr = (uint)(device_float_buf_ptr + (d + d * d) * sizeof(float));

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
                opcuda_sgemv(d, d, 1, aptr, d, bptr, 1, 0, cptr, 1);
            }

            opcuda_thread_synchronize();
            double time1 = sw.Peek();

            double nflops = 2d * niter * (double)d * (double)d;
            double gigaflops_per_second = nflops / (1000000000d * time1);

            opcuda_mem_free_device(device_float_buf_ptr);
            status = opcuda_shutdown();
            if (status != 0) throw new ExecutionEngineException();

            if (benchmarks == null) benchmarks = new SBenchmarks();
            if (benchmarks.gpu_sgemv2_performance_dev == null)
                benchmarks.gpu_sgemv2_performance_dev = new double[ndev];
            benchmarks.gpu_sgemv2_performance_dev[dev] = gigaflops_per_second;
            log.Add("performance: " + String.Format("{0:0.0}", gigaflops_per_second) + " GF/sec");

			opcuda_shutdown();
        }


        public void run_benchmark_gpu_sgemm()
        {
            for (uint dev = 0; dev < ndev; dev++)
                run_benchmark_gpu_sgemm(dev);
        }

        public void run_benchmark_gpu_sgemm(uint dev)
        {

            log.Add("running sgemm, device " + dev);

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

            if (benchmarks == null) benchmarks = new SBenchmarks();
            if (benchmarks.gpu_sgemm_performance_dev == null)
                benchmarks.gpu_sgemm_performance_dev = new double[ndev];
            benchmarks.gpu_sgemm_performance_dev[dev] = gigaflops_per_second;
            log.Add("performance: " + String.Format("{0:0.0}", gigaflops_per_second) + " GF/sec");

			opcuda_shutdown();
        }



        public void run_benchmark_cpu_sgemm()
        {

            log.Add("running sgemm on the cpu");

            Random rg = new Random();
            int d = 96 * 6;
            int fsz = 3 * d * d;
            float[] host_float_buf = new float[fsz];

            for (int a = 0; a < host_float_buf.Length; a++)
                host_float_buf[a] = (float)(0.01 * rg.Next(-1000, 1000));

            CStopWatch sw = new CStopWatch();
            sw.Reset();

            int niter = 100;

            unsafe
            {
                fixed (float* ap = &host_float_buf[0])
                {
                    fixed (float* bp = &host_float_buf[d * d])
                    {
                        fixed (float* cp = &host_float_buf[2 * d * d])
                        {
                            for (int iter = 0; iter < niter; iter++)
                            {
                                opc_sgemm(ap, bp, cp, d, d, d, d, d, d);
                            }
                        }
                    }
                }
            }

            double time1 = sw.Peek();

            double nflops = 2d * niter * (double)d * (double)d * (double)d;
            double gigaflops_per_second = nflops / (1000000000d * time1);

            if (benchmarks == null) benchmarks = new SBenchmarks();

            benchmarks.cpu_sgemm_performance = gigaflops_per_second;

        }




        public void run_benchmark_cpu_dgemm()
        {

            log.Add("running dgemm on the cpu");

            Random rg = new Random();
            int d = 96 * 6;
            int fsz = 3 * d * d;
            double[] host_double_buf = new double[fsz];

            for (int a = 0; a < host_double_buf.Length; a++)
                host_double_buf[a] = (double)(0.01 * rg.Next(-1000, 1000));

            CStopWatch sw = new CStopWatch();
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
                            }
                        }
                    }
                }
            }

            double time1 = sw.Peek();

            double nflops = 2d * niter * (double)d * (double)d * (double)d;
            double gigaflops_per_second = nflops / (1000000000d * time1);

            if (benchmarks == null) benchmarks = new SBenchmarks();

            benchmarks.cpu_dgemm_performance = gigaflops_per_second;

        }


        public void run_benchmark_register_peak()
        {
            for (uint dev = 0; dev < ndev; dev++)
                run_benchmark_register_peak(dev);
        }


        public void run_benchmark_register_peak(uint dev)
        {

            log.Add("running register_peak, device " + dev);

            CStopWatch sw = new CStopWatch();
            sw.Reset();

            int status = opcuda_cublas_init();
            if (status != 0) throw new ExecutionEngineException();

            opcuda_set_device(dev);

            int nblocks = 128;
            int block_dim = 64;

            float[] x1 = new float[128 * nblocks];
            float[] y1 = new float[128 * nblocks];
            float[] x2 = new float[128 * nblocks];
            float[] y2 = new float[128 * nblocks];
            Random rg = new Random();

            for (int i = 0; i < block_dim; i++)
            {
                double alpha = (float)rg.Next(-1000, 1000) / (float)1000;
                x1[i] = (float)Math.Cos(alpha);
                y1[i] = (float)Math.Sin(alpha);
                double beta = (float)rg.Next(-1000, 1000) / (float)1000;
                x2[i] = (float)Math.Cos(beta);
                y2[i] = (float)Math.Sin(beta);
            }
            uint x1ptr = opcuda_mem_alloc((uint)(block_dim * nblocks * sizeof(float)));
            uint y1ptr = opcuda_mem_alloc((uint)(block_dim * nblocks * sizeof(float)));
            uint x2ptr = opcuda_mem_alloc((uint)(block_dim * nblocks * sizeof(float)));
            uint y2ptr = opcuda_mem_alloc((uint)(block_dim * nblocks * sizeof(float)));

            unsafe
            {
                fixed (float* x1p = &x1[0])
                {
                    fixed (float* y1p = &y1[0])
                    {
                        status = opcuda_memcpy_h2d(x1ptr, (IntPtr)x1p, (uint)(block_dim * nblocks * sizeof(float)));
                        if (status != 0) throw new System.Exception();
                        status = opcuda_memcpy_h2d(y1ptr, (IntPtr)y1p, (uint)(block_dim * nblocks * sizeof(float)));
                        if (status != 0) throw new System.Exception();
                    }
                }
                fixed (float* x2p = &x2[0])
                {
                    fixed (float* y2p = &y2[0])
                    {
                        status = opcuda_memcpy_h2d(x2ptr, (IntPtr)x2p, (uint)(block_dim * nblocks * sizeof(float)));
                        if (status != 0) throw new System.Exception();
                        status = opcuda_memcpy_h2d(y2ptr, (IntPtr)y2p, (uint)(block_dim * nblocks * sizeof(float)));
                        if (status != 0) throw new System.Exception();
                    }
                }
            }

            opcuda_thread_synchronize();

            int niter = 100;
            sw.Reset();
            double time0 = sw.Peek();
            for (int iter = 0; iter < niter; iter++)
            {
                opcuda_benchmark_register_peak(x1ptr, y1ptr, x2ptr, y2ptr);
            }

            opcuda_thread_synchronize();
            double time1 = sw.Peek();

            unsafe
            {
                fixed (float* x1p = &x1[0])
                {
                    fixed (float* y1p = &y1[0])
                    {
                        status = opcuda_memcpy_d2h(x1ptr, (IntPtr)x1p, (uint)(128 * nblocks * sizeof(float)));
                        if (status != 0) throw new System.Exception();
                        status = opcuda_memcpy_d2h(y1ptr, (IntPtr)y1p, (uint)(128 * nblocks * sizeof(float)));
                        if (status != 0) throw new System.Exception();
                    }
                }
                fixed (float* x2p = &x2[0])
                {
                    fixed (float* y2p = &y2[0])
                    {
                        status = opcuda_memcpy_d2h(x2ptr, (IntPtr)x2p, (uint)(block_dim * nblocks * sizeof(float)));
                        if (status != 0) throw new System.Exception();
                        status = opcuda_memcpy_d2h(y2ptr, (IntPtr)y2p, (uint)(block_dim * nblocks * sizeof(float)));
                        if (status != 0) throw new System.Exception();
                    }
                }
            }

            double nflops = (double)niter * 6 * 8 * 300d * nblocks * (double)block_dim;
            double gigaflops_per_second = nflops / (1000000000d * (time1 - time0));
            if (benchmarks == null) benchmarks = new SBenchmarks();
            if (benchmarks.register_peak_performance_dev == null)
                benchmarks.register_peak_performance_dev = new double[ndev];
            benchmarks.register_peak_performance_dev[dev] = gigaflops_per_second;

            status = opcuda_shutdown();

        }


        public void run_benchmark_shared_peak()
        {
            for (uint dev = 0; dev < ndev; dev++)
                run_benchmark_shared_peak(dev);
        }

        public void run_benchmark_shared_peak(uint dev)
        {

            log.Add("running shared_peak, device " + dev);

            CStopWatch sw = new CStopWatch();
            sw.Reset();
            int status = opcuda_cublas_init();
            if (status != 0) throw new ExecutionEngineException();
            opcuda_set_device(dev);

            int nblocks = 128;
            int block_dim = 64;

            float[] x1 = new float[128 * nblocks];
            float[] y1 = new float[128 * nblocks];
            float[] x2 = new float[128 * nblocks];
            float[] y2 = new float[128 * nblocks];
            Random rg = new Random();

            for (int i = 0; i < block_dim; i++)
            {
                double alpha = (float)rg.Next(-1000, 1000) / (float)1000;
                x1[i] = (float)Math.Cos(alpha);
                y1[i] = (float)Math.Sin(alpha);
                double beta = (float)rg.Next(-1000, 1000) / (float)1000;
                x2[i] = (float)Math.Cos(beta);
                y2[i] = (float)Math.Sin(beta);
            }
            uint x1ptr = opcuda_mem_alloc((uint)(block_dim * nblocks * sizeof(float)));
            uint y1ptr = opcuda_mem_alloc((uint)(block_dim * nblocks * sizeof(float)));
            uint x2ptr = opcuda_mem_alloc((uint)(block_dim * nblocks * sizeof(float)));
            uint y2ptr = opcuda_mem_alloc((uint)(block_dim * nblocks * sizeof(float)));

            unsafe
            {
                fixed (float* x1p = &x1[0])
                {
                    fixed (float* y1p = &y1[0])
                    {
                        status = opcuda_memcpy_h2d(x1ptr, (IntPtr)x1p, (uint)(block_dim * nblocks * sizeof(float)));
                        if (status != 0) throw new System.Exception();
                        status = opcuda_memcpy_h2d(y1ptr, (IntPtr)y1p, (uint)(block_dim * nblocks * sizeof(float)));
                        if (status != 0) throw new System.Exception();
                    }
                }
                fixed (float* x2p = &x2[0])
                {
                    fixed (float* y2p = &y2[0])
                    {
                        status = opcuda_memcpy_h2d(x2ptr, (IntPtr)x2p, (uint)(block_dim * nblocks * sizeof(float)));
                        if (status != 0) throw new System.Exception();
                        status = opcuda_memcpy_h2d(y2ptr, (IntPtr)y2p, (uint)(block_dim * nblocks * sizeof(float)));
                        if (status != 0) throw new System.Exception();
                    }
                }
            }

            opcuda_thread_synchronize();

            int niter = 100;
            sw.Reset();
            double time0 = sw.Peek();
            for (int iter = 0; iter < niter; iter++)
            {
                opcuda_benchmark_shared_peak(x1ptr, y1ptr, x2ptr, y2ptr);
            }

            opcuda_thread_synchronize();
            double time1 = sw.Peek();

            unsafe
            {
                fixed (float* x1p = &x1[0])
                {
                    fixed (float* y1p = &y1[0])
                    {
                        status = opcuda_memcpy_d2h(x1ptr, (IntPtr)x1p, (uint)(128 * nblocks * sizeof(float)));
                        if (status != 0) throw new System.Exception();
                        status = opcuda_memcpy_d2h(y1ptr, (IntPtr)y1p, (uint)(128 * nblocks * sizeof(float)));
                        if (status != 0) throw new System.Exception();
                    }
                }
                fixed (float* x2p = &x2[0])
                {
                    fixed (float* y2p = &y2[0])
                    {
                        status = opcuda_memcpy_d2h(x2ptr, (IntPtr)x2p, (uint)(block_dim * nblocks * sizeof(float)));
                        if (status != 0) throw new System.Exception();
                        status = opcuda_memcpy_d2h(y2ptr, (IntPtr)y2p, (uint)(block_dim * nblocks * sizeof(float)));
                        if (status != 0) throw new System.Exception();
                    }
                }
            }

            double nflops = (double)niter * 4 * 8 * 300d * nblocks * (double)block_dim;
            double gigaflops_per_second = nflops / (1000000000d * (time1 - time0));
            if (benchmarks == null) benchmarks = new SBenchmarks();
            if (benchmarks.shared_peak_performance_dev == null)
                benchmarks.shared_peak_performance_dev = new double[ndev];

            benchmarks.shared_peak_performance_dev[dev] = gigaflops_per_second;

            status = opcuda_shutdown();

        }


        public void main_run_benchmark_gpu_sgemm4_finalize()
        {
            if (benchmark_gpu_sgemm4_dev == null) benchmark_gpu_sgemm4_dev = new Label[ndev];
            for (int dev = 0; dev < ndev; dev++)
            {
                if (benchmark_gpu_sgemm4_dev[dev] == null) benchmark_gpu_sgemm4_dev[dev] = new Label();
                benchmark_gpu_sgemm4_dev[dev].Parent = GBGPUBenchmarks;
                benchmark_gpu_sgemm4_dev[dev].Location = new Point(B_GPU_SGEMM4.Location.X + (1 + dev) * B_GPU_SGEMM4.Width + 20, B_GPU_SGEMM4.Location.Y + 2);
                benchmark_gpu_sgemm4_dev[dev].Size = new Size(100, 20);
                benchmark_gpu_sgemm4_dev[dev].Text = String.Format("{0:0.0}", benchmarks.gpu_sgemm4_performance_dev[dev]) + " GF/sec";
                benchmark_gpu_sgemm4_dev[dev].Show();
            }
            B_GPU_SGEMM4.Enabled = true;
        }


        public void main_run_benchmark_gpu_sgemv4_finalize()
        {
            if (benchmark_gpu_sgemv4_dev == null) benchmark_gpu_sgemv4_dev = new Label[ndev];
            for (int dev = 0; dev < ndev; dev++)
            {
                if (benchmark_gpu_sgemv4_dev[dev] == null) benchmark_gpu_sgemv4_dev[dev] = new Label();
                benchmark_gpu_sgemv4_dev[dev].Parent = GBGPUBenchmarks;
                benchmark_gpu_sgemv4_dev[dev].Location = new Point(B_GPU_SGEMV4.Location.X + (1 + dev) * B_GPU_SGEMV4.Width + 20, B_GPU_SGEMV4.Location.Y + 2);
                benchmark_gpu_sgemv4_dev[dev].Size = new Size(100, 20);
                benchmark_gpu_sgemv4_dev[dev].Text = String.Format("{0:0.0}", benchmarks.gpu_sgemv4_performance_dev[dev]) + " GF/sec";
                benchmark_gpu_sgemv4_dev[dev].Show();
            }
            B_GPU_SGEMV4.Enabled = true;
        }

        public void main_run_benchmark_gpu_sglv1f_finalize()
        {
            if (benchmark_gpu_sglv1f_blas_dev == null) benchmark_gpu_sglv1f_blas_dev = new Label[ndev];
            for (int dev = 0; dev < ndev; dev++)
            {
                if (benchmark_gpu_sglv1f_blas_dev[dev] == null) benchmark_gpu_sglv1f_blas_dev[dev] = new Label();
                benchmark_gpu_sglv1f_blas_dev[dev].Parent = GBGPUBenchmarks;
                benchmark_gpu_sglv1f_blas_dev[dev].Location = new Point(B_GPU_SGLV.Location.X + (1 + dev) * B_GPU_SGLV.Width + 20, B_GPU_SGLV.Location.Y + 23);
                benchmark_gpu_sglv1f_blas_dev[dev].Size = new Size(B_GPU_SGLV.Width, 20);
                benchmark_gpu_sglv1f_blas_dev[dev].Text = String.Format("{0:0.0}", benchmarks.gpu_sglv1f_blas_performance_dev[dev]) + " GF/sec";
                benchmark_gpu_sglv1f_blas_dev[dev].Show();

            }

            if (benchmark_gpu_sglv1f_mc_dev == null) benchmark_gpu_sglv1f_mc_dev = new Label[ndev];
            for (int dev = 0; dev < ndev; dev++)
            {
                if (benchmark_gpu_sglv1f_mc_dev[dev] == null) benchmark_gpu_sglv1f_mc_dev[dev] = new Label();
                benchmark_gpu_sglv1f_mc_dev[dev].Parent = GBGPUBenchmarks;
                benchmark_gpu_sglv1f_mc_dev[dev].Location = new Point(B_GPU_SGLV.Location.X + (1 + dev) * B_GPU_SGLV.Width + 20, B_GPU_SGLV.Location.Y + 2);
                benchmark_gpu_sglv1f_mc_dev[dev].Size = new Size(B_GPU_SGLV.Width, 20);
                benchmark_gpu_sglv1f_mc_dev[dev].Text = String.Format("{0:0.0}", benchmarks.gpu_sglv1f_mc_performance_dev[dev]) + " milion eval/sec";
                benchmark_gpu_sglv1f_mc_dev[dev].Show();

            }
            B_GPU_SGLV.Enabled = true;
        }


        public void main_run_benchmark_gpu_sgsv1f_finalize()
        {
            if (benchmark_gpu_sgsv1f_blas_dev == null) benchmark_gpu_sgsv1f_blas_dev = new Label[ndev];
            for (int dev = 0; dev < ndev; dev++)
            {
                if (benchmark_gpu_sgsv1f_blas_dev[dev] == null) benchmark_gpu_sgsv1f_blas_dev[dev] = new Label();
                benchmark_gpu_sgsv1f_blas_dev[dev].Parent = GBGPUBenchmarks;
                benchmark_gpu_sgsv1f_blas_dev[dev].Location = new Point(B_GPU_SGSV.Location.X + (1 + dev) * B_GPU_SGSV.Width + 20, B_GPU_SGSV.Location.Y + 23);
                benchmark_gpu_sgsv1f_blas_dev[dev].Size = new Size(B_GPU_SGSV.Width, 20);
                benchmark_gpu_sgsv1f_blas_dev[dev].Text = String.Format("{0:0.0}", benchmarks.gpu_sgsv1f_blas_performance_dev[dev]) + " GF/sec";
                benchmark_gpu_sgsv1f_blas_dev[dev].Show();

            }

            if (benchmark_gpu_sgsv1f_mc_dev == null) benchmark_gpu_sgsv1f_mc_dev = new Label[ndev];
            for (int dev = 0; dev < ndev; dev++)
            {
                if (benchmark_gpu_sgsv1f_mc_dev[dev] == null) benchmark_gpu_sgsv1f_mc_dev[dev] = new Label();
                benchmark_gpu_sgsv1f_mc_dev[dev].Parent = GBGPUBenchmarks;
                benchmark_gpu_sgsv1f_mc_dev[dev].Location = new Point(B_GPU_SGSV.Location.X + (1 + dev) * B_GPU_SGSV.Width + 20, B_GPU_SGSV.Location.Y + 2);
                benchmark_gpu_sgsv1f_mc_dev[dev].Size = new Size(B_GPU_SGSV.Width, 20);
                benchmark_gpu_sgsv1f_mc_dev[dev].Text = String.Format("{0:0.0}", benchmarks.gpu_sgsv1f_mc_performance_dev[dev]) + " milion eval/sec";
                benchmark_gpu_sgsv1f_mc_dev[dev].Show();

            }
            B_GPU_SGSV.Enabled = true;
        }



        public void main_run_benchmark_cpu_dclv1f_finalize()
        {
            if (benchmark_cpu_dclv1f_mc == null) benchmark_cpu_dclv1f_mc = new Label();
            benchmark_cpu_dclv1f_mc.Parent = GBCPUBenchmarks;
            benchmark_cpu_dclv1f_mc.Location = new Point(B_CPU_DCLV1F.Location.X + B_CPU_DCLV1F.Width + 20, B_CPU_DCLV1F.Location.Y + 2);
            benchmark_cpu_dclv1f_mc.Size = new Size(300, 20);
            benchmark_cpu_dclv1f_mc.Text = String.Format("{0:0.0}", benchmarks.cpu_dclv1f_mc_performance) + " milion eval/sec";
            benchmark_cpu_dclv1f_mc.Show();

            if (benchmark_cpu_dclv1f_blas == null) benchmark_cpu_dclv1f_blas = new Label();
            benchmark_cpu_dclv1f_blas.Parent = GBCPUBenchmarks;
            benchmark_cpu_dclv1f_blas.Location = new Point(B_CPU_DCLV1F.Location.X + B_CPU_DCLV1F.Width + 20, B_CPU_DCLV1F.Location.Y + 23);
            benchmark_cpu_dclv1f_blas.Size = new Size(300, 20);
            benchmark_cpu_dclv1f_blas.Text = String.Format("{0:0.0}", benchmarks.cpu_dclv1f_blas_performance) + " GF/sec";
            benchmark_cpu_dclv1f_blas.Show();
            B_CPU_DCLV1F.Enabled = true;

        }



        public void main_run_benchmark_cpu_dcsv1f_finalize()
        {
            if (benchmark_cpu_dcsv1f_mc == null) benchmark_cpu_dcsv1f_mc = new Label();
            benchmark_cpu_dcsv1f_mc.Parent = GBCPUBenchmarks;
            benchmark_cpu_dcsv1f_mc.Location = new Point(this.B_CPU_DCSV1F.Location.X + B_CPU_DCSV1F.Width + 20, B_CPU_DCSV1F.Location.Y + 2);
            benchmark_cpu_dcsv1f_mc.Size = new Size(300, 20);
            benchmark_cpu_dcsv1f_mc.Text = String.Format("{0:0.0}", benchmarks.cpu_dcsv1f_mc_performance) + " milion eval/sec";
            benchmark_cpu_dcsv1f_mc.Show();

            if (benchmark_cpu_dcsv1f_blas == null) benchmark_cpu_dcsv1f_blas = new Label();
            benchmark_cpu_dcsv1f_blas.Parent = GBCPUBenchmarks;
            benchmark_cpu_dcsv1f_blas.Location = new Point(B_CPU_DCSV1F.Location.X + B_CPU_DCSV1F.Width + 20, B_CPU_DCSV1F.Location.Y + 23);
            benchmark_cpu_dcsv1f_blas.Size = new Size(300, 20);
            benchmark_cpu_dcsv1f_blas.Text = String.Format("{0:0.0}", benchmarks.cpu_dcsv1f_blas_performance) + " GF/sec";
            benchmark_cpu_dcsv1f_blas.Show();
            B_CPU_DCSV1F.Enabled = true;
        }

        public void main_run_benchmark_gpu_all_finalize()
        {
            main_run_benchmark_shared_peak_finalize();
            main_run_benchmark_register_peak_finalize();
            main_run_benchmark_gpu_sgemv2_finalize();
            main_run_benchmark_gpu_mt_finalize();
            main_run_benchmark_gpu_sgemm_finalize();
            main_run_benchmark_gpu_sgemm4_finalize();
            main_run_benchmark_gpu_sgemv4_finalize();
            main_run_benchmark_gpu_sglv1f_finalize();
            main_run_benchmark_gpu_sgsv1f_finalize();
            B_GPU_RUN_ALL.Enabled = true;
        }

        public void main_run_benchmark_cpu_all_finalize()
        {
            main_run_benchmark_cpu_sgemm_finalize();
            main_run_benchmark_cpu_dgemm_finalize();
            main_run_benchmark_cpu_mt_finalize();
            main_run_benchmark_cpu_dclv1f_finalize();
            main_run_benchmark_cpu_dcsv1f_finalize();
            B_CPU_RUN_ALL.Enabled = true;
        }

        public void main_run_benchmark_gpu_sgemm_finalize()
        {
            if (benchmark_gpu_sgemm_dev == null) benchmark_gpu_sgemm_dev = new Label[ndev];
            for (int dev = 0; dev < ndev; dev++)
            {
                if (benchmark_gpu_sgemm_dev[dev] == null) benchmark_gpu_sgemm_dev[dev] = new Label();
                benchmark_gpu_sgemm_dev[dev].Parent = GBGPUBenchmarks;
                benchmark_gpu_sgemm_dev[dev].Location = new Point(B_GPU_SGEMM.Location.X + (1 + dev) * B_GPU_SGEMM.Width + 20, B_GPU_SGEMM.Location.Y + 2);
                benchmark_gpu_sgemm_dev[dev].Size = new Size(100, 20);
                benchmark_gpu_sgemm_dev[dev].Text = String.Format("{0:0.0}", benchmarks.gpu_sgemm_performance_dev[dev]) + " GF/sec";
                benchmark_gpu_sgemm_dev[dev].Show();
            }
            B_GPU_SGEMM.Enabled = true;

        }



        public void main_run_benchmark_gpu_sgemv2_finalize()
        {
            if (benchmark_gpu_sgemv2_dev == null) benchmark_gpu_sgemv2_dev = new Label[ndev];
            for (int dev = 0; dev < ndev; dev++)
            {
                if (benchmark_gpu_sgemv2_dev[dev] == null) benchmark_gpu_sgemv2_dev[dev] = new Label();
                benchmark_gpu_sgemv2_dev[dev].Parent = GBGPUBenchmarks;
                benchmark_gpu_sgemv2_dev[dev].Location = new Point(B_GPU_SGEMV2.Location.X + (1 + dev) * B_GPU_SGEMV2.Width + 20, B_GPU_SGEMV2.Location.Y + 2);
                benchmark_gpu_sgemv2_dev[dev].Size = new Size(100, 20);
                benchmark_gpu_sgemv2_dev[dev].Text = String.Format("{0:0.0}", benchmarks.gpu_sgemv2_performance_dev[dev]) + " GF/sec";
                benchmark_gpu_sgemv2_dev[dev].Show();
            }
            B_GPU_SGEMV2.Enabled = true;
        }


        public void main_run_benchmark_gpu_mt_finalize()
        {
            if (benchmark_gpu_mt_no_copy_dev == null) benchmark_gpu_mt_no_copy_dev = new Label[ndev];
            if (benchmark_gpu_mt_with_copy_dev == null) benchmark_gpu_mt_with_copy_dev = new Label[ndev];
            for (int dev = 0; dev < ndev; dev++)
            {
                if (benchmark_gpu_mt_no_copy_dev[dev] == null) benchmark_gpu_mt_no_copy_dev[dev] = new Label();
                benchmark_gpu_mt_no_copy_dev[dev].Parent = GBGPUBenchmarks;
                benchmark_gpu_mt_no_copy_dev[dev].Location = new Point(B_GPU_MT.Location.X + (1 + dev) * B_GPU_MT.Width + 20, B_GPU_MT.Location.Y + 2);
                benchmark_gpu_mt_no_copy_dev[dev].Size = new Size(B_GPU_MT.Width, 20);
                benchmark_gpu_mt_no_copy_dev[dev].Text = String.Format("{0:0.0}", benchmarks.gpu_mt_no_copy_performance_dev[dev]) + " milion eval/sec";
                benchmark_gpu_mt_no_copy_dev[dev].Show();

                if (benchmark_gpu_mt_with_copy_dev[dev] == null) benchmark_gpu_mt_with_copy_dev[dev] = new Label();
                benchmark_gpu_mt_with_copy_dev[dev].Parent = GBGPUBenchmarks;
                benchmark_gpu_mt_with_copy_dev[dev].Location = new Point(B_GPU_MT.Location.X + (1 + dev) * B_GPU_MT.Width + 20, L_GPU_MT_WITH_COPY.Location.Y + 2);
                benchmark_gpu_mt_with_copy_dev[dev].Size = new Size(B_GPU_MT.Width, 20);
                benchmark_gpu_mt_with_copy_dev[dev].Text = String.Format("{0:0.0}", benchmarks.gpu_mt_with_copy_performance_dev[dev]) + " milion eval/sec";
                benchmark_gpu_mt_with_copy_dev[dev].Show();

            }
            B_GPU_MT.Enabled = true;
        }

        public void main_run_benchmark_cpu_sgemm_finalize()
        {
            if (benchmark_cpu_sgemm == null) benchmark_cpu_sgemm = new Label();
            benchmark_cpu_sgemm.Parent = GBCPUBenchmarks;
            benchmark_cpu_sgemm.Location = new Point(B_CPU_SGEMM.Location.X + B_CPU_SGEMM.Width + 20, B_CPU_SGEMM.Location.Y + 2);
            benchmark_cpu_sgemm.Size = new Size(100, 20);
            benchmark_cpu_sgemm.Text = String.Format("{0:0.0}", benchmarks.cpu_sgemm_performance) + " GF/sec";
            benchmark_cpu_sgemm.Show();
            B_CPU_SGEMM.Enabled = true;

        }


        public void main_run_benchmark_cpu_mt_finalize()
        {
            if (benchmark_cpu_mt == null) benchmark_cpu_mt = new Label();
            benchmark_cpu_mt.Parent = GBCPUBenchmarks;
            benchmark_cpu_mt.Location = new Point(B_CPU_MT.Location.X + B_CPU_MT.Width + 20, B_CPU_MT.Location.Y + 2);
            benchmark_cpu_mt.Size = new Size(100, 20);
            benchmark_cpu_mt.Text = String.Format("{0:0.0}", benchmarks.cpu_mt_performance) + " milion eval/sec";
            benchmark_cpu_mt.Show();
            B_CPU_MT.Enabled = true;

        }

        public void main_run_benchmark_cpu_dgemm_finalize()
        {
            if (benchmark_cpu_dgemm == null) benchmark_cpu_dgemm = new Label();
            benchmark_cpu_dgemm.Parent = GBCPUBenchmarks;
            benchmark_cpu_dgemm.Location = new Point(B_CPU_DGEMM.Location.X + B_CPU_DGEMM.Width + 20, B_CPU_DGEMM.Location.Y + 2);
            benchmark_cpu_dgemm.Size = new Size(100, 20);
            benchmark_cpu_dgemm.Text = String.Format("{0:0.0}", benchmarks.cpu_dgemm_performance) + " GF/sec";
            benchmark_cpu_dgemm.Show();
            B_CPU_DGEMM.Enabled = true;

        }

        public void main_run_benchmark_register_peak_finalize()
        {
            if (benchmark_register_peak_dev == null) benchmark_register_peak_dev = new Label[ndev];
            for (int dev = 0; dev < ndev; dev++)
            {
                if (benchmark_register_peak_dev[dev] == null) benchmark_register_peak_dev[dev] = new Label();
                benchmark_register_peak_dev[dev].Parent = GBGPUBenchmarks;
                benchmark_register_peak_dev[dev].Location = new Point(B_GPU_RegisterPeak.Location.X + (1 + dev) * B_GPU_RegisterPeak.Width + 20, B_GPU_RegisterPeak.Location.Y + 2);
                benchmark_register_peak_dev[dev].Size = new Size(100, 20);
                benchmark_register_peak_dev[dev].Text = String.Format("{0:0.0}", benchmarks.register_peak_performance_dev[dev]) + " GF/sec";
                benchmark_register_peak_dev[dev].Show();
            }
            B_GPU_RegisterPeak.Enabled = true;

        }

        public void main_run_benchmark_shared_peak_finalize()
        {
            if (benchmark_shared_peak_dev == null) benchmark_shared_peak_dev = new Label[ndev];
            for (int dev = 0; dev < ndev; dev++)
            {
                if (benchmark_shared_peak_dev[dev] == null) benchmark_shared_peak_dev[dev] = new Label();
                benchmark_shared_peak_dev[dev].Parent = GBGPUBenchmarks;
                benchmark_shared_peak_dev[dev].Location = new Point(B_GPU_SharedPeak.Location.X + (1 + dev) * B_GPU_SharedPeak.Width + 20, B_GPU_SharedPeak.Location.Y + 2);
                benchmark_shared_peak_dev[dev].Size = new Size(100, 20);
                benchmark_shared_peak_dev[dev].Text = String.Format("{0:0.0}", benchmarks.shared_peak_performance_dev[dev]) + " GF/sec";
                benchmark_shared_peak_dev[dev].Show();
            }
            B_GPU_SharedPeak.Enabled = true;
        }

    }
}
