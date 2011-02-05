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


namespace CSV
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("running benchmark for stochastic volatility model on the cpu");

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
            Console.WriteLine("blas performance: " + String.Format("{0:0.0}", gigaflops_per_second) + " GF/sec");

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
            Console.WriteLine("mc performance: " + String.Format("{0:0.0}", milion_evals_per_second) + " milion eval/sec");

            Console.Read();
        }
    }

    public class emptyEvaluator : OPModel.CMCEvaluator
    {
        public emptyEvaluator()
        {
        }

        public override unsafe void mc_eval_implementation(short* y_sk, int th, int batch)
        {
        }

    };


}
