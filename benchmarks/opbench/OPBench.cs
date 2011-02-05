// OPTest.cs --- Part of the project OPLib 1.0, a high performance pricing library
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


namespace OPBench
{

    delegate void delegate_func_0();
    delegate void delegate_func_1(Object arg_);

    public partial class OPBench : Form
    {

        int ndev;

        System.Collections.ArrayList log;
        int log_n_items_outputted;

        Process process;
        Label cpu_number;
        Label cpu_name_system_tab;
        Label cpu_name_benchmarks_tab;
        Label cpu_total_memory;
        Label cpu_available_memory;
        Label gpu_number;
        Label[] gpu_name_system_tab_dev;
        Label[] gpu_name_benchmarks_tab_dev;
        Label process_start_time;
        Label process_virtual_memory;
        Label process_private_memory;
        Label process_physical_memory;
        Label process_cpu_time;
        Label process_exe_time;
        DateTime start_time;
        //Microsoft.VisualBasic.Devices.ComputerInfo myCompInfo;

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_device_get_count();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_device_total_memory(uint* bytes, int dev);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_device_get_name(byte* name, int len, int dev);


        public OPBench()
        {
            InitializeComponent();
            BSystemUpdate_exec();
            log = new System.Collections.ArrayList();
            log_n_items_outputted = 0;
            TABs.SelectedIndex = 2;
        }


        private void UpdateTimer_Tick(object sender, EventArgs e)
        {
            update_process_diagnostic();
            for (; log_n_items_outputted < log.Count; log_n_items_outputted++)
            {
                Display.AppendText(Environment.NewLine + log[log_n_items_outputted].ToString()); 
            }
            Application.DoEvents();
        }


        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            base.OnFormClosing(e);
        }

        private void OP_Load(object sender, EventArgs e)
        {
        }

        private void B_GPU_SGEMM4_Click(object sender, EventArgs e)
        {
            B_GPU_SGEMM4.Enabled = false;
            AsyncCallback callback;
            callback = new AsyncCallback(run_benchmark_gpu_sgemm4_finalize);
            delegate_func_0 delegate_run_gpu_sgemm4_benchmark = new delegate_func_0(run_benchmark_gpu_sgemm4);
            delegate_run_gpu_sgemm4_benchmark.BeginInvoke(callback, null);
        }

        private void B_CPU_SGEMM_Click(object sender, EventArgs e)
        {
            B_CPU_SGEMM.Enabled = false;
            AsyncCallback callback;
            callback = new AsyncCallback(run_benchmark_cpu_sgemm_finalize);
            delegate_func_0 delegate_run_cpu_sgemm_benchmark = new delegate_func_0(run_benchmark_cpu_sgemm);
            delegate_run_cpu_sgemm_benchmark.BeginInvoke(callback, null);
        }

        
        private void B_GPU_SGEMV4_Click(object sender, EventArgs e)
        {
            B_GPU_SGEMV4.Enabled = false;
            AsyncCallback callback;
            callback = new AsyncCallback(run_benchmark_gpu_sgemv4_finalize);
            delegate_func_0 delegate_run_gpu_sgemv4_benchmark = new delegate_func_0(run_benchmark_gpu_sgemv4);
            delegate_run_gpu_sgemv4_benchmark.BeginInvoke(callback, null);
        }

        private void B_CPU_DCLV1F_Click(object sender, EventArgs e)
        {
            B_CPU_DCLV1F.Enabled = false;
            AsyncCallback callback;
            callback = new AsyncCallback(run_benchmark_cpu_dclv1f_finalize);
            delegate_func_0 delegate_run_cpu_dclv1f_benchmark = new delegate_func_0(run_benchmark_cpu_dclv1f);
            delegate_run_cpu_dclv1f_benchmark.BeginInvoke(callback, null);
        }


        private void B_GPU_SGLV_Click(object sender, EventArgs e)
        {
            B_GPU_SGLV.Enabled = false;
            AsyncCallback callback;
            callback = new AsyncCallback(run_benchmark_gpu_sglv1f_finalize);
            delegate_func_0 delegate_run_gpu_sglv1f_benchmark = new delegate_func_0(run_benchmark_gpu_sglv1f);
            delegate_run_gpu_sglv1f_benchmark.BeginInvoke(callback, null);
        }

        private void B_GPU_RUN_ALL_Click(object sender, EventArgs e)
        {
            B_GPU_RUN_ALL.Enabled = false;
            B_GPU_MT.Enabled = false;
            B_GPU_SGSV.Enabled = false;
            B_GPU_SGLV.Enabled = false;
            B_GPU_SGEMV4.Enabled = false;
            B_GPU_SGEMV2.Enabled = false;
            B_GPU_SGEMM.Enabled = false;
            B_GPU_SGEMM.Enabled = false;
            B_GPU_SGEMM4.Enabled = false;
            B_GPU_RegisterPeak.Enabled = false;
            B_GPU_SharedPeak.Enabled = false;
            AsyncCallback callback;
            callback = new AsyncCallback(run_benchmark_gpu_all_finalize);
            delegate_func_0 delegate_run_gpu_all_benchmark = new delegate_func_0(run_benchmark_gpu_all);
            delegate_run_gpu_all_benchmark.BeginInvoke(callback, null);
        }

        private void B_CPU_RUN_ALL_Click(object sender, EventArgs e)
        {
            B_CPU_DCLV1F.Enabled = false;
            B_CPU_DCSV1F.Enabled = false;
            B_CPU_RUN_ALL.Enabled = false;
            B_CPU_SGEMM.Enabled = false;
            B_CPU_MT.Enabled = false;
            B_CPU_DGEMM.Enabled = false;
            AsyncCallback callback;
            callback = new AsyncCallback(run_benchmark_cpu_all_finalize);
            delegate_func_0 delegate_run_cpu_all_benchmark = new delegate_func_0(run_benchmark_cpu_all);
            delegate_run_cpu_all_benchmark.BeginInvoke(callback, null);
        }


        private void B_GPU_SGSV_Click(object sender, EventArgs e)
        {
            B_GPU_SGSV.Enabled = false;
            AsyncCallback callback;
            callback = new AsyncCallback(run_benchmark_gpu_sgsv1f_finalize);
            delegate_func_0 delegate_run_gpu_sgsv1f_benchmark = new delegate_func_0(run_benchmark_gpu_sgsv1f);
            delegate_run_gpu_sgsv1f_benchmark.BeginInvoke(callback, null);
        }

        private void B_GPU_RegisterPeak_Click(object sender, EventArgs e)
        {
            B_GPU_RegisterPeak.Enabled = false;
            AsyncCallback callback;
            callback = new AsyncCallback(run_benchmark_register_peak_finalize);
            delegate_func_0 delegate_run_register_peak_benchmark = new delegate_func_0(run_benchmark_register_peak);
            delegate_run_register_peak_benchmark.BeginInvoke(callback, null);
        }

        private void B_GPU_SharedPeak_Click(object sender, EventArgs e)
        {
            B_GPU_SharedPeak.Enabled = false;
            AsyncCallback callback;
            callback = new AsyncCallback(run_benchmark_shared_peak_finalize);
            delegate_func_0 delegate_run_shared_peak_benchmark = new delegate_func_0(run_benchmark_shared_peak);
            delegate_run_shared_peak_benchmark.BeginInvoke(callback, null);

        }

        private void B_GPU_SGEMM_Click(object sender, EventArgs e)
        {
            B_GPU_SGEMM.Enabled = false;
            AsyncCallback callback;
            callback = new AsyncCallback(run_benchmark_gpu_sgemm_finalize);
            delegate_func_0 delegate_run_gpu_sgemm_benchmark = new delegate_func_0(run_benchmark_gpu_sgemm);
            delegate_run_gpu_sgemm_benchmark.BeginInvoke(callback, null);
        }

        private void B_CPU_DCSV1F_Click(object sender, EventArgs e)
        {
            B_CPU_DCSV1F.Enabled = false;
            AsyncCallback callback;
            callback = new AsyncCallback(run_benchmark_cpu_dcsv1f_finalize);
            delegate_func_0 delegate_run_cpu_dcsv1f_benchmark = new delegate_func_0(run_benchmark_cpu_dcsv1f);
            delegate_run_cpu_dcsv1f_benchmark.BeginInvoke(callback, null);
        }

        private void B_CPU_DGEMM_Click(object sender, EventArgs e)
        {
            B_CPU_DGEMM.Enabled = false;
            AsyncCallback callback;
            callback = new AsyncCallback(run_benchmark_cpu_dgemm_finalize);
            delegate_func_0 delegate_run_cpu_dgemm_benchmark = new delegate_func_0(run_benchmark_cpu_dgemm);
            delegate_run_cpu_dgemm_benchmark.BeginInvoke(callback, null);
        }

        private void B_CPU_MT_Click(object sender, EventArgs e)
        {
            B_CPU_MT.Enabled = false;
            AsyncCallback callback;
            callback = new AsyncCallback(run_benchmark_cpu_mt_finalize);
            delegate_func_0 delegate_run_cpu_mt_benchmark = new delegate_func_0(run_benchmark_cpu_mt);
            delegate_run_cpu_mt_benchmark.BeginInvoke(callback, null);
        }

        private void B_GPU_MT_Click(object sender, EventArgs e)
        {
            B_GPU_MT.Enabled = false;
            AsyncCallback callback;
            callback = new AsyncCallback(run_benchmark_gpu_mt_finalize);
            delegate_func_0 delegate_run_gpu_mt_benchmark = new delegate_func_0(run_benchmark_gpu_mt);
            delegate_run_gpu_mt_benchmark.BeginInvoke(callback, null);
        }

        private void B_GPU_SGEMV2_Click(object sender, EventArgs e)
        {
            B_GPU_SGEMV2.Enabled = false;
            AsyncCallback callback;
            callback = new AsyncCallback(run_benchmark_gpu_sgemv2_finalize);
            delegate_func_0 delegate_run_gpu_sgemv2_benchmark = new delegate_func_0(run_benchmark_gpu_sgemv2);
            delegate_run_gpu_sgemv2_benchmark.BeginInvoke(callback, null);
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