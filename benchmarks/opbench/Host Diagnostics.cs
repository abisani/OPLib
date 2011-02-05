// Host Diagnostics.cs --- Part of the project OPLib 1.0, a high performance pricing library
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
        public class struct_host_info
        {
            public int number_of_cores;
            public string cpu_name;
            public double cpu_total_memory;
            public double cpu_available_memory;
        }

        struct_host_info host_info;
        private void BSystemUpdate_Click(object sender, EventArgs e)
        {
            cpu_number.Text = "";
            cpu_name_system_tab.Text = "";
            cpu_name_benchmarks_tab.Text = "";
            cpu_total_memory.Text = "";
            cpu_available_memory.Text = "";
            BSystemUpdate_exec();
        }


        private void BSystemUpdate_exec()
        {
            AsyncCallback callback;
            callback = new AsyncCallback(get_host_info_async_finalize);
            delegate_func_0 delegate_get_host_info_async = new delegate_func_0(get_host_info_async);
            host_info = new struct_host_info();
            delegate_get_host_info_async.BeginInvoke(callback, null);
            get_host_info_sync();
        }

        public void get_host_info_async()
        {
            try
            {
                System.Management.ManagementObjectSearcher searcher = new System.Management.ManagementObjectSearcher("select * from " + "Win32_Processor");
                string[] CPU_name_split = null;
                System.Management.ManagementObjectCollection sysinfo = searcher.Get();
                foreach (System.Management.ManagementObject share in sysinfo)
                {
                    try
                    {
                        CPU_name_split = share["Name"].ToString().Split(' ');
                        break;
                    }
                    catch
                    {
                    }
                }

                lock (host_info)
                {
                    host_info.cpu_name = "";
                    for (int i = 0; i < CPU_name_split.Length; i++)
                    {
                        if (CPU_name_split[i] != "")
                        {
                            host_info.cpu_name += CPU_name_split[i] + " ";
                        }
                    }
                }
            }
            catch
            {
            }
        }


        public void get_host_info_async_finalize(IAsyncResult asyncResult)
        {
            System.Runtime.Remoting.Messaging.AsyncResult
                result = (System.Runtime.Remoting.Messaging.AsyncResult)asyncResult;
            delegate_func_0 doWork = (delegate_func_0)result.AsyncDelegate;
            doWork.EndInvoke(asyncResult);

            ISynchronizeInvoke synchronizer = BSystemUpdate;
            if (synchronizer.InvokeRequired == false)
            {
                main_get_host_info_async_finalize();
                return;
            }
            delegate_func_0 delegate_get_host_info_async_finalize = new delegate_func_0(main_get_host_info_async_finalize);
            try
            {
                synchronizer.Invoke(delegate_get_host_info_async_finalize, new object[] { });
            }
            catch
            { }
        }


        public void main_get_host_info_async_finalize()
        {

            if (cpu_name_system_tab == null) cpu_name_system_tab = new Label();
            cpu_name_system_tab.Parent = GBHost;
            cpu_name_system_tab.Location = new Point(5, 20);
            cpu_name_system_tab.Size = new Size(GBHost.Width - 10, 20);
            cpu_name_system_tab.Text = "CPU: " + host_info.cpu_name;
            cpu_name_system_tab.Show();

            if (cpu_name_benchmarks_tab == null) cpu_name_benchmarks_tab = new Label();
            cpu_name_benchmarks_tab.Parent = this.GBCPUBenchmarks;
            cpu_name_benchmarks_tab.Location = new Point(200, 10);
            cpu_name_benchmarks_tab.Size = new Size(300, 20);
            cpu_name_benchmarks_tab.Text = "CPU: " + host_info.cpu_name;
            cpu_name_benchmarks_tab.Show();

        }


        public void get_host_info_sync()
        {
            try
            {
                int yoffset = 20;
                if (cpu_number == null) cpu_number = new Label();
                cpu_number.Parent = GBHost;
                cpu_number.Location = new Point(5, 20 + yoffset);
                cpu_number.Size = new Size(GBHost.Width - 10, 20);
                cpu_number.Text = "Number of CPU cores: " + System.Environment.ProcessorCount;

                yoffset += 20;
                if (cpu_total_memory == null) cpu_total_memory = new Label();
                cpu_total_memory.Parent = GBHost;
                cpu_total_memory.Location = new Point(5, 20 + yoffset);
                cpu_total_memory.Size = new Size(GBHost.Width - 10, 20);

                //myCompInfo = new Microsoft.VisualBasic.Devices.ComputerInfo();
                //cpu_total_memory.Text = "Total physical memory: " + String.Format("{0:0.000}", myCompInfo.TotalPhysicalMemory / Math.Pow(2f, 30)) + " GB";

                yoffset += 20;
                if (cpu_available_memory == null) cpu_available_memory = new Label();
                cpu_available_memory.Parent = GBHost;
                cpu_available_memory.Location = new Point(5, 20 + yoffset);
                cpu_available_memory.Size = new Size(GBHost.Width - 10, 20);
                //cpu_available_memory.Text = "Available physical memory: " + String.Format("{0:0.000}", myCompInfo.AvailablePhysicalMemory / Math.Pow(2f, 30)) + " GB";

                yoffset = 0;
                gpu_number = new Label();
                gpu_number.Parent = GBDevice;
                gpu_number.Location = new Point(5, 20 + yoffset);

                yoffset += 20;
                gpu_number.Size = new Size(GBDevice.Width - 10, 20);
                ndev = opcuda_device_get_count();
                gpu_number.Text = "Number of GPUs: " + ndev;

                gpu_name_system_tab_dev = new Label[ndev];
                gpu_name_benchmarks_tab_dev = new Label[ndev];

                for (int dev = 0; dev < ndev; dev++)
                {
                    gpu_name_system_tab_dev[dev] = new Label();
                    gpu_name_benchmarks_tab_dev[dev] = new Label();
                    gpu_name_system_tab_dev[dev].Parent = GBDevice;
                    gpu_name_benchmarks_tab_dev[dev].Parent = this.GBGPUBenchmarks;
                    gpu_name_system_tab_dev[dev].Location = new Point(5 + yoffset * 5, 20);
                    gpu_name_benchmarks_tab_dev[dev].Location = new Point(this.B_GPU_RUN_ALL.Location.X + (1 + dev) * B_GPU_RegisterPeak.Width + 20, B_GPU_RegisterPeak.Location.Y -20 ); 
                    yoffset += 20;
                    gpu_name_system_tab_dev[dev].Size = new Size(GBDevice.Width - 2, 20);
                    gpu_name_benchmarks_tab_dev[dev].Size = new Size(150, 20);

                    unsafe
                    {
                        int len = 1024;
                        byte[] name = new byte[len];
                        uint bytes;
                        fixed (byte* namep = &name[0])
                        {
                            opcuda_device_get_name(namep, len, dev);
                            opcuda_device_total_memory(&bytes, dev);
                        }
                        int count;
                        for (count = 0; count < len; count++)
                        {
                            if (name[count] == 0) break;
                        }
                        string str = ASCIIEncoding.ASCII.GetString(name, 0, count);

                        gpu_name_system_tab_dev[dev].Text = "GPU " + dev + ": " + str + ", total memory :" + bytes / Math.Pow(2f, 30) + " GB";
                        gpu_name_benchmarks_tab_dev[dev].Text = "GPU " + dev + ": " + str ;
                    }
                }


                process = Process.GetCurrentProcess();

                yoffset = 0;
                process_start_time = new Label();
                process_start_time.Parent = GBProcess;
                process_start_time.Location = new Point(5, 20 + yoffset);
                process_start_time.Size = new Size(GBProcess.Width / 2 - 10, 20);
                start_time = DateTime.Now;
                process_start_time.Text = "Process start time: " + start_time.ToString();

                yoffset += 20;
                process_cpu_time = new Label();
                process_cpu_time.Parent = GBProcess;
                process_cpu_time.Size = new Size(GBProcess.Width / 2 - 10, 20);
                process_cpu_time.Location = new Point(5, 20 + yoffset);

                yoffset += 20;
                process_exe_time = new Label();
                process_exe_time.Parent = GBProcess;
                process_exe_time.Size = new Size(GBProcess.Width / 2 - 10, 20);
                process_exe_time.Location = new Point(5, 20 + yoffset);

                yoffset = 0;
                process_virtual_memory = new Label();
                process_virtual_memory.Parent = GBProcess;
                process_virtual_memory.Size = new Size(GBProcess.Width / 2 - 10, 20);
                process_virtual_memory.Location = new Point(GBProcess.Width / 2 + 5, 20 + yoffset);

                yoffset += 20;
                process_private_memory = new Label();
                process_private_memory.Parent = GBProcess;
                process_private_memory.Size = new Size(GBProcess.Width / 2 - 10, 20);
                process_private_memory.Location = new Point(GBProcess.Width / 2 + 5, 20 + yoffset);

                yoffset += 20;
                process_physical_memory = new Label();
                process_physical_memory.Parent = GBProcess;
                process_physical_memory.Size = new Size(GBProcess.Width / 2 - 10, 20);
                process_physical_memory.Location = new Point(GBProcess.Width / 2 + 5, 20 + yoffset);

                update_process_diagnostic();

            }
            catch
            {
            }
        }


        void update_process_diagnostic()
        {
            try
            {
                process.Refresh();
                //myCompInfo = new Microsoft.VisualBasic.Devices.ComputerInfo();
                //cpu_available_memory.Text = "Available physical memory: " + String.Format("{0:0.000}", myCompInfo.AvailablePhysicalMemory / Math.Pow(2f, 30)) + " GB";
                process_virtual_memory.Text = "Virtual memory: " + String.Format("{0:0.0}", process.VirtualMemorySize64 / Math.Pow(2f, 20)) + " MB";
                process_private_memory.Text = "Private memory: " + String.Format("{0:0.0}", process.PrivateMemorySize64 / Math.Pow(2f, 20)) + " MB";
                process_physical_memory.Text = "Physical memory: " + String.Format("{0:0.0}", process.WorkingSet64 / Math.Pow(2f, 20)) + " MB";
                process_cpu_time.Text = "CPU time: " + process.TotalProcessorTime.ToString();
                process_exe_time.Text = "Execution time: " + String.Format("{0:0:0.00}", (DateTime.Now - start_time));
                //cpu_available_memory.Text = "Available physical memory: " + String.Format("{0:0.000}", myCompInfo.AvailablePhysicalMemory / Math.Pow(2f, 30)) + " GB";
            }
            catch
            {
            }
        }


    }
}
