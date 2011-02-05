// OPTest.designer.cs --- Part of the project OPLib 1.0, a high performance pricing library
// based on operator methods, higher level BLAS and multicore architectures 

// Author:     2009 Claudio Albanese
// Maintainer: Claudio Albanese <claudio@albanese.co.uk>
// Created:    April-July 2009
// Version:    1.0.0
// Credits:    The CUDA code for SGEMM4, SGEMV4 and SSQMM were inspired by 
//             Vasily Volkov's implementation of SGEMM
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


namespace OPBench
{
    partial class OPBench
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.TABs = new System.Windows.Forms.TabControl();
            this.back = new System.Windows.Forms.TabPage();
            this.GBProcess = new System.Windows.Forms.GroupBox();
            this.GBDevice = new System.Windows.Forms.GroupBox();
            this.BSystemUpdate = new System.Windows.Forms.Button();
            this.GBHost = new System.Windows.Forms.GroupBox();
            this.TBConsole = new System.Windows.Forms.TabPage();
            this.groupBox5 = new System.Windows.Forms.GroupBox();
            this.Display = new System.Windows.Forms.TextBox();
            this.TBBenchmarks = new System.Windows.Forms.TabPage();
            this.GBCPUBenchmarks = new System.Windows.Forms.GroupBox();
            this.B_CPU_MT = new System.Windows.Forms.Button();
            this.B_CPU_DGEMM = new System.Windows.Forms.Button();
            this.B_CPU_RUN_ALL = new System.Windows.Forms.Button();
            this.label6 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.B_CPU_DCSV1F = new System.Windows.Forms.Button();
            this.B_CPU_DCLV1F = new System.Windows.Forms.Button();
            this.B_CPU_SGEMM = new System.Windows.Forms.Button();
            this.GBGPUBenchmarks = new System.Windows.Forms.GroupBox();
            this.L_GPU_MT_WITH_COPY = new System.Windows.Forms.Label();
            this.B_GPU_SGEMV2 = new System.Windows.Forms.Button();
            this.B_GPU_MT = new System.Windows.Forms.Button();
            this.label7 = new System.Windows.Forms.Label();
            this.B_GPU_SGSV = new System.Windows.Forms.Button();
            this.B_GPU_RUN_ALL = new System.Windows.Forms.Button();
            this.label4 = new System.Windows.Forms.Label();
            this.B_GPU_SGLV = new System.Windows.Forms.Button();
            this.B_GPU_SGEMV4 = new System.Windows.Forms.Button();
            this.B_GPU_SGEMM = new System.Windows.Forms.Button();
            this.B_GPU_SharedPeak = new System.Windows.Forms.Button();
            this.B_GPU_RegisterPeak = new System.Windows.Forms.Button();
            this.B_GPU_SGEMM4 = new System.Windows.Forms.Button();
            this.UpdateTimer = new System.Windows.Forms.Timer(this.components);
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.TABs.SuspendLayout();
            this.back.SuspendLayout();
            this.TBConsole.SuspendLayout();
            this.groupBox5.SuspendLayout();
            this.TBBenchmarks.SuspendLayout();
            this.GBCPUBenchmarks.SuspendLayout();
            this.GBGPUBenchmarks.SuspendLayout();
            this.SuspendLayout();
            // 
            // TABs
            // 
            this.TABs.Controls.Add(this.back);
            this.TABs.Controls.Add(this.TBConsole);
            this.TABs.Controls.Add(this.TBBenchmarks);
            this.TABs.Location = new System.Drawing.Point(11, 66);
            this.TABs.Margin = new System.Windows.Forms.Padding(2);
            this.TABs.Name = "TABs";
            this.TABs.SelectedIndex = 0;
            this.TABs.Size = new System.Drawing.Size(840, 650);
            this.TABs.TabIndex = 0;
            // 
            // back
            // 
            this.back.Controls.Add(this.GBProcess);
            this.back.Controls.Add(this.GBDevice);
            this.back.Controls.Add(this.BSystemUpdate);
            this.back.Controls.Add(this.GBHost);
            this.back.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.back.Location = new System.Drawing.Point(4, 22);
            this.back.Margin = new System.Windows.Forms.Padding(2);
            this.back.Name = "back";
            this.back.Padding = new System.Windows.Forms.Padding(2);
            this.back.Size = new System.Drawing.Size(832, 624);
            this.back.TabIndex = 0;
            this.back.Text = "System";
            this.back.UseVisualStyleBackColor = true;
            // 
            // GBProcess
            // 
            this.GBProcess.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.GBProcess.Location = new System.Drawing.Point(5, 146);
            this.GBProcess.Name = "GBProcess";
            this.GBProcess.Size = new System.Drawing.Size(809, 143);
            this.GBProcess.TabIndex = 5;
            this.GBProcess.TabStop = false;
            this.GBProcess.Text = "PROCESSES";
            // 
            // GBDevice
            // 
            this.GBDevice.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.GBDevice.Location = new System.Drawing.Point(281, 6);
            this.GBDevice.Name = "GBDevice";
            this.GBDevice.Size = new System.Drawing.Size(295, 134);
            this.GBDevice.TabIndex = 4;
            this.GBDevice.TabStop = false;
            this.GBDevice.Text = "DEVICES";
            // 
            // BSystemUpdate
            // 
            this.BSystemUpdate.Location = new System.Drawing.Point(730, 12);
            this.BSystemUpdate.Name = "BSystemUpdate";
            this.BSystemUpdate.Size = new System.Drawing.Size(75, 23);
            this.BSystemUpdate.TabIndex = 3;
            this.BSystemUpdate.Text = "Update";
            this.BSystemUpdate.UseVisualStyleBackColor = true;
            this.BSystemUpdate.Click += new System.EventHandler(this.BSystemUpdate_Click);
            // 
            // GBHost
            // 
            this.GBHost.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.8F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.GBHost.Location = new System.Drawing.Point(5, 6);
            this.GBHost.Name = "GBHost";
            this.GBHost.Size = new System.Drawing.Size(270, 134);
            this.GBHost.TabIndex = 2;
            this.GBHost.TabStop = false;
            this.GBHost.Text = "HOST";
            // 
            // TBConsole
            // 
            this.TBConsole.Controls.Add(this.groupBox5);
            this.TBConsole.Location = new System.Drawing.Point(4, 22);
            this.TBConsole.Name = "TBConsole";
            this.TBConsole.Padding = new System.Windows.Forms.Padding(3);
            this.TBConsole.Size = new System.Drawing.Size(832, 624);
            this.TBConsole.TabIndex = 6;
            this.TBConsole.Text = "Console";
            this.TBConsole.UseVisualStyleBackColor = true;
            // 
            // groupBox5
            // 
            this.groupBox5.Controls.Add(this.Display);
            this.groupBox5.Location = new System.Drawing.Point(17, 16);
            this.groupBox5.Name = "groupBox5";
            this.groupBox5.Size = new System.Drawing.Size(695, 602);
            this.groupBox5.TabIndex = 0;
            this.groupBox5.TabStop = false;
            this.groupBox5.Text = "CONSOLE LOG";
            // 
            // Display
            // 
            this.Display.BackColor = System.Drawing.Color.Black;
            this.Display.Font = new System.Drawing.Font("Courier New", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Display.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(128)))), ((int)(((byte)(255)))), ((int)(((byte)(128)))));
            this.Display.Location = new System.Drawing.Point(6, 53);
            this.Display.Multiline = true;
            this.Display.Name = "Display";
            this.Display.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.Display.Size = new System.Drawing.Size(669, 529);
            this.Display.TabIndex = 95;
            // 
            // TBBenchmarks
            // 
            this.TBBenchmarks.Controls.Add(this.GBCPUBenchmarks);
            this.TBBenchmarks.Controls.Add(this.GBGPUBenchmarks);
            this.TBBenchmarks.Location = new System.Drawing.Point(4, 22);
            this.TBBenchmarks.Name = "TBBenchmarks";
            this.TBBenchmarks.Padding = new System.Windows.Forms.Padding(3);
            this.TBBenchmarks.Size = new System.Drawing.Size(832, 624);
            this.TBBenchmarks.TabIndex = 7;
            this.TBBenchmarks.Text = "Benchmarks";
            this.TBBenchmarks.UseVisualStyleBackColor = true;
            // 
            // GBCPUBenchmarks
            // 
            this.GBCPUBenchmarks.Controls.Add(this.B_CPU_MT);
            this.GBCPUBenchmarks.Controls.Add(this.B_CPU_DGEMM);
            this.GBCPUBenchmarks.Controls.Add(this.B_CPU_RUN_ALL);
            this.GBCPUBenchmarks.Controls.Add(this.label6);
            this.GBCPUBenchmarks.Controls.Add(this.label5);
            this.GBCPUBenchmarks.Controls.Add(this.B_CPU_DCSV1F);
            this.GBCPUBenchmarks.Controls.Add(this.B_CPU_DCLV1F);
            this.GBCPUBenchmarks.Controls.Add(this.B_CPU_SGEMM);
            this.GBCPUBenchmarks.Location = new System.Drawing.Point(6, 395);
            this.GBCPUBenchmarks.Name = "GBCPUBenchmarks";
            this.GBCPUBenchmarks.Size = new System.Drawing.Size(820, 229);
            this.GBCPUBenchmarks.TabIndex = 5;
            this.GBCPUBenchmarks.TabStop = false;
            this.GBCPUBenchmarks.Text = "HOST BENCHMARKS";
            // 
            // B_CPU_MT
            // 
            this.B_CPU_MT.Location = new System.Drawing.Point(6, 98);
            this.B_CPU_MT.Name = "B_CPU_MT";
            this.B_CPU_MT.Size = new System.Drawing.Size(181, 23);
            this.B_CPU_MT.TabIndex = 13;
            this.B_CPU_MT.Text = "MERSENNE TWISTER";
            this.B_CPU_MT.UseVisualStyleBackColor = true;
            this.B_CPU_MT.Click += new System.EventHandler(this.B_CPU_MT_Click);
            // 
            // B_CPU_DGEMM
            // 
            this.B_CPU_DGEMM.Location = new System.Drawing.Point(6, 75);
            this.B_CPU_DGEMM.Name = "B_CPU_DGEMM";
            this.B_CPU_DGEMM.Size = new System.Drawing.Size(181, 23);
            this.B_CPU_DGEMM.TabIndex = 12;
            this.B_CPU_DGEMM.Text = "DGEMM";
            this.B_CPU_DGEMM.UseVisualStyleBackColor = true;
            this.B_CPU_DGEMM.Click += new System.EventHandler(this.B_CPU_DGEMM_Click);
            // 
            // B_CPU_RUN_ALL
            // 
            this.B_CPU_RUN_ALL.Location = new System.Drawing.Point(7, 27);
            this.B_CPU_RUN_ALL.Name = "B_CPU_RUN_ALL";
            this.B_CPU_RUN_ALL.Size = new System.Drawing.Size(181, 23);
            this.B_CPU_RUN_ALL.TabIndex = 11;
            this.B_CPU_RUN_ALL.Text = "RUN ALL CPU BENCHMARKS";
            this.B_CPU_RUN_ALL.UseVisualStyleBackColor = true;
            this.B_CPU_RUN_ALL.Click += new System.EventHandler(this.B_CPU_RUN_ALL_Click);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(67, 190);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(59, 13);
            this.label6.TabIndex = 10;
            this.label6.Text = "SV Kernels";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(67, 146);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(58, 13);
            this.label5.TabIndex = 9;
            this.label5.Text = "LV Kernels";
            // 
            // B_CPU_DCSV1F
            // 
            this.B_CPU_DCSV1F.Location = new System.Drawing.Point(7, 164);
            this.B_CPU_DCSV1F.Name = "B_CPU_DCSV1F";
            this.B_CPU_DCSV1F.Size = new System.Drawing.Size(180, 23);
            this.B_CPU_DCSV1F.TabIndex = 8;
            this.B_CPU_DCSV1F.Text = "SV Model (Kernels)";
            this.B_CPU_DCSV1F.UseVisualStyleBackColor = true;
            this.B_CPU_DCSV1F.Click += new System.EventHandler(this.B_CPU_DCSV1F_Click);
            // 
            // B_CPU_DCLV1F
            // 
            this.B_CPU_DCLV1F.Location = new System.Drawing.Point(6, 121);
            this.B_CPU_DCLV1F.Name = "B_CPU_DCLV1F";
            this.B_CPU_DCLV1F.Size = new System.Drawing.Size(181, 23);
            this.B_CPU_DCLV1F.TabIndex = 7;
            this.B_CPU_DCLV1F.Text = "LV Model (Monte Carlo)";
            this.B_CPU_DCLV1F.UseVisualStyleBackColor = true;
            this.B_CPU_DCLV1F.Click += new System.EventHandler(this.B_CPU_DCLV1F_Click);
            // 
            // B_CPU_SGEMM
            // 
            this.B_CPU_SGEMM.Location = new System.Drawing.Point(6, 51);
            this.B_CPU_SGEMM.Name = "B_CPU_SGEMM";
            this.B_CPU_SGEMM.Size = new System.Drawing.Size(181, 23);
            this.B_CPU_SGEMM.TabIndex = 6;
            this.B_CPU_SGEMM.Text = "SGEMM";
            this.B_CPU_SGEMM.UseVisualStyleBackColor = true;
            this.B_CPU_SGEMM.Click += new System.EventHandler(this.B_CPU_SGEMM_Click);
            // 
            // GBGPUBenchmarks
            // 
            this.GBGPUBenchmarks.Controls.Add(this.L_GPU_MT_WITH_COPY);
            this.GBGPUBenchmarks.Controls.Add(this.B_GPU_SGEMV2);
            this.GBGPUBenchmarks.Controls.Add(this.B_GPU_MT);
            this.GBGPUBenchmarks.Controls.Add(this.label7);
            this.GBGPUBenchmarks.Controls.Add(this.B_GPU_SGSV);
            this.GBGPUBenchmarks.Controls.Add(this.B_GPU_RUN_ALL);
            this.GBGPUBenchmarks.Controls.Add(this.label4);
            this.GBGPUBenchmarks.Controls.Add(this.B_GPU_SGLV);
            this.GBGPUBenchmarks.Controls.Add(this.B_GPU_SGEMV4);
            this.GBGPUBenchmarks.Controls.Add(this.B_GPU_SGEMM);
            this.GBGPUBenchmarks.Controls.Add(this.B_GPU_SharedPeak);
            this.GBGPUBenchmarks.Controls.Add(this.B_GPU_RegisterPeak);
            this.GBGPUBenchmarks.Controls.Add(this.B_GPU_SGEMM4);
            this.GBGPUBenchmarks.Location = new System.Drawing.Point(7, 37);
            this.GBGPUBenchmarks.Name = "GBGPUBenchmarks";
            this.GBGPUBenchmarks.Size = new System.Drawing.Size(820, 356);
            this.GBGPUBenchmarks.TabIndex = 4;
            this.GBGPUBenchmarks.TabStop = false;
            this.GBGPUBenchmarks.Text = "DEVICE BENCHMARKS";
            // 
            // L_GPU_MT_WITH_COPY
            // 
            this.L_GPU_MT_WITH_COPY.AutoSize = true;
            this.L_GPU_MT_WITH_COPY.Location = new System.Drawing.Point(43, 235);
            this.L_GPU_MT_WITH_COPY.Name = "L_GPU_MT_WITH_COPY";
            this.L_GPU_MT_WITH_COPY.Size = new System.Drawing.Size(119, 13);
            this.L_GPU_MT_WITH_COPY.TabIndex = 14;
            this.L_GPU_MT_WITH_COPY.Text = "WITH COPY TO HOST";
            // 
            // B_GPU_SGEMV2
            // 
            this.B_GPU_SGEMV2.Location = new System.Drawing.Point(16, 186);
            this.B_GPU_SGEMV2.Name = "B_GPU_SGEMV2";
            this.B_GPU_SGEMV2.Size = new System.Drawing.Size(172, 23);
            this.B_GPU_SGEMV2.TabIndex = 13;
            this.B_GPU_SGEMV2.Text = "SGEMV2";
            this.B_GPU_SGEMV2.UseVisualStyleBackColor = true;
            this.B_GPU_SGEMV2.Click += new System.EventHandler(this.B_GPU_SGEMV2_Click);
            // 
            // B_GPU_MT
            // 
            this.B_GPU_MT.Location = new System.Drawing.Point(15, 209);
            this.B_GPU_MT.Name = "B_GPU_MT";
            this.B_GPU_MT.Size = new System.Drawing.Size(172, 23);
            this.B_GPU_MT.TabIndex = 12;
            this.B_GPU_MT.Text = "MERSENNE TWISTER";
            this.B_GPU_MT.UseVisualStyleBackColor = true;
            this.B_GPU_MT.Click += new System.EventHandler(this.B_GPU_MT_Click);
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(75, 320);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(59, 13);
            this.label7.TabIndex = 11;
            this.label7.Text = "SV Kernels";
            // 
            // B_GPU_SGSV
            // 
            this.B_GPU_SGSV.Location = new System.Drawing.Point(17, 294);
            this.B_GPU_SGSV.Name = "B_GPU_SGSV";
            this.B_GPU_SGSV.Size = new System.Drawing.Size(172, 23);
            this.B_GPU_SGSV.TabIndex = 10;
            this.B_GPU_SGSV.Text = "SV Model (Monte Carlo)";
            this.B_GPU_SGSV.UseVisualStyleBackColor = true;
            this.B_GPU_SGSV.Click += new System.EventHandler(this.B_GPU_SGSV_Click);
            // 
            // B_GPU_RUN_ALL
            // 
            this.B_GPU_RUN_ALL.Location = new System.Drawing.Point(15, 46);
            this.B_GPU_RUN_ALL.Name = "B_GPU_RUN_ALL";
            this.B_GPU_RUN_ALL.Size = new System.Drawing.Size(172, 23);
            this.B_GPU_RUN_ALL.TabIndex = 9;
            this.B_GPU_RUN_ALL.Text = "RUN ALL GPU BENCHMARKS";
            this.B_GPU_RUN_ALL.UseVisualStyleBackColor = true;
            this.B_GPU_RUN_ALL.Click += new System.EventHandler(this.B_GPU_RUN_ALL_Click);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(73, 281);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(58, 13);
            this.label4.TabIndex = 8;
            this.label4.Text = "LV Kernels";
            // 
            // B_GPU_SGLV
            // 
            this.B_GPU_SGLV.Location = new System.Drawing.Point(15, 255);
            this.B_GPU_SGLV.Name = "B_GPU_SGLV";
            this.B_GPU_SGLV.Size = new System.Drawing.Size(172, 23);
            this.B_GPU_SGLV.TabIndex = 7;
            this.B_GPU_SGLV.Text = "LV Model (Monte Carlo)";
            this.B_GPU_SGLV.UseVisualStyleBackColor = true;
            this.B_GPU_SGLV.Click += new System.EventHandler(this.B_GPU_SGLV_Click);
            // 
            // B_GPU_SGEMV4
            // 
            this.B_GPU_SGEMV4.Location = new System.Drawing.Point(15, 162);
            this.B_GPU_SGEMV4.Name = "B_GPU_SGEMV4";
            this.B_GPU_SGEMV4.Size = new System.Drawing.Size(172, 23);
            this.B_GPU_SGEMV4.TabIndex = 6;
            this.B_GPU_SGEMV4.Text = "SGEMV4";
            this.B_GPU_SGEMV4.UseVisualStyleBackColor = true;
            this.B_GPU_SGEMV4.Click += new System.EventHandler(this.B_GPU_SGEMV4_Click);
            // 
            // B_GPU_SGEMM
            // 
            this.B_GPU_SGEMM.Location = new System.Drawing.Point(15, 139);
            this.B_GPU_SGEMM.Name = "B_GPU_SGEMM";
            this.B_GPU_SGEMM.Size = new System.Drawing.Size(172, 23);
            this.B_GPU_SGEMM.TabIndex = 5;
            this.B_GPU_SGEMM.Text = "SGEMM3";
            this.B_GPU_SGEMM.UseVisualStyleBackColor = true;
            this.B_GPU_SGEMM.Click += new System.EventHandler(this.B_GPU_SGEMM_Click);
            // 
            // B_GPU_SharedPeak
            // 
            this.B_GPU_SharedPeak.Location = new System.Drawing.Point(15, 93);
            this.B_GPU_SharedPeak.Name = "B_GPU_SharedPeak";
            this.B_GPU_SharedPeak.Size = new System.Drawing.Size(172, 23);
            this.B_GPU_SharedPeak.TabIndex = 4;
            this.B_GPU_SharedPeak.Text = "SHARED MEMORY PEAK";
            this.B_GPU_SharedPeak.UseVisualStyleBackColor = true;
            this.B_GPU_SharedPeak.Click += new System.EventHandler(this.B_GPU_SharedPeak_Click);
            // 
            // B_GPU_RegisterPeak
            // 
            this.B_GPU_RegisterPeak.Location = new System.Drawing.Point(15, 69);
            this.B_GPU_RegisterPeak.Name = "B_GPU_RegisterPeak";
            this.B_GPU_RegisterPeak.Size = new System.Drawing.Size(172, 23);
            this.B_GPU_RegisterPeak.TabIndex = 3;
            this.B_GPU_RegisterPeak.Text = "REGISTER PEAK";
            this.B_GPU_RegisterPeak.UseVisualStyleBackColor = true;
            this.B_GPU_RegisterPeak.Click += new System.EventHandler(this.B_GPU_RegisterPeak_Click);
            // 
            // B_GPU_SGEMM4
            // 
            this.B_GPU_SGEMM4.Location = new System.Drawing.Point(15, 116);
            this.B_GPU_SGEMM4.Name = "B_GPU_SGEMM4";
            this.B_GPU_SGEMM4.Size = new System.Drawing.Size(172, 23);
            this.B_GPU_SGEMM4.TabIndex = 2;
            this.B_GPU_SGEMM4.Text = "SGEMM4";
            this.B_GPU_SGEMM4.UseVisualStyleBackColor = true;
            this.B_GPU_SGEMM4.Click += new System.EventHandler(this.B_GPU_SGEMM4_Click);
            // 
            // UpdateTimer
            // 
            this.UpdateTimer.Enabled = true;
            this.UpdateTimer.Interval = 1000;
            this.UpdateTimer.Tick += new System.EventHandler(this.UpdateTimer_Tick);
            // 
            // menuStrip1
            // 
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(984, 24);
            this.menuStrip1.TabIndex = 1;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // OPTest
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(984, 744);
            this.Controls.Add(this.TABs);
            this.Controls.Add(this.menuStrip1);
            this.Margin = new System.Windows.Forms.Padding(2);
            this.Name = "OPTest";
            this.Text = "OPLibs Benchmarks";
            this.Load += new System.EventHandler(this.OP_Load);
            this.TABs.ResumeLayout(false);
            this.back.ResumeLayout(false);
            this.TBConsole.ResumeLayout(false);
            this.groupBox5.ResumeLayout(false);
            this.groupBox5.PerformLayout();
            this.TBBenchmarks.ResumeLayout(false);
            this.GBCPUBenchmarks.ResumeLayout(false);
            this.GBCPUBenchmarks.PerformLayout();
            this.GBGPUBenchmarks.ResumeLayout(false);
            this.GBGPUBenchmarks.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TabControl TABs;
        private System.Windows.Forms.TabPage back;
        private System.Windows.Forms.GroupBox GBHost;
        private System.Windows.Forms.Button BSystemUpdate;
        private System.Windows.Forms.GroupBox GBDevice;
        private System.Windows.Forms.GroupBox GBProcess;
        private System.Windows.Forms.Timer UpdateTimer;
        private System.Windows.Forms.TabPage TBConsole;
        private System.Windows.Forms.GroupBox groupBox5;
        internal System.Windows.Forms.TextBox Display;
        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.TabPage TBBenchmarks;
        private System.Windows.Forms.Button B_GPU_SGEMM4;
        private System.Windows.Forms.GroupBox GBGPUBenchmarks;
        private System.Windows.Forms.Button B_GPU_RegisterPeak;
        private System.Windows.Forms.Button B_GPU_SharedPeak;
        private System.Windows.Forms.Button B_GPU_SGEMM;
        private System.Windows.Forms.Button B_CPU_SGEMM;
        private System.Windows.Forms.GroupBox GBCPUBenchmarks;
        private System.Windows.Forms.Button B_GPU_SGEMV4;
        private System.Windows.Forms.Button B_CPU_DCLV1F;
        private System.Windows.Forms.Button B_CPU_DCSV1F;
        private System.Windows.Forms.Button B_GPU_SGLV;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Button B_GPU_RUN_ALL;
        private System.Windows.Forms.Button B_CPU_RUN_ALL;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Button B_GPU_SGSV;
        private System.Windows.Forms.Button B_CPU_DGEMM;
        private System.Windows.Forms.Button B_CPU_MT;
        private System.Windows.Forms.Button B_GPU_MT;
        private System.Windows.Forms.Button B_GPU_SGEMV2;
        private System.Windows.Forms.Label L_GPU_MT_WITH_COPY;

    }
}

