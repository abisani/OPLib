// SGrid.cs --- Part of the project OPLib 1.0, a high performance pricing library
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

namespace OPModel.Types
{
    /** Defines a discretisation of the real line.
     **/
    abstract public class SGrid
    {
        private CDevice _device;
        private bool _initialized;
        private int _d;                     
        private int _nx;
        private int _nr;
        private int _ni;
        private DateTime[] _t_i;
        private TimeSpan _dt0;
        private double[] _host_d_xval_y;
        public double[] _host_d_rval_y;
        private int _x0;
        private int _r0;
        private int _y0;
        private double[] _host_d_invm;             
        private DateTime _today;
        private int _ncurves;
        private int _ntmcols;
        private double[] _host_d_df_i;
        private float[] _host_s_df_i;
        internal CArray _device_df_i;
        private double[] _host_d_ir_yi;
        private float[] _host_s_ir_yi;
        internal CArray _device_ir_yi;


        public SGrid(CDevice device, int nx, int nr, DateTime today, DateTime[] t_i,
                     TimeSpan dt0, double xval0, double[] xpivot_p,
                     double[] xgridspacing_p, double[] rval_r, int r0)
        {
            if (_initialized) throw new System.Exception();
            _initialized = true;
            init(device, nx, nr, today, t_i, dt0, xval0, xpivot_p, xgridspacing_p, rval_r, r0);
        }

        public CArray device_df_i
        {
            get
            {
                return _device_df_i;
            }
        }

        public CArray device_ir_yi
        {
            get
            {
                return _device_ir_yi;
            }
        }

        public double[] host_d_df_i
        {
            get
            {
                return _host_d_df_i;
            }
        }

        public float[] host_s_df_i
        {
            get
            {
                return _host_s_df_i;
            }
        }

        public double[] host_d_ir_yi
        {
            get
            {
                return _host_d_ir_yi;
            }
        }

        public float[] host_s_ir_yi
        {
            get
            {
                return _host_s_ir_yi;
            }
        }

        public void host_d_set_discount_curve(double[] df_i)
        {
            if (df_i != null)
            {
                if (ni != df_i.Length) throw new System.Exception();
                _host_d_df_i = new double[ni];
            }

            _host_d_ir_yi = new double[ni * d];
            for (int i = 0; i < ni; i++)
            {
                if (df_i != null)
                {
                    _host_d_df_i[i] = df_i[i];
                }
                if (df_i != null)
                {
                    if (i == 0)
                    {
                        for (int y = 0; y < d; y++)
                        {
                            _host_d_ir_yi[y + 0] = -Math.Log(df_i[i]) / ((t_i[i] - today).Days / 365.25);
                        }
                    }
                    else
                    {
                        for (int y = 0; y < d; y++)
                        {
                            _host_d_ir_yi[y + d * i] = -Math.Log(df_i[i]) / ((t_i[i] - t_i[i - 1]).Days / 365.25);
                        }
                    }
                }
            }
        }



        public void host_s_set_discount_curve(double[] df_i)
        {
            host_d_set_discount_curve(df_i);
            if (df_i != null)
            {
                if (ni != df_i.Length) throw new System.Exception();
                _host_s_df_i = new float[ni];
            }

            _host_s_ir_yi = new float[ni * d];

            for (int i = 0; i < ni; i++)
            {
                if (df_i != null)
                {
                    _host_s_df_i[i] = (float)df_i[i];
                }
                if (df_i != null)
                {
                    if (i == 0)
                    {
                        for (int y = 0; y < d; y++)
                        {
                            _host_s_ir_yi[y + 0] = -(float)Math.Log(df_i[i]) / ((t_i[i] - today).Days / 365.25f);
                        }
                    }
                    else
                    {
                        for (int y = 0; y < d; y++)
                        {
                            _host_s_ir_yi[y + d * i] = -(float)Math.Log(host_s_df_i[i]) / ((t_i[i] - t_i[i - 1]).Days / 365.25f);
                        }
                    }
                }
            }
        }

        public void device_set_discount_curve(double[] df_i, CModel model)
        {

            host_d_set_discount_curve(df_i);
            CArray.alloc(ref _device_ir_yi, _host_d_ir_yi.Length, EType.float_t, EMemorySpace.device, model, "device_ir_yi");

            if (df_i != null)
            {
                CArray.alloc(ref _device_df_i, host_d_df_i.Length, EType.float_t, EMemorySpace.device, model, "device_df_i");
                float[] fdf_i = new float[host_d_df_i.Length];
                for (int i = 0; i < host_d_df_i.Length; i++)
                {
                    fdf_i[i] = (float)host_d_df_i[i];
                }
                CArray.copy(ref _device_df_i, fdf_i);
            }
        }


        public int ncurves
        {
            get
            {
                return _ncurves;
            }
        }

        public int ntmcols
        {
            get
            {
                return _ntmcols;
            }
        }



        public void Set_ncurves(int ncurves)
        {
            _ncurves = ncurves;
        }

        public void Set_ntmcols(int ntmcols)
        {
            _ntmcols = ntmcols;
        }

        protected void init(CDevice device, int nx, int nr,
                            DateTime today, DateTime[] t_i, TimeSpan dt0, double xval0,
                            double[] xpivot_p, double[] xgridspacing_p, double[] rval_r, int r0)
        {
            this._device = device;
            this._d = (int)(nx * nr);
            this._nx = nx;
            this._nr = nr;
            this._today = today;
            if (t_i == null) throw new System.Exception();
            this._ni = t_i.Length;
            this._t_i = new DateTime[ni]; t_i.CopyTo(this.t_i, 0);
            this._dt0 = dt0;
            this._host_d_xval_y = new double[d];
            this._host_d_rval_y = new double[d];

            if (nr > 1)
            {
                if (rval_r.Length != nr) throw new System.Exception();
            }
            else
            {
                if (rval_r != null) throw new System.Exception();
            }

            this._r0 = r0;           
            mkgrid(ref _host_d_xval_y, ref _host_d_rval_y, ref _x0, r0, ref _y0, xval0, xpivot_p, xgridspacing_p, rval_r);
            set_invm(pregen());
        }




        public SGrid(SGrid grid, bool deep_copy)
        {
            if (grid == null || _initialized) throw new System.Exception();

            _initialized = true;
            this._device = grid._device;
            this._d = grid.d;
            this._nx = grid.nx;
            this._nr = grid.nr;
            this._ni = grid.ni;
            this._dt0 = grid.dt0;
            this._x0 = grid._x0;
            this._r0 = grid._r0;
            this._y0 = grid._y0;
            this._today = grid._today;

            if (deep_copy)
            {
                this._t_i = new DateTime[ni]; grid.t_i.CopyTo(this.t_i, 0);
                this._host_d_xval_y = new double[d]; grid._host_d_xval_y.CopyTo(this._host_d_xval_y, 0);
                set_invm(grid._host_d_invm);
            }
            else
            {
                this._t_i = grid.t_i;
                this._host_d_xval_y = grid._host_d_xval_y;
                this._host_d_invm = grid._host_d_invm;
            }
        }



        public double[] host_d_invm
        {
            get
            {
                return _host_d_invm;
            }
        }


        public int d
        {
            get
            {
                return _d;
            }
        }

        public int nx
        {
            get
            {
                return _nx;
            }
        }


        public int nr
        {
            get
            {
                return _nr;
            }
        }

        public DateTime[] t_i
        {
            get
            {
                return _t_i;
            }
        }

        public int ni
        {
            get
            {
                return _ni;
            }
        }

        public TimeSpan dt0
        {
            get
            {
                return _dt0;
            }
        }

        public double host_d_xval(int y)
        {
            return _host_d_xval_y[y];
        }

        public double host_d_rval(int y)
        {
            return _host_d_rval_y[y];
        }

        public double[] host_d_xval_y
        {
            get
            {
                return _host_d_xval_y;
            }
        }

        public double[] host_d_rval_y
        {
            get
            {
                return _host_d_rval_y;
            }
        }


        public int x0
        {
            get
            {
                return _x0;
            }
        }

        public int r0
        {
            get
            {
                return _r0;
            }
        }

        public int y0
        {
            get
            {
                return _y0;
            }
        }

        public CDevice device
        {
            get
            {
                return _device;
            }
        }

        public EFloatingPointUnit fpu
        {
            get
            {
                return device.fpu;
            }
        }

        public EFloatingPointPrecision fpp
        {
            get
            {
                return device.fpp;
            }
        }

        public DateTime today
        {
            get
            {
                return _today;
            }
        }

        protected SGrid()
        {
            _initialized = false;
        }

        protected void set_invm(double[] invm)
        {
            this._host_d_invm = new double[invm.Length];
            invm.CopyTo(this._host_d_invm, 0);
        }

        abstract public double[] pregen();

        abstract public void mkgrid(ref double[] xval_y, ref double[] _rval_y, ref int x0, int r0, ref int y0, double xval0,
            double[] xpivot_p, double[] xgridspacing_p, double[] rval_r);


        public int ycoord(int x, int r)
        {
            return x + nx * r;
        }

        public int xcoord(int y)
        {
            return y % nx;
        }

        public int rcoord(int y)
        {
            return y / nx;
        }

        public void coords(int y, ref int x, ref int r)
        {
            x = y % nx;
            r = y / nx;
        }

        protected void xgrid(ref double[] xval_y, ref int x0, double[] xpivot_p, double[] xgridspacing_p, double xval0)
        {
            int np = xpivot_p.Length;
            if (np <= 2) throw new System.Exception();
            if (xpivot_p.Length != xgridspacing_p.Length) throw new System.Exception();

            double f0, f1, f2;
            double scale0 = 1, scale1 = 1;
            int niter = 10;

            f0 = xpivot_p[np - 1] / grid_iterate(xval_y, xpivot_p, xgridspacing_p, 1);
            if (f0 > 1)
            {
                for (int iter = 1; ; iter++)
                {
                    scale0 = 1 + 0.1 * iter;
                    f1 = xpivot_p[np - 1] / grid_iterate(xval_y, xpivot_p, xgridspacing_p, scale0);
                    if (f1 < 1) break;
                    else
                    {
                        scale1 = scale0;
                    }
                }
            }

            if (f0 <= 1)
            {
                for (int iter = 1; ; iter++)
                {
                    scale1 = 1 / (1 + 0.1 * iter);
                    f1 = xpivot_p[np - 1] / grid_iterate(xval_y, xpivot_p, xgridspacing_p, scale1);
                    if (f1 > 1) break;
                    else
                    {
                        scale0 = scale1;
                    }
                }
            }


            for (int iter = 0; iter < niter; iter++)
            {
                double mid = 0.5 * (scale0 + scale1);
                f2 = xpivot_p[np - 1] / grid_iterate(xval_y, xpivot_p, xgridspacing_p, mid);
                if (f2 > 1) scale1 = mid;
                if (f2 <= 1) scale0 = mid;
            }

            double minerror = Math.Abs(xval_y[0] - xval0);
            for (int x = 1; x < nx; x++)
            {
                double errore;
                errore = Math.Abs(xval_y[x] - xval0);
                if (errore < minerror)
                {
                    x0 = x;
                    minerror = errore;
                }
            }

            double scale = xval0 / xval_y[x0];

            for (int x = 0; x < nx; x++)
            {
                xval_y[x] *= scale;
            }

        }

        private double grid_iterate(double[] xval_y, double[] xpivot_p, double[] xgridspacing_p, double scale)
        {
            xval_y[0] = xpivot_p[0];
            double step = xgridspacing_p[0];

            int np = xpivot_p.Length;
            int p = 1;
            double nextp = 0;
            if (p < np)
            {
                nextp = xpivot_p[p];
            }
 
            for (int x = 1; x < nx; x++)
            {
                xval_y[x] = xval_y[x - 1] + scale * step;

                if (p < np)
                {
                    if (xval_y[x] > nextp)
                    {
                        step = xgridspacing_p[p];
                        p++;
                        if (p < np)
                        {
                            nextp = xpivot_p[p];
                        }
                    }
                }
            }

            return xval_y[nx - 1];
        }
    }
}
