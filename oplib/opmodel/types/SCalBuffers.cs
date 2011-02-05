using System;
using System.Collections.Generic;
using System.Text;

namespace OPModel.Types
{
		public class SCalBuffers
		{

				public CArray _device_dtpk_yy_w;									//discounted transition probability kernel
				public CArray _device_dtpkbuf_yy_w;								//discounted transition probability kernel buffer
				
				public CArray _device_dtpkpeps_yy_h;							//discounted transition probability kernel with lambda = 1+eps
				public CArray _device_dtpkmeps_yy_h;							//discounted transition probability kernel with lambda = 1-eps

				public CArray _device_dtpk_yy;									//discounted transition probability kernel solution
				public CArray _device_dtpkbuf_yy;								//buffer for discounted transition probability kernel solution

				public CArray _device_dprob_y;										//discounted probability to maturity from spot
				public CArray _device_dprobbuf_y;									//buffer for discounted probability to maturity from spot



				public double[] _host_dtpk_yy_w;									//discounted transition probability kernel
				public double[] _host_dtpkbuf_yy_w;								//discounted transition probability kernel buffer

				public double[] _host_dtpkpeps_yy_h;							//discounted transition probability kernel with lambda = 1+eps
				public double[] _host_dtpkmeps_yy_h;							//discounted transition probability kernel with lambda = 1-eps

				public double[] _host_dtpk_yy;									//discounted transition probability kernel solution
				public double[] _host_dtpkbuf_yy;								//buffer for discounted transition probability kernel solution

				public double[] _host_dprob_y;										//discounted probability to maturity from spot
				public double[] _host_dprobbuf_y;									//buffer for discounted probability to maturity from spot



				//public CArray _device_m_k;											//GPU ctpker_m_yy
				//public CArray _device_xtpk_yy_j;								//transition probability kernels, could be discounted or not
				//public CArray _device_xtpkbuf_yy_idx;					//buffer for transition probability kernels, could be discounted or not, idx could be j or m
				//public CArray _device_tpk_yy_m;
				//public CArray _device_ctpk_yy_m;
				//public CArray _device_dtpk_yy_m;								//discounted transition probability kernel
				//public CArray _device_dtpk_ky;									//discounted transition probability kernel
				//public CArray _device_dtpkbuf_ky;						  //discounted transition probability kernel
				//public CArray _device_rgstatus;

				//public double[] _host_xtpk_yy_j;									//transition probability kernels, could be discounted or not
				//public double[] _host_xtpkbuf_yy_idx;						//transition probability kernels, could be discounted or not, idx could be j or m
				//public double[] _host_tpk_yy_m;
				//public double[] _host_ctpk_yy_m;
				//public double[] _host_dtpk_yy_m;									//discounted transition probability kernel
				//public double[] _host_dtpk_ky;									  //discounted transition probability kernel
				//public double[] _host_dtpkbuf_ky;								//discounted transition probability kernel
				//public int[] _host_hash_ys_m;
				//public double[] _host_tpkbuf_ky;									//transition probability kernel from the spot

		}
}
