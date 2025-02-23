/* 
 * MIT License
 *
 *
 * Copyright (c) 2025 Kun Chen <kunchen@siom.ac.cn>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *
 */

#ifndef __CNear2Fresnel__
#define __CNear2Fresnel__

#include <string>
#include <mpi.h>
#include "mkl.h"
#include "mkl_dfti.h"
#include "run_environment.h"
#include "utility.h"

class CNear2Fresnel 
{
   private:
      // *************section 1***************
      // this section of member variables are for MPI and OpenMP setup
      int m_nProcs, m_Rank;
      MPI_Comm m_comm;
      int m_nThread;

      // *************section 2***************
      // this section of member variables will be read in from the output 
      // of the PSTD run on the internal model
      long m_nXGlb, m_nYGlb, m_nZGlb;	// the 3D sizes of the entire lattice of the internal model 
      long m_nYSurf, m_nZSurf;	// varibles to hold the sizes of the virtual sufaces, such as:
				        // Back & Front: m_nYGlb*m_nZSurf
     					// Left & Right: m_nXGlb*m_nZsurf
      					// Top & Bottom: m_nXGlb*m_nYSurf; also used for m_coskyy etc.
      long m_nABC;	// In case the read-in virtual surface Psi and DPsi do not vanish at the boundary,
			// we enforce that by multiplying a weight function to damp the surface terms to 0,
      			// to faciliate the FFT operation.  Using m_nABC as the width of the weight function
     			// is natuaral, because it will not affect the surface terms on the virtual box.
      QPrecision *m_Surf_x1, *m_SurfD_x1;  // Back surface
      QPrecision *m_Surf_x2, *m_SurfD_x2;  // Front surface
      QPrecision *m_Surf_y1, *m_SurfD_y1;  // Left surface
      QPrecision *m_Surf_y2, *m_SurfD_y2;  // Right surface
      QPrecision *m_Surf_z1, *m_SurfD_z1;  // Bottom surface
      QPrecision *m_Surf_z2, *m_SurfD_z2;  // Top surface
      
      // coordinate definition of the virtual surfaces
      long m_X1Vrtl, m_X2Vrtl;
      long m_Y1Vrtl, m_Y2Vrtl;
      long m_Z1Vrtl, m_Z2Vrtl;
      long m_nXVrtl, m_nYVrtl, m_nZVrtl;
      QPrecision m_dx, m_dy, m_dz;	// grid size
      QPrecision m_dx2, m_dy2, m_dz2;	// half grid size
      QPrecision m_OrigX, m_OrigY, m_OrigZ;  // Origin of the model
      QPrecision *m_x, *m_y, *m_z;  // coordinates of each cell center
      QPrecision m_x1v, m_x2v, m_y1v, m_y2v, m_z1v, m_z2v;  // positions of virtual surfaces

      // neutron energy and its corresponding wavelength
      QPrecision m_E0, m_lambdabar0;

      // *************section 3***************
      // this section of member variables are for local computations
      // FFT related
      QPrecision *m_kx, *m_ky, *m_kz;  // FFT k
      // variables to facilitate computations
      QPrecision *m_coskxx, *m_coskyy, *m_coskzz;  // matrices: exp(ik_x*x), exp(ik_y*y), exp(ik_z*z)
      QPrecision *m_sinkxx, *m_sinkyy, *m_sinkzz;  //           dimension m_nXVrtl*(m_nXGlb*+padding)
      long m_nXkxx, m_nYkyy, m_nZkzz;		   // the 2nd dimension size of coskxx, sinkxx, etc.
      QPrecision *m_rsincx, *m_rsincy, *m_rsincz;  // place holder for the sinc(\delta k)*exp(ik) in Green's function
      QPrecision *m_isincx, *m_isincy, *m_isincz;

      QPrecision m_cfx, m_cfy, m_cfz;  // constant overall factors for surface calculations

      // parallel performance related, load-balance considerations
      long m_iStart, m_iEnd, m_jStart, m_jEnd, m_kStart, m_kEnd;  // indices for load-balance purpose
      // [m_iStart, m_iEnd] for the Back & Front surfaces, each cell contains m_nYGlb*m_nZGlb k-sums;
      // [m_jStart, m_jEnd] for the Left & Right Surfaces, each cell contains m_nXGlb*m_nZGlb k-sums;
      // [m_kStart, m_kEnd] for the Bottom & Top surfaces, each cell contains m_nXGlb*m_nYGlb k-sums.

      void ConvertR2K(QPrecision* &Surf, long l0, long l1, long ndim1,
		      long a1Vrtl, long a2Vrtl, long a1ABC, long a2ABC,
		      long b1Vrtl, long b2Vrtl, long b1ABC, long b2ABC);
      void BackSurfContribution(long j, long k, QPrecision xbar, QPrecision ybar,
	    QPrecision zbar, QPrecision R0, QPrecision &cpsi_r, QPrecision &cpsi_i);
      void FrontSurfContribution(long j, long k, QPrecision xbar, QPrecision ybar,
	    QPrecision zbar, QPrecision R0, QPrecision &cpsi_r, QPrecision &cpsi_i);
      void LeftSurfContribution(long i, long k, QPrecision xbar, QPrecision ybar,
	    QPrecision zbar, QPrecision R0, QPrecision &cpsi_r, QPrecision &cpsi_i);
      void RightSurfContribution(long i, long k, QPrecision xbar, QPrecision ybar,
	    QPrecision zbar, QPrecision R0, QPrecision &cpsi_r, QPrecision &cpsi_i);
      void BottomSurfContribution(long i, long j, QPrecision xbar, QPrecision ybar,
	    QPrecision zbar, QPrecision R0, QPrecision &cpsi_r, QPrecision &cpsi_i);
      void TopSurfContribution(long i, long j, QPrecision xbar, QPrecision ybar,
	    QPrecision zbar, QPrecision R0, QPrecision &cpsi_r, QPrecision &cpsi_i);
      void HandleError(int val, std::string id_str);

   public:
      CNear2Fresnel(MPI_Comm comm);
      ~CNear2Fresnel();
      void InitData(std::string& surfname);
      void Psi(QPrecision* r, QPrecision* psi, QPrecision* SphericalWaveFactor);
      void SaveSurfTerm(std::string savename);
};

#endif
