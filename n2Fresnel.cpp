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

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <omp.h>
#include <stdio.h>
#include "run_environment.h"
#include "utility.h"
#include "n2Fresnel.h"

using namespace std;

CNear2Fresnel::CNear2Fresnel(MPI_Comm comm)
{
   m_comm=comm;

   m_Surf_x1=NULL;
   m_Surf_x2=NULL;
   m_Surf_y1=NULL;
   m_Surf_y2=NULL;
   m_Surf_z1=NULL;
   m_Surf_z2=NULL;

   m_SurfD_x1=NULL;
   m_SurfD_x2=NULL;
   m_SurfD_y1=NULL;
   m_SurfD_y2=NULL;
   m_SurfD_z1=NULL;
   m_SurfD_z2=NULL;

   m_x=NULL;
   m_y=NULL;
   m_z=NULL;

   m_kx=NULL;
   m_ky=NULL;
   m_kz=NULL;

   m_coskxx=m_sinkxx=NULL;
   m_coskyy=m_sinkyy=NULL;
   m_coskzz=m_sinkzz=NULL;

   m_rsincx=m_isincx=NULL;
   m_rsincy=m_isincy=NULL;
   m_rsincz=m_isincz=NULL;
}

void CNear2Fresnel::InitData(std::string& surfname)
{
   MPI_Comm_size(m_comm, &m_nProcs);
   MPI_Comm_rank(m_comm, &m_Rank);
   m_nThread=__OMP_NUM_THREADS__;

   MPI_File fh;
   MPI_Offset disp;

   // The inputs are read in from the virtual surface data of a pre-run PSTD simulation
   if (MPI_File_open(m_comm, surfname.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)!=MPI_SUCCESS)
      throw runtime_error("Cannot open data file " + surfname);

   // The PSTD simulation can be of either float or double presion,
   // whereas the current code can be compiled either as a float 
   // or a double version.  The precisions of both must match.  
   int precision;
   if (!m_Rank) {
      MPI_File_read(fh, &precision, 1, MPIInt, MPI_STATUS_IGNORE);

      if (precision!=sizeof(QPrecision)) {
         stringstream sst;
         if (precision==4)
            sst << "Error: precision incompatible\n" << surfname
               << ": float data processed as double";
         else if (precision==8)
            sst << "Error: precision incompatible\n" << surfname
               << ": double data processed as float";
         else
            sst << "Error: incompatible data file " << surfname;

         throw runtime_error(sst.str());
      }

      MPI_File_read(fh, &m_nXGlb, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_nYGlb, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_nZGlb, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_X1Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_X2Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_Y1Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_Y2Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_Z1Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_Z2Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_nABC, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_dx, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_dy, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_dz, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_OrigX, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_OrigY, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_OrigZ, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_E0, 1, MPIQPrecision, MPI_STATUS_IGNORE);
   }

   disp=sizeof(precision)+sizeof(m_nXGlb)+sizeof(m_nYGlb)+sizeof(m_nZGlb)
      +sizeof(m_X1Vrtl)+sizeof(m_X2Vrtl)+sizeof(m_Y1Vrtl)+sizeof(m_Y2Vrtl)
      +sizeof(m_Z1Vrtl)+sizeof(m_Z2Vrtl)+sizeof(m_nABC)+sizeof(m_dx)
      +sizeof(m_dy)+sizeof(m_dz)+sizeof(m_OrigX)+sizeof(m_OrigY)
      +sizeof(m_OrigZ)+sizeof(m_E0);
   MPI_Bcast(&precision, 1, MPIInt, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m_nXGlb, 1, MPILong, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m_nYGlb, 1, MPILong, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m_nZGlb, 1, MPILong, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m_X1Vrtl, 1, MPILong, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m_X2Vrtl, 1, MPILong, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m_Y1Vrtl, 1, MPILong, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m_Y2Vrtl, 1, MPILong, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m_Z1Vrtl, 1, MPILong, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m_Z2Vrtl, 1, MPILong, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m_nABC, 1, MPILong, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m_dx, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m_dy, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m_dz, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m_OrigX, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m_OrigY, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m_OrigZ, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m_E0, 1, MPIQPrecision, 0, MPI_COMM_WORLD);

   m_nZSurf=m_nZGlb*2;
   int val=Init_Aligned_Matrix_2D<QPrecision>(m_Surf_x1, m_nYGlb, m_nZSurf, sizeof(QPrecision)*2);
   if (val) HandleError(val, "m_Surf_x1");

   val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfD_x1, m_nYGlb, m_nZSurf, sizeof(QPrecision)*2);
   if (val) HandleError(val, "m_SurfD_x1");

   val=Init_Aligned_Matrix_2D<QPrecision>(m_Surf_x2, m_nYGlb, m_nZSurf, sizeof(QPrecision)*2);
   if (val) HandleError(val, "m_Surf_x2");

   val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfD_x2, m_nYGlb, m_nZSurf, sizeof(QPrecision)*2);
   if (val) HandleError(val, "m_SurfD_x2");

   val=Init_Aligned_Matrix_2D<QPrecision>(m_Surf_y1, m_nXGlb, m_nZSurf, sizeof(QPrecision)*2);
   if (val) HandleError(val, "m_Surf_y1");

   val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfD_y1, m_nXGlb, m_nZSurf, sizeof(QPrecision)*2);
   if (val) HandleError(val, "m_SurfD_y1");

   val=Init_Aligned_Matrix_2D<QPrecision>(m_Surf_y2, m_nXGlb, m_nZSurf, sizeof(QPrecision)*2);
   if (val) HandleError(val, "m_Surf_y2");

   val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfD_y2, m_nXGlb, m_nZSurf, sizeof(QPrecision)*2);
   if (val) HandleError(val, "m_SurfD_y2");

   m_nYSurf=m_nYGlb*2;
   val=Init_Aligned_Matrix_2D<QPrecision>(m_Surf_z1, m_nXGlb, m_nYSurf, sizeof(QPrecision)*2);
   if (val) HandleError(val, "m_Surf_z1");

   val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfD_z1, m_nXGlb, m_nYSurf, sizeof(QPrecision)*2);
   if (val) HandleError(val, "m_SurfD_z1");

   val=Init_Aligned_Matrix_2D<QPrecision>(m_Surf_z2, m_nXGlb, m_nYSurf, sizeof(QPrecision)*2);
   if (val) HandleError(val, "m_Surf_z2");

   val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfD_z2, m_nXGlb, m_nYSurf, sizeof(QPrecision)*2);
   if (val) HandleError(val, "m_SurfD_z2");

   m_nXVrtl=m_X2Vrtl-m_X1Vrtl;  // the number of cell centers, not the number of grids
   val=Init_Aligned_Vector<QPrecision>(m_x, m_nXVrtl);
   if (val) HandleError(val, "m_x");

   m_nYVrtl=m_Y2Vrtl-m_Y1Vrtl;
   val=Init_Aligned_Vector<QPrecision>(m_y, m_nYVrtl);
   if (val) HandleError(val, "m_y");

   m_nZVrtl=m_Z2Vrtl-m_Z1Vrtl;
   val=Init_Aligned_Vector<QPrecision>(m_z, m_nZVrtl);
   if (val) HandleError(val, "m_z");

   m_lambdabar0=MICROEV2NMBAR/SQRT(m_E0);

   m_dx2=m_dx/2.0;
   m_x1v=((QPrecision) m_X1Vrtl-m_OrigX)*m_dx;  // x-coordinate for back surface, in unit of lambdabar0
   m_x2v=((QPrecision) m_X2Vrtl-m_OrigX)*m_dx;  // x-coordinate for front surface

   m_dy2=m_dy/2.0;
   m_y1v=((QPrecision) m_Y1Vrtl-m_OrigY)*m_dy;  // y-coordinate for left surface
   m_y2v=((QPrecision) m_Y2Vrtl-m_OrigY)*m_dy;  // y-coordinate for right surface

   m_dz2=m_dz/2.0;
   m_z1v=((QPrecision) m_Z1Vrtl-m_OrigZ)*m_dz;  // z-coordinate for bottom surface
   m_z2v=((QPrecision) m_Z2Vrtl-m_OrigZ)*m_dz;  // z-coordinate for top surface

   // k indices
   val=Init_Aligned_Vector<QPrecision>(m_kx, m_nXGlb);
   if (val) HandleError(val, "m_kx");

   val=Init_Aligned_Vector<QPrecision>(m_ky, m_nYGlb);
   if (val) HandleError(val, "m_ky");

   val=Init_Aligned_Vector<QPrecision>(m_kz, m_nZGlb);
   if (val) HandleError(val, "m_kz");

   // init exp(ik_x*x), exp(ik_y*y), exp(ik_z*z) matrices
   m_nXkxx=m_nXGlb;
   val=Init_Aligned_Matrix_2D<QPrecision>(m_coskxx, m_nXVrtl, m_nXkxx, sizeof(QPrecision));
   if (val) HandleError(val, "m_coskxx");
   val=Init_Aligned_Matrix_2D<QPrecision>(m_sinkxx, m_nXVrtl, m_nXkxx, sizeof(QPrecision));
   if (val) HandleError(val, "m_sinkxx");

   m_nYkyy=m_nYGlb;
   val=Init_Aligned_Matrix_2D<QPrecision>(m_coskyy, m_nYVrtl, m_nYkyy, sizeof(QPrecision));
   if (val) HandleError(val, "m_coskyy");
   val=Init_Aligned_Matrix_2D<QPrecision>(m_sinkyy, m_nYVrtl, m_nYkyy, sizeof(QPrecision));
   if (val) HandleError(val, "m_sinkyy");

   m_nZkzz=m_nZGlb;
   val=Init_Aligned_Matrix_2D<QPrecision>(m_coskzz, m_nZVrtl, m_nZkzz, sizeof(QPrecision));
   if (val) HandleError(val, "m_coskzz");
   val=Init_Aligned_Matrix_2D<QPrecision>(m_sinkzz, m_nZVrtl, m_nZkzz, sizeof(QPrecision));
   if (val) HandleError(val, "m_sinkzz");

   // allocate memory for sinc(\delta_k)*exp(ik) indices
   val=Init_Aligned_Vector<QPrecision>(m_rsincx, m_nXGlb);
   if (val) HandleError(val, "m_rsincx");
   val=Init_Aligned_Vector<QPrecision>(m_isincx, m_nXGlb);
   if (val) HandleError(val, "m_isincx");

   val=Init_Aligned_Vector<QPrecision>(m_rsincy, m_nYGlb);
   if (val) HandleError(val, "m_rsincy");
   val=Init_Aligned_Vector<QPrecision>(m_isincy, m_nYGlb);
   if (val) HandleError(val, "m_isincy");

   val=Init_Aligned_Vector<QPrecision>(m_rsincz, m_nZGlb);
   if (val) HandleError(val, "m_rsincz");
   val=Init_Aligned_Vector<QPrecision>(m_isincz, m_nZGlb);
   if (val) HandleError(val, "m_isincz");

   // some constant factors needed in surface calculations
   m_cfx=m_dy*m_dz/(FOURPI*m_nYGlb*m_nZGlb);
   m_cfy=m_dx*m_dz/(FOURPI*m_nXGlb*m_nZGlb);
   m_cfz=m_dx*m_dy/(FOURPI*m_nXGlb*m_nYGlb);

#ifdef __INIT_STATUS__
   stringstream sstr;
   sstr << "Rank " << m_Rank << ":" << endl;
   sstr << "\t# of threads: " << m_nThread << endl;
   sstr << "\tPrecision: " << precision << endl;
   sstr << "\tTotal grids (xGlb,yGlb,zGlb): (" << m_nXGlb << "," <<
      m_nYGlb << "," << m_nZGlb << ")" << endl;
   sstr << "\tVirtual surfaces: X(" << m_X1Vrtl << "," << m_X2Vrtl <<
      ") Y(" << m_Y1Vrtl << "," << m_Y2Vrtl << ") Z(" << m_Z1Vrtl <<
      "," << m_Z2Vrtl << ")" << endl;
   sstr << "\t(dx, dy, dz): (" << m_dx << "," << m_dy << "," << m_dz <<
      ")" << endl;
   sstr << "\tOrigin (" << m_OrigX << "," << m_OrigY << "," << m_OrigZ <<
      ")" << endl;
   sstr << "\tEnergy: " << m_E0 << endl;
   sstr << "\tnYSurf: " << m_nYSurf << "\tnZSurf: " << m_nZSurf << endl;
   sstr << "\tnXkxx: " << m_nXkxx << "\tnYkyy: " << m_nYkyy << 
      "\tnZkzz: " << m_nZkzz << endl;
   cout << sstr.str() << endl;
#endif

   int back_psi=0;
   int back_dpsi=1%m_nProcs;
   int front_psi=2%m_nProcs;
   int front_dpsi=3%m_nProcs;
   int left_psi=4%m_nProcs;
   int left_dpsi=5%m_nProcs;
   int right_psi=6%m_nProcs;
   int right_dpsi=7%m_nProcs;
   int bottom_psi=8%m_nProcs;
   int bottom_dpsi=9%m_nProcs;
   int top_psi=10%m_nProcs;
   int top_dpsi=11%m_nProcs;

   MPI_Datatype newtype;
   int sizes[2], subsizes[2], starts[2];

   long incre=m_nYGlb*m_nZGlb*2*sizeof(QPrecision);
   sizes[0]=m_nYGlb; sizes[1]=m_nZSurf;
   subsizes[0]=m_nYGlb; subsizes[1]=m_nZGlb*2;
   starts[0]=0; starts[1]=0;
   MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPIQPrecision, &newtype);
   MPI_Type_commit(&newtype);
   // back surface psi data
   if (m_Rank==back_psi)
      MPI_File_read_at(fh, disp, m_Surf_x1, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // back surface dpsi data
   if (m_Rank==back_dpsi)
      MPI_File_read_at(fh, disp, m_SurfD_x1, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // front surface psi data
   if (m_Rank==front_psi)
      MPI_File_read_at(fh, disp, m_Surf_x2, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // front surface dpsi data
   if (m_Rank==front_dpsi)
      MPI_File_read_at(fh, disp, m_SurfD_x2, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;
   MPI_Type_free(&newtype);

   incre=m_nXGlb*m_nZGlb*2*sizeof(QPrecision);
   sizes[0]=m_nXGlb; sizes[1]=m_nZSurf;
   subsizes[0]=m_nXGlb; subsizes[1]=m_nZGlb*2;
   starts[0]=0; starts[1]=0;
   MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPIQPrecision, &newtype);
   MPI_Type_commit(&newtype);
   // left surface psi data
   if (m_Rank==left_psi)
      MPI_File_read_at(fh, disp, m_Surf_y1, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // left surface dpsi data
   if (m_Rank==left_dpsi)
      MPI_File_read_at(fh, disp, m_SurfD_y1, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // right surface psi data
   if (m_Rank==right_psi)
      MPI_File_read_at(fh, disp, m_Surf_y2, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // right surface dpsi data
   if (m_Rank==right_dpsi)
      MPI_File_read_at(fh, disp, m_SurfD_y2, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;
   MPI_Type_free(&newtype);

   incre=m_nXGlb*m_nYGlb*2*sizeof(QPrecision);
   sizes[0]=m_nXGlb; sizes[1]=m_nYSurf;
   subsizes[0]=m_nXGlb; subsizes[1]=m_nYGlb*2;
   starts[0]=0; starts[1]=0;
   MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPIQPrecision, &newtype);
   MPI_Type_commit(&newtype);
   // bottom surface psi data
   if (m_Rank==bottom_psi)
      MPI_File_read_at(fh, disp, m_Surf_z1, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // bottom surface dpsi data
   if (m_Rank==bottom_dpsi)
      MPI_File_read_at(fh, disp, m_SurfD_z1, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // top surface psi data
   if (m_Rank==top_psi)
      MPI_File_read_at(fh, disp, m_Surf_z2, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // top surface dpsi data
   if (m_Rank==top_dpsi)
      MPI_File_read_at(fh, disp, m_SurfD_z2, 1, newtype, MPI_STATUS_IGNORE);
   MPI_Type_free(&newtype);

   MPI_File_close(&fh);

   SaveSurfTerm(string("BeforeFFT"));

   if (m_Rank==back_psi)
      ConvertR2K(m_Surf_x1, m_nYGlb, m_nZGlb, m_nZSurf, m_Y1Vrtl, m_Y2Vrtl, m_nABC-1,
	    m_nYGlb-m_nABC, m_Z1Vrtl, m_Z2Vrtl, m_nABC-1, m_nZGlb-m_nABC);
   if (m_Rank==back_dpsi)
      ConvertR2K(m_SurfD_x1, m_nYGlb, m_nZGlb, m_nZSurf, m_Y1Vrtl, m_Y2Vrtl, m_nABC-1,
	    m_nYGlb-m_nABC, m_Z1Vrtl, m_Z2Vrtl, m_nABC-1, m_nZGlb-m_nABC);
   if (m_Rank==front_psi)
      ConvertR2K(m_Surf_x2, m_nYGlb, m_nZGlb, m_nZSurf, m_Y1Vrtl, m_Y2Vrtl, m_nABC-1,
	    m_nYGlb-m_nABC, m_Z1Vrtl, m_Z2Vrtl, m_nABC-1, m_nZGlb-m_nABC);
   if (m_Rank==front_dpsi)
      ConvertR2K(m_SurfD_x2, m_nYGlb, m_nZGlb, m_nZSurf, m_Y1Vrtl, m_Y2Vrtl, m_nABC-1,
	    m_nYGlb-m_nABC, m_Z1Vrtl, m_Z2Vrtl, m_nABC-1, m_nZGlb-m_nABC);
   if (m_Rank==left_psi)
      ConvertR2K(m_Surf_y1, m_nXGlb, m_nZGlb, m_nZSurf, m_X1Vrtl, m_X2Vrtl, m_nABC-1,
	    m_nXGlb-m_nABC, m_Z1Vrtl, m_Z2Vrtl, m_nABC-1, m_nZGlb-m_nABC);
   if (m_Rank==left_dpsi)
      ConvertR2K(m_SurfD_y1, m_nXGlb, m_nZGlb, m_nZSurf, m_X1Vrtl, m_X2Vrtl, m_nABC-1,
	    m_nXGlb-m_nABC, m_Z1Vrtl, m_Z2Vrtl, m_nABC-1, m_nZGlb-m_nABC);
   if (m_Rank==right_psi)
      ConvertR2K(m_Surf_y2, m_nXGlb, m_nZGlb, m_nZSurf, m_X1Vrtl, m_X2Vrtl, m_nABC-1,
	    m_nXGlb-m_nABC, m_Z1Vrtl, m_Z2Vrtl, m_nABC-1, m_nZGlb-m_nABC);
   if (m_Rank==right_dpsi)
      ConvertR2K(m_SurfD_y2, m_nXGlb, m_nZGlb, m_nZSurf, m_X1Vrtl, m_X2Vrtl, m_nABC-1,
	    m_nXGlb-m_nABC, m_Z1Vrtl, m_Z2Vrtl, m_nABC-1, m_nZGlb-m_nABC);
   if (m_Rank==bottom_psi)
      ConvertR2K(m_Surf_z1, m_nXGlb, m_nYGlb, m_nYSurf, m_X1Vrtl, m_X2Vrtl, m_nABC-1,
	    m_nXGlb-m_nABC, m_Y1Vrtl, m_Y2Vrtl, m_nABC-1, m_nYGlb-m_nABC);
   if (m_Rank==bottom_dpsi)
      ConvertR2K(m_SurfD_z1, m_nXGlb, m_nYGlb, m_nYSurf, m_X1Vrtl, m_X2Vrtl, m_nABC-1,
	    m_nXGlb-m_nABC, m_Y1Vrtl, m_Y2Vrtl, m_nABC-1, m_nYGlb-m_nABC);
   if (m_Rank==top_psi)
      ConvertR2K(m_Surf_z2, m_nXGlb, m_nYGlb, m_nYSurf, m_X1Vrtl, m_X2Vrtl, m_nABC-1,
	    m_nXGlb-m_nABC, m_Y1Vrtl, m_Y2Vrtl, m_nABC-1, m_nYGlb-m_nABC);
   if (m_Rank==top_dpsi)
      ConvertR2K(m_SurfD_z2, m_nXGlb, m_nYGlb, m_nYSurf, m_X1Vrtl, m_X2Vrtl, m_nABC-1,
	    m_nXGlb-m_nABC, m_Y1Vrtl, m_Y2Vrtl, m_nABC-1, m_nYGlb-m_nABC);

   SaveSurfTerm(string("AfterFFT"));

   MPI_Bcast(m_Surf_x1, m_nYGlb*m_nZSurf, MPIQPrecision, back_psi, m_comm);
   MPI_Bcast(m_SurfD_x1, m_nYGlb*m_nZSurf, MPIQPrecision, back_dpsi, m_comm);
   MPI_Bcast(m_Surf_x2, m_nYGlb*m_nZSurf, MPIQPrecision, front_psi, m_comm);
   MPI_Bcast(m_SurfD_x2, m_nYGlb*m_nZSurf, MPIQPrecision, front_dpsi, m_comm);

   MPI_Bcast(m_Surf_y1, m_nXGlb*m_nZSurf, MPIQPrecision, left_psi, m_comm);
   MPI_Bcast(m_SurfD_y1, m_nXGlb*m_nZSurf, MPIQPrecision, left_dpsi, m_comm);
   MPI_Bcast(m_Surf_y2, m_nXGlb*m_nZSurf, MPIQPrecision, right_psi, m_comm);
   MPI_Bcast(m_SurfD_y2, m_nXGlb*m_nZSurf, MPIQPrecision, right_dpsi, m_comm);

   MPI_Bcast(m_Surf_z1, m_nXGlb*m_nYSurf, MPIQPrecision, bottom_psi, m_comm);
   MPI_Bcast(m_SurfD_z1, m_nXGlb*m_nYSurf, MPIQPrecision, bottom_dpsi, m_comm);
   MPI_Bcast(m_Surf_z2, m_nXGlb*m_nYSurf, MPIQPrecision, top_psi, m_comm);
   MPI_Bcast(m_SurfD_z2, m_nXGlb*m_nYSurf, MPIQPrecision, top_dpsi, m_comm);

#pragma omp parallel default(shared) num_threads(m_nThread)
   {
      // coordinates of the centers of the virtual surface cells
      QPrecision *p=m_x;
#pragma omp for simd aligned(p:CACHE_LINE)
      for (long i=0; i<m_nXVrtl; ++i)       // x-coordinates for cells on the left, right, bottom, top surfaces
	 *(p+i)=((QPrecision) (i+m_X1Vrtl)-m_OrigX+0.5)*m_dx;      // Green's function is evaluated at the center of grids

      p=m_y;
#pragma omp for simd aligned(p:CACHE_LINE)
      for (long j=0; j<m_nYVrtl; ++j)       // y-coordinates for cells on the back, front, bottom, top surfaces
	 *(p+j)=((QPrecision) (j+m_Y1Vrtl)-m_OrigY+0.5)*m_dy;      // Green's function is evaluated at the center of grids

      p=m_z;
#pragma omp for simd aligned(p:CACHE_LINE)
      for (long k=0; k<m_nZVrtl; ++k)       // z-coordinates for cells on the back, front, left, right surfaces
	 *(p+k)=((QPrecision) (k+m_Z1Vrtl)-m_OrigZ+0.5)*m_dz;      // Green's function is evaluated at the center of grids

      // the k indices
      p=m_kx;
#pragma omp for simd aligned(p:CACHE_LINE)
      for (long i=0; i<=(m_nXGlb>>1); ++i)
	 p[i]=TWOPI/m_dx*(QPrecision) i/(QPrecision) m_nXGlb;
#pragma omp for simd aligned(p:CACHE_LINE)
      for (long i=(m_nXGlb>>1)+1; i<m_nXGlb; ++i)
	 p[i]=TWOPI/m_dx*((QPrecision) i/(QPrecision) m_nXGlb - 1.0);

      p=m_ky;
#pragma omp for simd aligned(p:CACHE_LINE)
      for (long j=0; j<=(m_nYGlb>>1); ++j)
	 p[j]=TWOPI/m_dy*(QPrecision) j/(QPrecision) m_nYGlb;
#pragma omp for simd aligned(p:CACHE_LINE)
      for (long j=(m_nYGlb>>1)+1; j<m_nYGlb; ++j)
	 p[j]=TWOPI/m_dy*((QPrecision) j/(QPrecision) m_nYGlb - 1.0);

      p=m_kz;
#pragma omp for simd aligned(p:CACHE_LINE)
      for (long k=0; k<=(m_nZGlb>>1); ++k)
	 p[k]=TWOPI/m_dz*(QPrecision) k/(QPrecision) m_nZGlb;
#pragma omp for simd aligned(p:CACHE_LINE)
      for (long k=(m_nZGlb>>1)+1; k<m_nZGlb; ++k)
	 p[k]=TWOPI/m_dz*((QPrecision) k/(QPrecision) m_nZGlb - 1.0);

      QPrecision *pcos, *psin;
      QPrecision offset=m_OrigX*m_dx;
      pcos=m_coskxx; psin=m_sinkxx;
#pragma omp for
      for (long i=0; i<m_nXVrtl; ++i)
#pragma omp simd aligned(pcos:CACHE_LINE) aligned(psin:CACHE_LINE)
	 for (long ki=0; ki<m_nXGlb; ++ki) {
	    *(pcos+i*m_nXkxx+ki)=COS(m_kx[ki]*(m_x[i]+offset));
	    *(psin+i*m_nXkxx+ki)=SIN(m_kx[ki]*(m_x[i]+offset));
	 }

      offset=m_OrigY*m_dy;
      pcos=m_coskyy; psin=m_sinkyy;
#pragma omp for
      for (long j=0; j<m_nYVrtl; ++j)
#pragma omp simd aligned(pcos:CACHE_LINE) aligned(psin:CACHE_LINE)
	 for (long kj=0; kj<m_nYGlb; ++kj) {
	    *(pcos+j*m_nYkyy+kj)=COS(m_ky[kj]*(m_y[j]+offset));
	    *(psin+j*m_nYkyy+kj)=SIN(m_ky[kj]*(m_y[j]+offset));
	 }

      offset=m_OrigZ*m_dz;
      pcos=m_coskzz; psin=m_sinkzz;
#pragma omp for
      for (long k=0; k<m_nZVrtl; ++k)
#pragma omp simd aligned(pcos:CACHE_LINE) aligned(psin:CACHE_LINE)
	 for (long kk=0; kk<m_nZGlb; ++kk) {
	    *(pcos+k*m_nZkzz+kk)=COS(m_kz[kk]*(m_z[k]+offset));
	    *(psin+k*m_nZkzz+kk)=SIN(m_kz[kk]*(m_z[k]+offset));
	 }
   }

   // load-balance indices
   // the back + the front surfaces
   long total_cell=2*m_nYVrtl*m_nZVrtl;
   long bin_cell=total_cell/m_nProcs;
   long residual_cell=total_cell%m_nProcs;
   m_iStart=(m_Rank<residual_cell)?((bin_cell+1)*m_Rank):(bin_cell*m_Rank+residual_cell);
   m_iEnd=(m_Rank<residual_cell)?(m_iStart+bin_cell):(m_iStart+bin_cell-1);

   // the left + the right surfaces
   total_cell=2*m_nXVrtl*m_nZVrtl;
   bin_cell=total_cell/m_nProcs;
   residual_cell=total_cell%m_nProcs;
   m_jStart=(m_Rank<residual_cell)?((bin_cell+1)*m_Rank):(bin_cell*m_Rank+residual_cell);
   m_jEnd=(m_Rank<residual_cell)?(m_jStart+bin_cell):(m_jStart+bin_cell-1);

   // the bottom + the top surfaces
   total_cell=2*m_nXVrtl*m_nYVrtl;
   bin_cell=total_cell/m_nProcs;
   residual_cell=total_cell%m_nProcs;
   m_kStart=(m_Rank<residual_cell)?((bin_cell+1)*m_Rank):(bin_cell*m_Rank+residual_cell);
   m_kEnd=(m_Rank<residual_cell)?(m_kStart+bin_cell):(m_kStart+bin_cell-1);
}

void CNear2Fresnel::ConvertR2K(QPrecision* &Surf, long l0, long l1, long ndim1,
      long a1Vrtl, long a2Vrtl, long a1ABC, long a2ABC,
      long b1Vrtl, long b2Vrtl, long b1ABC, long b2ABC)
{
   MKLComplex *mFFT=NULL;
   long n0=l0;
   long n1=l1;
   int val=Init_Aligned_MKL_Matrix_2D<MKLComplex>(mFFT, n0, n1, sizeof(QPrecision)*2);
   if (val) {
      char msg[80];
      sprintf(msg, "mFFT of dimensions: %ld, %ld", n0, n1);
      HandleError(val, msg);
   }

   MKL_LONG length[2];
   MKL_LONG strides[3];
   DFTI_DESCRIPTOR *desc;

   length[0]=l0;	length[1]=l1;
   strides[0]=0; strides[1]=n1; strides[2]=1;
   DftiCreateDescriptor(&desc, DFTIPrecision, DFTI_COMPLEX, 2, length);
   DftiSetValue(desc,DFTI_INPUT_STRIDES,strides);
   DftiSetValue(desc,DFTI_FORWARD_SCALE,1.0);
   DftiCommitDescriptor(desc);

#pragma omp parallel for num_threads(m_nThread)
   for (long i=0; i<l0; ++i) {
      if (i<=a1ABC || i>=a2ABC) {
#pragma omp simd aligned(mFFT:CACHE_LINE)
	 for (long j=0; j<l1; ++j) {
	    (mFFT+i*n1+j)->real=0.0;
	    (mFFT+i*n1+j)->imag=0.0;
	 }
      } else {
	 QPrecision c=1.0; // correction factor to damp the two ends
	 if (i<a1Vrtl) c=weight((QPrecision) (i-(a1ABC+1))/(QPrecision) ((a1Vrtl-1)-(a1ABC+1)));
	 else if (i>a2Vrtl) c=weight((QPrecision) ((a2ABC-1)-i)/(QPrecision) ((a2ABC-1)-(a2Vrtl+1)));
#pragma omp simd aligned(mFFT:CACHE_LINE) aligned(Surf:CACHE_LINE)
	 for (long j=0; j<l1; ++j) {
	    if (j<=b1ABC || j>=b2ABC) {
	       (mFFT+i*n1+j)->real=0.0;
	       (mFFT+i*n1+j)->imag=0.0;
	    } else {
	       QPrecision d=c;
	       if (j<b1Vrtl) d*=weight((QPrecision) (j-(b1ABC+1))/(QPrecision) ((b1Vrtl-1)-(b1ABC+1)));
	       else if (j>b2Vrtl) d*=weight((QPrecision) ((b2ABC-1)-j)/(QPrecision) ((b2ABC-1)-(b2Vrtl+1)));
	       (mFFT+i*n1+j)->real = *(Surf+i*ndim1+j*2)*d;
	       (mFFT+i*n1+j)->imag = *(Surf+i*ndim1+j*2+1)*d;
	    }
	 }
      }
   }
   DftiComputeForward(desc,mFFT);
#pragma omp parallel for num_threads(m_nThread)
   for (long i=0; i<l0; ++i)
#pragma omp simd aligned(mFFT:CACHE_LINE) aligned(Surf:CACHE_LINE)
      for (long j=0; j<l1; ++j) {
	 *(Surf+i*ndim1+j*2)=(mFFT+i*n1+j)->real;
	 *(Surf+i*ndim1+j*2+1)=(mFFT+i*n1+j)->imag;
      }

   DftiFreeDescriptor(&desc);
   Free_Aligned_MKL_Matrix_2D<MKLComplex>(mFFT, n0, n1);
}

void CNear2Fresnel::Psi(QPrecision* r, QPrecision* psi, QPrecision* SphericalWaveFactor)
{
   // QPrecision xbar=r[0]/m_lambdabar0;
   // QPrecision ybar=r[1]/m_lambdabar0;
   // QPrecision zbar=r[2]/m_lambdabar0;
   QPrecision xbar=r[0]; // use m_lambdabar0 as length unit
   QPrecision ybar=r[1];
   QPrecision zbar=r[2];
   long i, j, k;
   QPrecision R0=SQRT(xbar*xbar+ybar*ybar+zbar*zbar);
   SphericalWaveFactor[0]=COS(R0)/R0;
   SphericalWaveFactor[1]=SIN(R0)/R0;

   QPrecision subpsi[2] = {0.0, 0.0};	// domain sum of psi
   QPrecision spsi_r, spsi_i;		// surface sum of psi
   QPrecision cpsi_r, cpsi_i;		// psi of a single surface cell
   
   // back surface
   spsi_r=spsi_i=0.0;
   for (i=m_iStart; i<m_nYVrtl*m_nZVrtl && i<=m_iEnd; ++i) { // loop ignored if m_iStart>=m_nYVrtl*m_nZVrtl
      j=i/m_nZVrtl;
      k=i%m_nZVrtl;
      BackSurfContribution(j, k, xbar, ybar, zbar, R0, cpsi_r, cpsi_i);
      spsi_r+=cpsi_r;
      spsi_i+=cpsi_i;
   }
   subpsi[0] += m_cfx*spsi_r;
   subpsi[1] += m_cfx*spsi_i;

   // front surface
   spsi_r=spsi_i=0.0;
   for (; i<=m_iEnd; ++i) { // loop ignored if i from the previous loop > m_iEnd
      j=i/m_nZVrtl-m_nYVrtl;
      k=i%m_nZVrtl;
      FrontSurfContribution(j, k, xbar, ybar, zbar, R0, cpsi_r, cpsi_i);
      spsi_r+=cpsi_r;
      spsi_i+=cpsi_i;
   }
   subpsi[0] += m_cfx*spsi_r;
   subpsi[1] += m_cfx*spsi_i;

   // left surface
   spsi_r=spsi_i=0.0;
   for (j=m_jStart; j<m_nXVrtl*m_nZVrtl && j<=m_jEnd; ++j) {
      i=j/m_nZVrtl;
      k=j%m_nZVrtl;
      LeftSurfContribution(i, k, xbar, ybar, zbar, R0, cpsi_r, cpsi_i);
      spsi_r+=cpsi_r;
      spsi_i+=cpsi_i;
   }
   subpsi[0] += m_cfy*spsi_r;
   subpsi[1] += m_cfy*spsi_i;

   // right surface
   spsi_r=spsi_i=0.0;
   for (; j<=m_jEnd; ++j) {
      i=j/m_nZVrtl-m_nXVrtl;
      k=j%m_nZVrtl;
      RightSurfContribution(i, k, xbar, ybar, zbar, R0, cpsi_r, cpsi_i);
      spsi_r+=cpsi_r;
      spsi_i+=cpsi_i;
   }
   subpsi[0] += m_cfy*spsi_r;
   subpsi[1] += m_cfy*spsi_i;

   // bottom surface
   spsi_r=spsi_i=0.0;
   for (k=m_kStart; k<m_nXVrtl*m_nYVrtl && k<=m_kEnd; ++k) {
      i=k/m_nYVrtl;
      j=k%m_nYVrtl;
      BottomSurfContribution(i, j, xbar, ybar, zbar, R0, cpsi_r, cpsi_i);
      spsi_r+=cpsi_r;
      spsi_i+=cpsi_i;
   }
   subpsi[0] += m_cfz*spsi_r;
   subpsi[1] += m_cfz*spsi_i;

   // top surface
   spsi_r=spsi_i=0.0;
   for (; k<=m_kEnd; ++k) {
      i=k/m_nYVrtl-m_nXVrtl;
      j=k%m_nYVrtl;
      TopSurfContribution(i, j, xbar, ybar, zbar, R0, cpsi_r, cpsi_i);
      spsi_r+=cpsi_r;
      spsi_i+=cpsi_i;
   }
   subpsi[0] += m_cfz*spsi_r;
   subpsi[1] += m_cfz*spsi_i;

   MPI_Barrier(m_comm);
   MPI_Allreduce(subpsi, psi, 2, MPIQPrecision, MPI_SUM, m_comm);

   // use m_lambdabar0 as length unit.  so comment out the following 2 lines.
   // psi[0]/=m_lambdabar0;
   // psi[1]/=m_lambdabar0;
}

void CNear2Fresnel::BackSurfContribution(long j, long k, QPrecision xbar,
      QPrecision ybar, QPrecision zbar, QPrecision R0,
      QPrecision &cpsi_r, QPrecision &cpsi_i)
{
   QPrecision Rx=xbar-m_x1v;
   QPrecision Ry=ybar-m_y[j];
   QPrecision Rz=zbar-m_z[k];
   QPrecision R=SQRT(Rx*Rx+Ry*Ry+Rz*Rz);
   QPrecision Rxh=Rx/R;
   QPrecision Ryh=Ry/R;
   QPrecision Rzh=Rz/R;
   QPrecision coefr=-Rxh/R;
   QPrecision coefi=Rxh;
   QPrecision *ky=m_ky, *kz=m_kz;
   QPrecision *rsincy=m_rsincy, *isincy=m_isincy, *rsincz=m_rsincz, *isincz=m_isincz;
   QPrecision sum1r, sum1i;

#pragma omp parallel default(shared) num_threads(m_nThread)
   {
      QPrecision *cosk=m_coskyy+j*m_nYkyy;
      QPrecision *sink=m_sinkyy+j*m_nYkyy;
#pragma omp for simd aligned(ky:CACHE_LINE) aligned(rsincy:CACHE_LINE) aligned(isincy:CACHE_LINE) aligned(cosk:CACHE_LINE) aligned(sink:CACHE_LINE)
      for (long jj=0; jj<m_nYGlb; ++jj) {
	 QPrecision tmp=(ky[jj]-Ryh)*m_dy2;
	 if (tmp<TINY && tmp>-TINY)
	    tmp=1.0;
	 else
	    tmp=SIN(tmp)/tmp;
	 rsincy[jj]=tmp*cosk[jj];
	 isincy[jj]=tmp*sink[jj];
      }

      cosk=m_coskzz+k*m_nZkzz;
      sink=m_sinkzz+k*m_nZkzz;
#pragma omp for simd aligned(kz:CACHE_LINE) aligned(rsincz:CACHE_LINE) aligned(isincz:CACHE_LINE) aligned(cosk:CACHE_LINE) aligned(sink:CACHE_LINE)
      for (long kk=0; kk<m_nZGlb; ++kk) {
	 QPrecision tmp=(kz[kk]-Rzh)*m_dz2;
	 if (tmp<TINY && tmp>-TINY)
	    tmp=1.0;
	 else
	    tmp=SIN(tmp)/tmp;
	 rsincz[kk]=tmp*cosk[kk];
	 isincz[kk]=tmp*sink[kk];
      }

      sum1r=sum1i=0.0;
#pragma omp for reduction(+:sum1r, sum1i)
      for (long jj=0; jj<m_nYGlb; ++jj) {
	 QPrecision sum2r, sum2i;
	 sum2r=sum2i=0.0;
	 QPrecision *surf=m_Surf_x1+jj*m_nZSurf;
	 QPrecision *surfd=m_SurfD_x1+jj*m_nZSurf;
#pragma omp simd aligned(surf:CACHE_LINE) aligned(surfd:CACHE_LINE) aligned(rsincz:CACHE_LINE) aligned(isincz:CACHE_LINE) reduction(+:sum2r, sum2i)
	 for (long kk=0; kk<m_nZGlb; ++kk) {
	    QPrecision tmpr=surfd[kk*2]+coefr*surf[kk*2]-coefi*surf[kk*2+1];
	    QPrecision tmpi=surfd[kk*2+1]+coefr*surf[kk*2+1]+coefi*surf[kk*2];
	    sum2r += tmpr*rsincz[kk]-tmpi*isincz[kk];
	    sum2i += tmpr*isincz[kk]+tmpi*rsincz[kk];
	 }
	 sum1r += sum2r*rsincy[jj]-sum2i*isincy[jj];
	 sum1i += sum2r*isincy[jj]+sum2i*rsincy[jj];
      }
   } /**************end of omp parallel***************/

   coefr=COS(R-R0)*R0/R;
   coefi=SIN(R-R0)*R0/R;
   cpsi_r=sum1r*coefr-sum1i*coefi;
   cpsi_i=sum1r*coefi+sum1i*coefr;
}

void CNear2Fresnel::FrontSurfContribution(long j, long k, QPrecision xbar,
      QPrecision ybar, QPrecision zbar, QPrecision R0,
      QPrecision &cpsi_r, QPrecision &cpsi_i)
{
   QPrecision Rx=xbar-m_x2v;
   QPrecision Ry=ybar-m_y[j];
   QPrecision Rz=zbar-m_z[k];
   QPrecision R=SQRT(Rx*Rx+Ry*Ry+Rz*Rz);
   QPrecision Rxh=Rx/R;
   QPrecision Ryh=Ry/R;
   QPrecision Rzh=Rz/R;
   QPrecision coefr=-Rxh/R;
   QPrecision coefi=Rxh;
   QPrecision *ky=m_ky, *kz=m_kz;
   QPrecision *rsincy=m_rsincy, *isincy=m_isincy, *rsincz=m_rsincz, *isincz=m_isincz;
   QPrecision sum1r, sum1i;

#pragma omp parallel default(shared) num_threads(m_nThread)
   {
      QPrecision *cosk=m_coskyy+j*m_nYkyy;
      QPrecision *sink=m_sinkyy+j*m_nYkyy;
#pragma omp for simd aligned(ky:CACHE_LINE) aligned(rsincy:CACHE_LINE) aligned(isincy:CACHE_LINE) aligned(cosk:CACHE_LINE) aligned(sink:CACHE_LINE)
      for (long jj=0; jj<m_nYGlb; ++jj) {
	 QPrecision tmp=(ky[jj]-Ryh)*m_dy2;
	 if (tmp<TINY && tmp>-TINY)
	    tmp=1.0;
	 else
	    tmp=SIN(tmp)/tmp;
	 rsincy[jj]=tmp*cosk[jj];
	 isincy[jj]=tmp*sink[jj];
      }

      cosk=m_coskzz+k*m_nZkzz;
      sink=m_sinkzz+k*m_nZkzz;
#pragma omp for simd aligned(kz:CACHE_LINE) aligned(rsincz:CACHE_LINE) aligned(isincz:CACHE_LINE) aligned(cosk:CACHE_LINE) aligned(sink:CACHE_LINE)
      for (long kk=0; kk<m_nZGlb; ++kk) {
	 QPrecision tmp=(kz[kk]-Rzh)*m_dz2;
	 if (tmp<TINY && tmp>-TINY)
	    tmp=1.0;
	 else
	    tmp=SIN(tmp)/tmp;
	 rsincz[kk]=tmp*cosk[kk];
	 isincz[kk]=tmp*sink[kk];
      }

      sum1r=sum1i=0.0;
#pragma omp for reduction(+:sum1r, sum1i)
      for (long jj=0; jj<m_nYGlb; ++jj) {
	 QPrecision sum2r, sum2i;
	 sum2r=sum2i=0.0;
	 QPrecision *surf=m_Surf_x2+jj*m_nZSurf;
	 QPrecision *surfd=m_SurfD_x2+jj*m_nZSurf;
#pragma omp simd aligned(surf:CACHE_LINE) aligned(surfd:CACHE_LINE) aligned(rsincz:CACHE_LINE) aligned(isincz:CACHE_LINE) reduction(+:sum2r, sum2i)
	 for (long kk=0; kk<m_nZGlb; ++kk) {
	    QPrecision tmpr=surfd[kk*2]+coefr*surf[kk*2]-coefi*surf[kk*2+1];
	    QPrecision tmpi=surfd[kk*2+1]+coefr*surf[kk*2+1]+coefi*surf[kk*2];
	    sum2r += tmpr*rsincz[kk]-tmpi*isincz[kk];
	    sum2i += tmpr*isincz[kk]+tmpi*rsincz[kk];
	 }
	 sum1r += sum2r*rsincy[jj]-sum2i*isincy[jj];
	 sum1i += sum2r*isincy[jj]+sum2i*rsincy[jj];
      }
   } /**************end of omp parallel***************/

   coefr=-COS(R-R0)*R0/R;
   coefi=-SIN(R-R0)*R0/R;
   cpsi_r=sum1r*coefr-sum1i*coefi;
   cpsi_i=sum1r*coefi+sum1i*coefr;
}

void CNear2Fresnel::LeftSurfContribution(long i, long k, QPrecision xbar,
      QPrecision ybar, QPrecision zbar, QPrecision R0,
      QPrecision &cpsi_r, QPrecision &cpsi_i)
{
   QPrecision Rx=xbar-m_x[i];
   QPrecision Ry=ybar-m_y1v;
   QPrecision Rz=zbar-m_z[k];
   QPrecision R=SQRT(Rx*Rx+Ry*Ry+Rz*Rz);
   QPrecision Rxh=Rx/R;
   QPrecision Ryh=Ry/R;
   QPrecision Rzh=Rz/R;
   QPrecision coefr=-Ryh/R;
   QPrecision coefi=Ryh;
   QPrecision *kx=m_kx, *kz=m_kz;
   QPrecision *rsincx=m_rsincx, *isincx=m_isincx, *rsincz=m_rsincz, *isincz=m_isincz;
   QPrecision sum1r, sum1i;

#pragma omp parallel default(shared) num_threads(m_nThread)
   {
      QPrecision *cosk=m_coskxx+i*m_nXkxx;
      QPrecision *sink=m_sinkxx+i*m_nXkxx;
#pragma omp for simd aligned(kx:CACHE_LINE) aligned(rsincx:CACHE_LINE) aligned(isincx:CACHE_LINE) aligned(cosk:CACHE_LINE) aligned(sink:CACHE_LINE)
      for (long ii=0; ii<m_nXGlb; ++ii) {
	 QPrecision tmp=(kx[ii]-Rxh)*m_dx2;
	 if (tmp<TINY && tmp>-TINY)
	    tmp=1.0;
	 else
	    tmp=SIN(tmp)/tmp;
	 rsincx[ii]=tmp*cosk[ii];
	 isincx[ii]=tmp*sink[ii];
      }

      cosk=m_coskzz+k*m_nZkzz;
      sink=m_sinkzz+k*m_nZkzz;
#pragma omp for simd aligned(kz:CACHE_LINE) aligned(rsincz:CACHE_LINE) aligned(isincz:CACHE_LINE) aligned(cosk:CACHE_LINE) aligned(sink:CACHE_LINE)
      for (long kk=0; kk<m_nZGlb; ++kk) {
	 QPrecision tmp=(kz[kk]-Rzh)*m_dz2;
	 if (tmp<TINY && tmp>-TINY)
	    tmp=1.0;
	 else
	    tmp=SIN(tmp)/tmp;
	 rsincz[kk]=tmp*cosk[kk];
	 isincz[kk]=tmp*sink[kk];
      }

      sum1r=sum1i=0.0;
#pragma omp for reduction(+:sum1r, sum1i)
      for (long ii=0; ii<m_nXGlb; ++ii) {
	 QPrecision sum2r, sum2i;
	 sum2r=sum2i=0.0;
	 QPrecision *surf=m_Surf_y1+ii*m_nZSurf;
	 QPrecision *surfd=m_SurfD_y1+ii*m_nZSurf;
#pragma omp simd aligned(surf:CACHE_LINE) aligned(surfd:CACHE_LINE) aligned(rsincz:CACHE_LINE) aligned(isincz:CACHE_LINE) reduction(+:sum2r, sum2i)
	 for (long kk=0; kk<m_nZGlb; ++kk) {
	    QPrecision tmpr=surfd[kk*2]+coefr*surf[kk*2]-coefi*surf[kk*2+1];
	    QPrecision tmpi=surfd[kk*2+1]+coefr*surf[kk*2+1]+coefi*surf[kk*2];
	    sum2r += tmpr*rsincz[kk]-tmpi*isincz[kk];
	    sum2i += tmpr*isincz[kk]+tmpi*rsincz[kk];
	 }
	 sum1r += sum2r*rsincx[ii]-sum2i*isincx[ii];
	 sum1i += sum2r*isincx[ii]+sum2i*rsincx[ii];
      }
   } /**************end of omp parallel***************/

   coefr=COS(R-R0)*R0/R;
   coefi=SIN(R-R0)*R0/R;
   cpsi_r=sum1r*coefr-sum1i*coefi;
   cpsi_i=sum1r*coefi+sum1i*coefr;
}

void CNear2Fresnel::RightSurfContribution(long i, long k, QPrecision xbar,
      QPrecision ybar, QPrecision zbar, QPrecision R0,
      QPrecision &cpsi_r, QPrecision &cpsi_i)
{
   QPrecision Rx=xbar-m_x[i];
   QPrecision Ry=ybar-m_y2v;
   QPrecision Rz=zbar-m_z[k];
   QPrecision R=SQRT(Rx*Rx+Ry*Ry+Rz*Rz);
   QPrecision Rxh=Rx/R;
   QPrecision Ryh=Ry/R;
   QPrecision Rzh=Rz/R;
   QPrecision coefr=-Ryh/R;
   QPrecision coefi=Ryh;
   QPrecision *kx=m_kx, *kz=m_kz;
   QPrecision *rsincx=m_rsincx, *isincx=m_isincx, *rsincz=m_rsincz, *isincz=m_isincz;
   QPrecision sum1r, sum1i;

#pragma omp parallel default(shared) num_threads(m_nThread)
   {
      QPrecision *cosk=m_coskxx+i*m_nXkxx;
      QPrecision *sink=m_sinkxx+i*m_nXkxx;
#pragma omp for simd aligned(kx:CACHE_LINE) aligned(rsincx:CACHE_LINE) aligned(isincx:CACHE_LINE) aligned(cosk:CACHE_LINE) aligned(sink:CACHE_LINE)
      for (long ii=0; ii<m_nXGlb; ++ii) {
	 QPrecision tmp=(kx[ii]-Rxh)*m_dx2;
	 if (tmp<TINY && tmp>-TINY)
	    tmp=1.0;
	 else
	    tmp=SIN(tmp)/tmp;
	 rsincx[ii]=tmp*cosk[ii];
	 isincx[ii]=tmp*sink[ii];
      }

      cosk=m_coskzz+k*m_nZkzz;
      sink=m_sinkzz+k*m_nZkzz;
#pragma omp for simd aligned(kz:CACHE_LINE) aligned(rsincz:CACHE_LINE) aligned(isincz:CACHE_LINE) aligned(cosk:CACHE_LINE) aligned(sink:CACHE_LINE)
      for (long kk=0; kk<m_nZGlb; ++kk) {
	 QPrecision tmp=(kz[kk]-Rzh)*m_dz2;
	 if (tmp<TINY && tmp>-TINY)
	    tmp=1.0;
	 else
	    tmp=SIN(tmp)/tmp;
	 rsincz[kk]=tmp*cosk[kk];
	 isincz[kk]=tmp*sink[kk];
      }

      sum1r=sum1i=0.0;
#pragma omp for reduction(+:sum1r, sum1i)
      for (long ii=0; ii<m_nXGlb; ++ii) {
	 QPrecision sum2r, sum2i;
	 sum2r=sum2i=0.0;
	 QPrecision *surf=m_Surf_y2+ii*m_nZSurf;
	 QPrecision *surfd=m_SurfD_y2+ii*m_nZSurf;
#pragma omp simd aligned(surf:CACHE_LINE) aligned(surfd:CACHE_LINE) aligned(rsincz:CACHE_LINE) aligned(isincz:CACHE_LINE) reduction(+:sum2r, sum2i)
	 for (long kk=0; kk<m_nZGlb; ++kk) {
	    QPrecision tmpr=surfd[kk*2]+coefr*surf[kk*2]-coefi*surf[kk*2+1];
	    QPrecision tmpi=surfd[kk*2+1]+coefr*surf[kk*2+1]+coefi*surf[kk*2];
	    sum2r += tmpr*rsincz[kk]-tmpi*isincz[kk];
	    sum2i += tmpr*isincz[kk]+tmpi*rsincz[kk];
	 }
	 sum1r += sum2r*rsincx[ii]-sum2i*isincx[ii];
	 sum1i += sum2r*isincx[ii]+sum2i*rsincx[ii];
      }
   } /**************end of omp parallel***************/

   coefr=-COS(R-R0)*R0/R;
   coefi=-SIN(R-R0)*R0/R;
   cpsi_r=sum1r*coefr-sum1i*coefi;
   cpsi_i=sum1r*coefi+sum1i*coefr;
}

void CNear2Fresnel::BottomSurfContribution(long i, long j, QPrecision xbar,
      QPrecision ybar, QPrecision zbar, QPrecision R0,
      QPrecision &cpsi_r, QPrecision &cpsi_i)
{
   QPrecision Rx=xbar-m_x[i];
   QPrecision Ry=ybar-m_y[j];
   QPrecision Rz=zbar-m_z1v;
   QPrecision R=SQRT(Rx*Rx+Ry*Ry+Rz*Rz);
   QPrecision Rxh=Rx/R;
   QPrecision Ryh=Ry/R;
   QPrecision Rzh=Rz/R;
   QPrecision coefr=-Rzh/R;
   QPrecision coefi=Rzh;
   QPrecision *kx=m_kx, *ky=m_ky;
   QPrecision *rsincx=m_rsincx, *isincx=m_isincx, *rsincy=m_rsincy, *isincy=m_isincy;
   QPrecision sum1r, sum1i;

#pragma omp parallel default(shared) num_threads(m_nThread)
   {
      QPrecision *cosk=m_coskxx+i*m_nXkxx;
      QPrecision *sink=m_sinkxx+i*m_nXkxx;
#pragma omp for simd aligned(kx:CACHE_LINE) aligned(rsincx:CACHE_LINE) aligned(isincx:CACHE_LINE) aligned(cosk:CACHE_LINE) aligned(sink:CACHE_LINE)
      for (long ii=0; ii<m_nXGlb; ++ii) {
	 QPrecision tmp=(kx[ii]-Rxh)*m_dx2;
	 if (tmp<TINY && tmp>-TINY)
	    tmp=1.0;
	 else
	    tmp=SIN(tmp)/tmp;
	 rsincx[ii]=tmp*cosk[ii];
	 isincx[ii]=tmp*sink[ii];
      }

      cosk=m_coskyy+j*m_nYkyy;
      sink=m_sinkyy+j*m_nYkyy;
#pragma omp for simd aligned(ky:CACHE_LINE) aligned(rsincy:CACHE_LINE) aligned(isincy:CACHE_LINE) aligned(cosk:CACHE_LINE) aligned(sink:CACHE_LINE)
      for (long jj=0; jj<m_nYGlb; ++jj) {
	 QPrecision tmp=(ky[jj]-Ryh)*m_dy2;
	 if (tmp<TINY && tmp>-TINY)
	    tmp=1.0;
	 else
	    tmp=SIN(tmp)/tmp;
	 rsincy[jj]=tmp*cosk[jj];
	 isincy[jj]=tmp*sink[jj];
      }

      sum1r=sum1i=0.0;
#pragma omp for reduction(+:sum1r, sum1i)
      for (long ii=0; ii<m_nXGlb; ++ii) {
	 QPrecision sum2r, sum2i;
	 sum2r=sum2i=0.0;
	 QPrecision *surf=m_Surf_z1+ii*m_nYSurf;
	 QPrecision *surfd=m_SurfD_z1+ii*m_nYSurf;
#pragma omp simd aligned(surf:CACHE_LINE) aligned(surfd:CACHE_LINE) aligned(rsincy:CACHE_LINE) aligned(isincy:CACHE_LINE) reduction(+:sum2r, sum2i)
	 for (long jj=0; jj<m_nYGlb; ++jj) {
	    QPrecision tmpr=surfd[jj*2]+coefr*surf[jj*2]-coefi*surf[jj*2+1];
	    QPrecision tmpi=surfd[jj*2+1]+coefr*surf[jj*2+1]+coefi*surf[jj*2];
	    sum2r += tmpr*rsincy[jj]-tmpi*isincy[jj];
	    sum2i += tmpr*isincy[jj]+tmpi*rsincy[jj];
	 }
	 sum1r += sum2r*rsincx[ii]-sum2i*isincx[ii];
	 sum1i += sum2r*isincx[ii]+sum2i*rsincx[ii];
      }
   } /**************end of omp parallel***************/

   coefr=COS(R-R0)*R0/R;
   coefi=SIN(R-R0)*R0/R;
   cpsi_r=sum1r*coefr-sum1i*coefi;
   cpsi_i=sum1r*coefi+sum1i*coefr;
}

void CNear2Fresnel::TopSurfContribution(long i, long j, QPrecision xbar,
      QPrecision ybar, QPrecision zbar, QPrecision R0,
      QPrecision &cpsi_r, QPrecision &cpsi_i)
{
   QPrecision Rx=xbar-m_x[i];
   QPrecision Ry=ybar-m_y[j];
   QPrecision Rz=zbar-m_z2v;
   QPrecision R=SQRT(Rx*Rx+Ry*Ry+Rz*Rz);
   QPrecision Rxh=Rx/R;
   QPrecision Ryh=Ry/R;
   QPrecision Rzh=Rz/R;
   QPrecision coefr=-Rzh/R;
   QPrecision coefi=Rzh;
   QPrecision *kx=m_kx, *ky=m_ky;
   QPrecision *rsincx=m_rsincx, *isincx=m_isincx, *rsincy=m_rsincy, *isincy=m_isincy;
   QPrecision sum1r, sum1i;

#pragma omp parallel default(shared) num_threads(m_nThread)
   {
      QPrecision *cosk=m_coskxx+i*m_nXkxx;
      QPrecision *sink=m_sinkxx+i*m_nXkxx;
#pragma omp for simd aligned(kx:CACHE_LINE) aligned(rsincx:CACHE_LINE) aligned(isincx:CACHE_LINE) aligned(cosk:CACHE_LINE) aligned(sink:CACHE_LINE)
      for (long ii=0; ii<m_nXGlb; ++ii) {
	 QPrecision tmp=(kx[ii]-Rxh)*m_dx2;
	 if (tmp<TINY && tmp>-TINY)
	    tmp=1.0;
	 else
	    tmp=SIN(tmp)/tmp;
	 rsincx[ii]=tmp*cosk[ii];
	 isincx[ii]=tmp*sink[ii];
      }

      cosk=m_coskyy+j*m_nYkyy;
      sink=m_sinkyy+j*m_nYkyy;
#pragma omp for simd aligned(ky:CACHE_LINE) aligned(rsincy:CACHE_LINE) aligned(isincy:CACHE_LINE) aligned(cosk:CACHE_LINE) aligned(sink:CACHE_LINE)
      for (long jj=0; jj<m_nYGlb; ++jj) {
	 QPrecision tmp=(ky[jj]-Ryh)*m_dy2;
	 if (tmp<TINY && tmp>-TINY)
	    tmp=1.0;
	 else
	    tmp=SIN(tmp)/tmp;
	 rsincy[jj]=tmp*cosk[jj];
	 isincy[jj]=tmp*sink[jj];
      }

      sum1r=sum1i=0.0;
#pragma omp for reduction(+:sum1r, sum1i)
      for (long ii=0; ii<m_nXGlb; ++ii) {
	 QPrecision sum2r, sum2i;
	 sum2r=sum2i=0.0;
	 QPrecision *surf=m_Surf_z2+ii*m_nYSurf;
	 QPrecision *surfd=m_SurfD_z2+ii*m_nYSurf;
#pragma omp simd aligned(surf:CACHE_LINE) aligned(surfd:CACHE_LINE) aligned(rsincy:CACHE_LINE) aligned(isincy:CACHE_LINE) reduction(+:sum2r, sum2i)
	 for (long jj=0; jj<m_nYGlb; ++jj) {
	    QPrecision tmpr=surfd[jj*2]+coefr*surf[jj*2]-coefi*surf[jj*2+1];
	    QPrecision tmpi=surfd[jj*2+1]+coefr*surf[jj*2+1]+coefi*surf[jj*2];
	    sum2r += tmpr*rsincy[jj]-tmpi*isincy[jj];
	    sum2i += tmpr*isincy[jj]+tmpi*rsincy[jj];
	 }
	 sum1r += sum2r*rsincx[ii]-sum2i*isincx[ii];
	 sum1i += sum2r*isincx[ii]+sum2i*rsincx[ii];
      }
   } /**************end of omp parallel***************/

   coefr=-COS(R-R0)*R0/R;
   coefi=-SIN(R-R0)*R0/R;
   cpsi_r=sum1r*coefr-sum1i*coefi;
   cpsi_i=sum1r*coefi+sum1i*coefr;
}

void CNear2Fresnel::HandleError(int val, string id_str)
{
   stringstream sst;
   sst << "Error " << val << "(Rank " << m_Rank <<
      "): cannot allocate " << id_str;
   throw runtime_error(sst.str());
}

void CNear2Fresnel::SaveSurfTerm(std::string savename)
{
   stringstream sstr;
   sstr << savename  << ".dat";
   MPI_File fh;
   MPI_Offset disp;
   if (MPI_File_open(MPI_COMM_WORLD, sstr.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh)!=MPI_SUCCESS)
      throw runtime_error("Cannot open output file " + sstr.str());

   if (!m_Rank) {
      int precision=sizeof(QPrecision);
      MPI_File_write(fh, &precision, 1, MPIInt, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nXGlb, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nYGlb, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nZGlb, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_X1Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_X2Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_Y1Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_Y2Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_Z1Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_Z2Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nABC, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_dx, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_dy, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_dz, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_OrigX, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_OrigY, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_OrigZ, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_E0, 1, MPIQPrecision, MPI_STATUS_IGNORE);
   }

   disp=sizeof(int)+sizeof(m_nXGlb)+sizeof(m_nYGlb)+sizeof(m_nZGlb)
         +sizeof(m_X1Vrtl)+sizeof(m_X2Vrtl)+sizeof(m_Y1Vrtl)+sizeof(m_Y2Vrtl)
         +sizeof(m_Z1Vrtl)+sizeof(m_Z2Vrtl)+sizeof(m_nABC)+sizeof(m_dx)
         +sizeof(m_dy)+sizeof(m_dz)+sizeof(m_OrigX)+sizeof(m_OrigY)
         +sizeof(m_OrigZ)+sizeof(m_E0);

   int back_psi=0;
   int back_dpsi=1%m_nProcs;
   int front_psi=2%m_nProcs;
   int front_dpsi=3%m_nProcs;
   int left_psi=4%m_nProcs;
   int left_dpsi=5%m_nProcs;
   int right_psi=6%m_nProcs;
   int right_dpsi=7%m_nProcs;
   int bottom_psi=8%m_nProcs;
   int bottom_dpsi=9%m_nProcs;
   int top_psi=10%m_nProcs;
   int top_dpsi=11%m_nProcs;

   MPI_Datatype newtype;
   int sizes[2], subsizes[2], starts[2];

   long incre=m_nYGlb*m_nZGlb*2*sizeof(QPrecision);
   sizes[0]=m_nYGlb; sizes[1]=m_nZSurf;
   subsizes[0]=m_nYGlb; subsizes[1]=m_nZGlb*2;
   starts[0]=0; starts[1]=0;
   MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPIQPrecision, &newtype);
   MPI_Type_commit(&newtype);
   // back surface psi data
   if (m_Rank==back_psi)
      MPI_File_write_at(fh, disp, m_Surf_x1, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // back surface dpsi data
   if (m_Rank==back_dpsi)
      MPI_File_write_at(fh, disp, m_SurfD_x1, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // front surface psi data
   if (m_Rank==front_psi)
      MPI_File_write_at(fh, disp, m_Surf_x2, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // front surface dpsi data
   if (m_Rank==front_dpsi)
      MPI_File_write_at(fh, disp, m_SurfD_x2, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;
   MPI_Type_free(&newtype);

   incre=m_nXGlb*m_nZGlb*2*sizeof(QPrecision);
   sizes[0]=m_nXGlb; sizes[1]=m_nZSurf;
   subsizes[0]=m_nXGlb; subsizes[1]=m_nZGlb*2;
   starts[0]=0; starts[1]=0;
   MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPIQPrecision, &newtype);
   MPI_Type_commit(&newtype);
   // left surface psi data
   if (m_Rank==left_psi)
      MPI_File_write_at(fh, disp, m_Surf_y1, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // left surface dpsi data
   if (m_Rank==left_dpsi)
      MPI_File_write_at(fh, disp, m_SurfD_y1, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // right surface psi data
   if (m_Rank==right_psi)
      MPI_File_write_at(fh, disp, m_Surf_y2, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // right surface dpsi data
   if (m_Rank==right_dpsi)
      MPI_File_write_at(fh, disp, m_SurfD_y2, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;
   MPI_Type_free(&newtype);

   incre=m_nXGlb*m_nYGlb*2*sizeof(QPrecision);
   sizes[0]=m_nXGlb; sizes[1]=m_nYSurf;
   subsizes[0]=m_nXGlb; subsizes[1]=m_nYGlb*2;
   starts[0]=0; starts[1]=0;
   MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPIQPrecision, &newtype);
   MPI_Type_commit(&newtype);
   // bottom surface psi data
   if (m_Rank==bottom_psi)
      MPI_File_write_at(fh, disp, m_Surf_z1, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // bottom surface dpsi data
   if (m_Rank==bottom_dpsi)
      MPI_File_write_at(fh, disp, m_SurfD_z1, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // top surface psi data
   if (m_Rank==top_psi)
      MPI_File_write_at(fh, disp, m_Surf_z2, 1, newtype, MPI_STATUS_IGNORE);
   disp += incre;

   // top surface dpsi data
   if (m_Rank==top_dpsi)
      MPI_File_write_at(fh, disp, m_SurfD_z2, 1, newtype, MPI_STATUS_IGNORE);
   MPI_Type_free(&newtype);

   MPI_File_close(&fh);
}

CNear2Fresnel::~CNear2Fresnel()
{
   Free_Aligned_Matrix_2D<QPrecision>(m_Surf_x1, m_nYGlb, m_nZSurf);
   Free_Aligned_Matrix_2D<QPrecision>(m_Surf_x2, m_nYGlb, m_nZSurf);
   Free_Aligned_Matrix_2D<QPrecision>(m_Surf_y1, m_nXGlb, m_nZSurf);
   Free_Aligned_Matrix_2D<QPrecision>(m_Surf_y2, m_nXGlb, m_nZSurf);
   Free_Aligned_Matrix_2D<QPrecision>(m_Surf_z1, m_nXGlb, m_nYSurf);
   Free_Aligned_Matrix_2D<QPrecision>(m_Surf_z2, m_nXGlb, m_nYSurf);

   Free_Aligned_Matrix_2D<QPrecision>(m_SurfD_x1, m_nYGlb, m_nZSurf);
   Free_Aligned_Matrix_2D<QPrecision>(m_SurfD_x2, m_nYGlb, m_nZSurf);
   Free_Aligned_Matrix_2D<QPrecision>(m_SurfD_y1, m_nXGlb, m_nZSurf);
   Free_Aligned_Matrix_2D<QPrecision>(m_SurfD_y2, m_nXGlb, m_nZSurf);
   Free_Aligned_Matrix_2D<QPrecision>(m_SurfD_z1, m_nXGlb, m_nYSurf);
   Free_Aligned_Matrix_2D<QPrecision>(m_SurfD_z2, m_nXGlb, m_nYSurf);

   Free_Aligned_Vector<QPrecision>(m_x, m_nXVrtl);
   Free_Aligned_Vector<QPrecision>(m_y, m_nYVrtl);
   Free_Aligned_Vector<QPrecision>(m_z, m_nZVrtl);

   Free_Aligned_Vector<QPrecision>(m_kx, m_nXGlb);
   Free_Aligned_Vector<QPrecision>(m_ky, m_nYGlb);
   Free_Aligned_Vector<QPrecision>(m_kz, m_nZGlb);

   Free_Aligned_Matrix_2D<QPrecision>(m_coskxx, m_nXVrtl, m_nXkxx);
   Free_Aligned_Matrix_2D<QPrecision>(m_sinkxx, m_nXVrtl, m_nXkxx);
   Free_Aligned_Matrix_2D<QPrecision>(m_coskyy, m_nYVrtl, m_nYkyy);
   Free_Aligned_Matrix_2D<QPrecision>(m_sinkyy, m_nYVrtl, m_nYkyy);
   Free_Aligned_Matrix_2D<QPrecision>(m_coskzz, m_nZVrtl, m_nZkzz);
   Free_Aligned_Matrix_2D<QPrecision>(m_sinkzz, m_nZVrtl, m_nZkzz);
   
   Free_Aligned_Vector<QPrecision>(m_rsincx, m_nXGlb);
   Free_Aligned_Vector<QPrecision>(m_isincx, m_nXGlb);
   Free_Aligned_Vector<QPrecision>(m_rsincy, m_nYGlb);
   Free_Aligned_Vector<QPrecision>(m_isincy, m_nYGlb);
   Free_Aligned_Vector<QPrecision>(m_rsincz, m_nZGlb);
   Free_Aligned_Vector<QPrecision>(m_isincz, m_nZGlb);
}
