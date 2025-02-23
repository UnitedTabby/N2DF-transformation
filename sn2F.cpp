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
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <string.h>
#include <mpi.h>
#include "run_environment.h"
#include "utility.h"
#include "n2Fresnel.h"

using namespace std;

void show_usage(char *app)
{
   cout << "Usage: " << app << " -i FILE\nwhere, FILE: input file" << endl;
}

int main(int argc, char *argv[])
{
   string filename;

   for (int i=1; i<argc; ++i) {
      string arg=argv[i];
      if ((arg=="-h") || (arg=="--help")) {
	 show_usage(argv[0]);
	 return 0;
      }
      if (arg=="-i") {
	 if (i<argc-1)
	    filename=argv[i+1];
	 if (i==argc-1 || filename[0]=='-') {
	    cerr << "Error: -i requires an input" << endl;
	    show_usage(argv[0]);
	    return 1;
	 }
      }
   }

   if (filename.empty()) {
      cerr << "Error: mandatory -i FILE" << endl;
      show_usage(argv[0]);
      return 1;
   }

   MPI_Init(&argc, &argv);

   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//   QPrecision R,Eula_phi,Eula_theta;
//   int num_Eula_psi;
   QPrecision R,Eula_alpha,Eula_beta;
   int num_Eula_gamma;

   CNear2Fresnel n2F(MPI_COMM_WORLD);

   try {
      string surfname;
      if (!rank) {
	 string aline;
	 stringstream sstr;
	 ifstream ifs(filename.c_str());
	 if (!ifs)
	    throw runtime_error("Cannot open file " + filename);

	 if (getaline(ifs, aline))
	    surfname=aline;
	 else {
	    ifs.close();
	    throw runtime_error("Incomplete input file " + filename);
	 }

	 if (getaline(ifs, aline)) {
	    sstr.str(aline);
	    sstr >> R;
	 } else {
	    ifs.close();
	    throw runtime_error("Incomplete input file " + filename);
	 }

	 if (getaline(ifs, aline)) {
	    sstr.clear();
	    sstr.str(aline);
//	    sstr >> Eula_phi >> Eula_theta ;
//	    Eula_phi *= PI/180.0;
//	    Eula_theta *= PI/180.0;
	    sstr >> Eula_alpha >> Eula_beta ;
	    Eula_alpha *= PI/180.0;
	    Eula_beta *= PI/180.0;
	 } else {
	    ifs.close();
	    throw runtime_error("Incomplete input file " + filename);
	 }

	 if (getaline(ifs, aline)) {
	    sstr.clear();
	    sstr.str(aline);
//	    sstr >> num_Eula_psi;
	    sstr >> num_Eula_gamma;
	 } else {
	    ifs.close();
	    throw runtime_error("Incomplete input file " + filename);
	 }

	 ifs.close();
      }
   
      int cnt;
      if (!rank) cnt=surfname.length();
      MPI_Bcast(&cnt, 1, MPIInt, 0, MPI_COMM_WORLD);
      char *str=new char[cnt+1];
      str[cnt]='\0';
      if (!rank) strcpy(str, surfname.c_str());
      MPI_Bcast(str, cnt, MPI_CHAR, 0, MPI_COMM_WORLD);
      if (rank) surfname=str;
      delete[] str;
      MPI_Bcast(&R, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
//      MPI_Bcast(&Eula_phi, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
//      MPI_Bcast(&Eula_theta, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
//      MPI_Bcast(&num_Eula_psi, 1, MPIInt, 0, MPI_COMM_WORLD);
      MPI_Bcast(&Eula_alpha, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
      MPI_Bcast(&Eula_beta, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
      MPI_Bcast(&num_Eula_gamma, 1, MPIInt, 0, MPI_COMM_WORLD);
      n2F.InitData(surfname);
      MPI_Barrier(MPI_COMM_WORLD);
   } catch (exception const &e) {
      cerr << e.what() << endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
   }

   QPrecision r[3];
   QPrecision psi[2];
   QPrecision SphericalWaveFactor[2];

/*	   
   for (int iEula_psi=0; iEula_psi<num_Eula_psi; ++iEula_psi) {
      // cross-polarization plane
      QPrecision Eula_psi=(QPrecision) iEula_psi*PI/180.0;
      // co-polarization plane
      //QPrecision Eula_psi=(QPrecision) (iEula_psi-90)*PI/180.0;
      // along_diagonal (Eula_phi=135, Eula_theta=90)
      //QPrecision Eula_psi=(QPrecision) (iEula_psi-90)*PI/180.0+asin(1.0/sqrt(3.0));
      r[0]=R*(COS(Eula_phi)*COS(Eula_psi)-SIN(Eula_phi)*COS(Eula_theta)*SIN(Eula_psi));
      r[1]=R*(SIN(Eula_phi)*COS(Eula_psi)+COS(Eula_phi)*COS(Eula_theta)*SIN(Eula_psi));
      r[2]=R*SIN(Eula_theta)*SIN(Eula_psi);
      n2F.Psi(r, psi, SphericalWaveFactor);
      if (!rank) {
	 cout << resetiosflags(ios::scientific) << iEula_psi-90 << "\t" \
	    << setiosflags(ios::scientific) << setprecision(16) \
	    << "(" << r[0] << "," << r[1] << "," << r[2] << ")\t(" \
	    << psi[0] << "," << psi[1] << ")\t(" << SphericalWaveFactor[0] \
	    << "," << SphericalWaveFactor[1] << ")" << endl;
      }
   }
 */
   for (int iEula_gamma=0; iEula_gamma<num_Eula_gamma; ++iEula_gamma) {
      // cross-polarization plane (Eula_alpha=0, Eula_beta=0)
      // co-polarization plane (Eula_alpha=0, Eula_beta=90)
      QPrecision Eula_gamma=(QPrecision) iEula_gamma*PI/180.0;
      // along_diagonal (Eula_alpha=45, Eula_beta=90)
      //QPrecision Eula_gamma=(QPrecision) iEula_gamma*PI/180.0+asin(1.0/sqrt(3.0));
      r[0]=R*(-SIN(Eula_alpha)*SIN(Eula_gamma)+COS(Eula_alpha)*COS(Eula_beta)*COS(Eula_gamma));
      r[1]=R*(COS(Eula_alpha)*SIN(Eula_gamma)+SIN(Eula_alpha)*COS(Eula_beta)*COS(Eula_gamma));
      r[2]=-R*SIN(Eula_beta)*COS(Eula_gamma);
      n2F.Psi(r, psi, SphericalWaveFactor);
      if (!rank) {
	 cout << resetiosflags(ios::scientific) << iEula_gamma << "\t" \
	    << setiosflags(ios::scientific) << setprecision(16) \
	    << "(" << r[0] << "," << r[1] << "," << r[2] << ")\t(" \
	    << psi[0] << "," << psi[1] << ")\t(" << SphericalWaveFactor[0] \
	    << "," << SphericalWaveFactor[1] << ")" << endl;
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();
}
