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
#include <ifstream>
#include <sstream>
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

   ifstream ifs;
   string aline;
   stringstream sstr;
   int n_coordinates;

   CNear2Fresnel n2F(MPI_COMM_WORLD);

   try {
      string surfname;
      if (!rank) {
	 ifs.open(filename.c_str());
	 if (!ifs)
	    throw runtime_error("Cannot open file " + filename);

	 if (getaline(ifs, aline))
	    surfname=aline;
	 else
	    throw runtime_error("Incomplete input file " + filename);

	 if (getaline(ifs, aline)) {
	    sstr.clear();
	    sstr.str(aline);
	    sstr >> n_coordinates;
	 } else
	    throw runtime_error("Incomplete input file " + filename);
      }

      int cnt;
      if (!rank) cnt=surfname_prefix.length();
      MPI_Bcast(&cnt, 1, MPIInt, 0, MPI_COMM_WORLD);
      char *str=new char[cnt+1];
      str[cnt]='\0';
      if (!rank) strcpy(str, surfname_prefix.c_str());
      MPI_Bcast(str, cnt, MPI_CHAR, 0, MPI_COMM_WORLD);
      if (rank) surfname_prefix=str;
      delete[] str;
      MPI_Bcast(&n_coordinates, 1, MPIInt, 0, MPI_COMM_WORLD);
      n2F.InitData(surfname);
   } catch (exception const &e) {
      cerr << e.what() << endl;
      if (ifs.is_open()) ifs.close();
      MPI_Abort(MPI_COMM_WORLD, 1);
   }

   MPI_Barrier(MPI_COMM_WORLD);

   QPrecision r[3];
   QPrecision psi[2];
   QPrecision SphericalWaveFactor[2];

   cout << setiosflags(ios::scientific) << setprecision(16);
   for (int i=0; i<n_coordinates; ++i) {
      if (!rank) {
	 if (getaline(ifs, aline)) {
	    sstr.clear();
	    sstr.str(aline);
	    sstr >> r[0] >> r[1] >> r[2];
	 } else {
	    break;
	 }
      }
      MPI_Bcast(r, 3, MPIQPrecision, 0, MPI_COMM_WORLD);

      n2F.Psi(r, psi, SphericalWaveFactor);
      if (!rank) {
	 cout << r[0] << "\t" << r[1] << "\t" << r[2] <<
	    "\t" << psi[0] << "\t" << psi[1] << "\t" <<
	    SphericalWaveFactor[0] << "\t" << SphericalWaveFactor[1] << endl;
      }
   }

   if (!rank) ifs.close();
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();
}
