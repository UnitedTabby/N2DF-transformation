# N2DF-transformation
This software is a parallel converter to conduct the near-to-distant-field transformation.  It is built on the quantum surface equivalence theorem (QSET), which asserts the near field wavefunction and its surface normal derivative on a closed virtual surface enclosing the interaction structure contain complete information about the scattering wavefunction outside the enclosure.  The mathematical expression of QSET and the numerical technique, involving the conversion of the discrete surface data set to an analytical form, to accomplish high precision surface integration, can be found in the paper arXiv:2403.04053, https://doi.org/10.48550/arXiv.2403.04053.

This package requires the Intel OneAPI with the MPI and MKL libraries.  It has been programmed to achieve MPI-OpenMP-SIMD-vectorization hybrid parallelization.  It takes two input files.  The first one is the data file from the PSTD computation on the Schrodinger equation, which contains the values of the wavefunction and its surface normal derivative on the grids of the enclosing virtual surface.  Its filename is specified in the second input file, which itself is an entry to the execution command line.  The second file would further specify the detector's location where the wavefunction is to be calculated.  The output is the wavefunction value at the point.

The space external to the virtual enclosure is often separately referred to as the near field, the Fresnel region, and the far field, depending on the distance.  The Fresnel region sits between the near-field and the far-field, and becomes more important in high resolution imaging.  So far, one century after the birth of quantum mechanics, there has been no method to accurately calculate the scattering wavefunction in the Fresnel region.  The high precision of this code guarantees the removal of this obstacle.  This is the first time the Fresnel region is covered.

We believe the detector location is better described using the Euler-angles, including the scatterer-detector distance, the Euler-angles (alpha, beta) of the scattering plane, and the gamma angle of the detector within the plane.  Note there are two conventions of the Euler-angles.  We employ the y-convention commonly adopted by quantum mechanics to depict rotations.

An example of the second input file is provided in the package.  On a workstation of two CPUs, typical commands are (show_init_status is optional),

for compiling:
  num_threads=10 show_init_status=__INIT_STATUS__ make dsn2F 

for execution:
  mpirun -np 2 -iface lo ./dsn2F -i sn2f_exampleinput.txt

The source code is copyrighted by: Kun Chen, Shanghai Institute of Optics and
Fine Mechanics, Chinese Academy of Sciences

and it is distributed under the terms of the MIT license.

Please see LICENSE file for details.

