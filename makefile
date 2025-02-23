MPIICPX=mpiicpx -std=c++17
N2FFLAGS=-xALDERLAKE -O3 -qopenmp -qopt-report -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread
QN2FRESNEL=n2Fresnel.cpp utility.cpp n2F.cpp
QSN2FRESNEL=n2Fresnel.cpp utility.cpp sn2F.cpp
HEADER_N2F=n2Fresnel.h run_environment.h utility.h
num_threads?=1

fn2F:	$(QN2FRESNEL) $(HEADER_N2F)
	echo =======================make fn2F=======================
	$(MPIICPX) $(N2FFLAGS) -D__USE_FLOAT__ -D__OMP_NUM_THREADS__=$(num_threads) -D$(show_init_status) -o fn2F $(QN2FRESNEL) -lm

fsn2F:	$(QSN2FRESNEL) $(HEADER_N2F)
	echo =======================make fsn2F=======================
	$(MPIICPX) $(N2FFLAGS) -D__USE_FLOAT__ -D__OMP_NUM_THREADS__=$(num_threads) -D$(show_init_status) -o fsn2F $(QSN2FRESNEL) -lm

dn2F:	$(QN2FRESNEL) $(HEADER_N2F)
	echo =======================make dn2F=======================
	$(MPIICPX) $(N2FFLAGS) -D__OMP_NUM_THREADS__=$(num_threads) -D$(show_init_status) -o dn2F $(QN2FRESNEL) -lm

dsn2F:	$(QSN2FRESNEL) $(HEADER_N2F)
	echo =======================make dsn2F=======================
	$(MPIICPX) $(N2FFLAGS) -D__OMP_NUM_THREADS__=$(num_threads) -D$(show_init_status) -o dsn2F $(QSN2FRESNEL) -lm

clean:
	rm -f fn2F fsn2F dn2F dsn2F *.optrpt *.opt.yaml
