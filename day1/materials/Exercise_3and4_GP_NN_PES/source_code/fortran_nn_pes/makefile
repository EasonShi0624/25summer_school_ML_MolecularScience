## 

FC = ifort
FFLAGS = -132 -O2
LDFLAGS = -L/work1/soft/intel/mkl/10.1.2.024/lib/em64t -lmkl -lmkl_lapack -lmkl_em64t -lguide -limf -lm
OBJ = main.o oh3pes.o

##

a.out:      $(OBJ)
	$(FC) $(FFLAGS) $(OBJ) $(LDFLAGS) -o a.out

clean:
	rm -f *.mod *.exe *.o fort.* a.out
