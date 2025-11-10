
FC = ifort
FFLAGS = -O1 -traceback

LDFLAGS = -mkl

#LDFLAGS = -L/work1/private/chenjun/soft/intel/mkl/lib/intel64 -lmkl_rt -Wl,-rpath /work1/private/chenjun/soft/intel/mkl/lib/intel64

objs = 	main.o gp_prediction.o \
		covMaternard.o covMaterniso.o \
		meanZero.o meanOne.o meanConst.o meanLinear.o \
		infExact.o \
		likGauss_prediction.o \
		sq_dist.o

a.exe:	$(objs)
	$(FC) $(FFLAGS) $+ -o $@ $(LDFLAGS)

%.o:	%.f90
	$(FC) $(FFLAGS) -c $< -o $@

clean:
	rm -f *.o *.mod *.exe

run:	a.exe
	./a.exe
