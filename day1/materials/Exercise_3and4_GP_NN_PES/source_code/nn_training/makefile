#Macro definition

FC = ifort
# -pg

#FFLAGS = -O2 -openmp
#LDFLAGS = -L/work1/soft/intel/mkl/10.1.2.024/lib/em64t -lmkl -lmkl_lapack -lguide

#FFLAGS = -O2 -openmp -static
#LDFLAGS = -L/work1/soft/intel/mkl/10.1.2.024/lib/em64t -lmkl_em64t -lmkl_lapack -lguide

 MKLROOT = /work1/private/chenjun/soft/intel/mkl
 MKLROOT = /share/intel/2015/composer_xe_2015.6.233/mkl
 FFLAGS = -O3 -fast -openmp -static -I$(MKLROOT)/include/intel64/lp64 -I$(MKLROOT)/include
 LDFLAGS =  $(MKLROOT)/lib/intel64/libmkl_blas95_lp64.a \
            $(MKLROOT)/lib/intel64/libmkl_lapack95_lp64.a \
            -Wl,--start-group \
            $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a \
            $(MKLROOT)/lib/intel64/libmkl_core.a \
            $(MKLROOT)/lib/intel64/libmkl_intel_thread.a \
            -Wl,--end-group -lpthread -lm

 FFLAGS = -O2 -qopenmp
#MKLROOT = /work1/private/chenjun/soft/intel/mkl
#MKLROOT = /share/intel/2015/composer_xe_2015.6.233/mkl
#MKLROOT = /work1/soft/intel/compilers_and_libraries_2017.1.132/linux/mkl
#MKLROOT = /work1/soft/intel/compilers_and_libraries_2019.0.117/linux/mkl
 MKLROOT = /opt/intel/compilers_and_libraries_2019.5.281/mac/mkl
 LDFLAGS = -L$(MKLROOT)/lib/intel64 -lmkl_rt -Wl,-rpath $(MKLROOT)/lib/intel64 

 FFLAGS = -O2 -qopenmp
 LDFLAGS = -Wl,-ld_classic -mkl

#end of Macro definition

OBJ_NN = nn.o

nn.exe: $(OBJ_NN)
	${FC} ${FFLAGS} $^ -o $@ ${LDFLAGS}

clean:
	rm -f nn.exe nn.o *.mod

#rm -f *.exe *.mod *.o fort.* *.out
rm:
	rm -f M?? W??.txt

# a.out; gprof a.out gmon.out > out

