# oh3_nn

This PES is published in:
    Jun Chen, Xin Xu, Xin Xu and Dong H. Zhang*
    A global potential energy surface for the H2 + OH ↔ H2O + H reaction using neural networks
    J. Chem. Phys., 2013, 138(15):154301. 

1. pes
    energy_oh3.F
        Source code of the oh3-cxz PES.
        How to use: call spotdriva(n,r,e)
            n: number of geometries
            r(6,n): distances/bohr
            e(n):   energies/hartree
        !! r and e in single precision -------------------------Important-------
        !! energy_oh3.F should complied with -save option ------Important-------
        The first line, #define pesdir, defines the directory containing following txt files.

    n13-l.txt      n22-r.txt     n40-2w.txt     nh2o_2_6.txt
        Parameters of NN function

2. test
    test.f90     Makefile
        A test program for evaluating abstraction and exchange intrinsic reaction paths.

3. abinitio-data
    oh3-abinitio.txt
        16814 geometries and energies used for NN fitting of OH3-CXZ PES
        Column 1~6: distances of O-H1, O-H2, O-H3, H1-H2, H1-H3, H2-H3 (Bohr)
        Column 7:   potential energy (eV)
        Calculated at ROHF-UCCSD(T)-F12a/aug-cc-pVTZ level using Molpro 2010.1
        The energies are relative to the asymptitic valley of OH + H2

    OH.POT
        Potential energy curve of OH from UCCSD(T)-F12a/AVTZ
        Re=1.834037920 Bohr
           0.970531068 Angstrom
        E0=-75.67000441 Hartree

    H2.POT
        Potential energy curve of H2 from UCCSD(T)-F12a/AVTZ
        Re= 1.402004353 Bohr
            0.741908750 Angstrom
        E0= -1.17440249 Hartree
