c=====================================================================
c pes: select which pes to use, NN1 NN2 or NN3
c icount: number of geometries
c rstore(6,icount): distances of O H H H, Bohr
c output enout(icount): potential energy in Hartree, V(OH+H2)=0.D0
      subroutine oh3pes(pes,icount,rstore,enout)
      implicit none
      real*8 rstore(6,1),enout(1)
      real*8 rin(6),v,dv(6)
      integer icount,i,dx
      character*3 pes

c     v22=76.84440690d0 ! uccsd(t)-f12a/avtz  oh+h2
c     v13=76.86995525d0 ! uccsd(t)-f12a/avtz  h+h2o

      do i=1,icount
        rin(:)=rstore(:,i)
        select case(pes)
          case default
          call nninter1(rin,v,0,dv)
          case ('NN1')
          call nninter1(rin,v,0,dv)
          case ('nn1')
          call nninter1(rin,v,0,dv)
          case ('NN2')
          call nninter2(rin,v,0,dv)
          case ('nn2')
          call nninter2(rin,v,0,dv)
          case ('NN3')
          call nninter3(rin,v,0,dv)
          case ('nn3')
          call nninter3(rin,v,0,dv)
        end select
        enout(i)=v
      enddo

      return
      end

c=====================================================================
      module nnsim
      implicit none

      real*8,parameter :: pi=dacos(-1.d0)
      interface nsim
              module procedure nsim_1 ! one hidden layer
              module procedure nsim_2 ! two hidden layers
      end interface

      contains
      subroutine nsim_1(rin,vx,ndim,neu,nxin,nxw1,nxb1,nxw2,nxb2,nxout,
     &dx,dv)
      implicit none
      integer :: ndim,neu,i,j,dx
      real*8 :: rin(ndim),vrange,rrange(ndim),dv(ndim)
      real*8 :: r(ndim),vx,nxin(2,ndim),nxout(2),nxb2,rtmp
      real*8 :: nxb1(neu),nxw1(ndim,neu),nxw2(neu),ax(neu)

      vx=0.d0
      dv=0.d0
      r=rin
      vrange=nxout(2)-nxout(1)
      ! mapminmax [-1,1]
      do i=1,ndim
        rrange(i)=nxin(2,i)-nxin(1,i)
        r(i)=2.d0*(r(i)-nxin(1,i))/rrange(i)-1.d0
      end do
      ! 1st layer
      do i=1,neu
        rtmp=nxb1(i)
        do j=1,ndim
          rtmp=rtmp+nxw1(j,i)*r(j)
        end do
        ax(i)=tanh(rtmp)
      end do

      ! output layer
      vx=nxb2
      do i=1,neu
        vx=vx+nxw2(i)*ax(i)
      end do
      !reverse map
      vx=vrange*(vx+1.d0)/2+nxout(1)

      if (dx.eq.1) then
      ! calculate first derivatives, dv/dr(i), dv(i)
        do i=1,ndim
          do j=1,neu
            dv(i)=dv(i)+nxw2(j)*nxw1(i,j)*(1-ax(j)**2)
          enddo
          dv(i)=dv(i)*vrange/rrange(i)
        enddo
      endif

      return
      end subroutine nsim_1

      subroutine nsim_2(rin,vx,ndim,neu1,neu2,trfcn,nxin,nxw1,nxb1,
     &nxw2,nxb2,nxw3,nxb3,nxout,dx,dv)
      implicit none
      integer :: ndim,neu1,neu2,i,j,k,dx
      real*8 :: r(ndim),rin(ndim),vx,nxin(2,ndim),nxout(2)
      real*8 :: nxw1(ndim,neu1),nxb1(neu1),nxw2(neu1,neu2),nxb2(neu2)
      real*8 :: nxw3(neu2),nxb3,ax(neu1),bx(neu2)
      real*8 :: dv(ndim),dvtm,vrange,rrange(ndim)
      character*8 :: trfcn(2)

      real*8 :: rtmp,rt1(neu1),rt2(neu2)
      real*8,external :: ddot

      vx=0.d0
      dv=0.d0
      r=rin
      vrange=nxout(2)-nxout(1)
      ! map to [-1,1]
      do i=1,ndim
        rrange(i)=nxin(2,i)-nxin(1,i)
        r(i)=2.0*(r(i)-nxin(1,i))/rrange(i)-1.0
      end do

      ! 1st layer
      rt1=nxb1
      call dgemv('t',ndim,neu1,1.d0,nxw1,ndim,r,1,1.d0,rt1,1)
      do j=1,neu1
        ax(j)=dtanh(rt1(j))
      end do

      ! 2nd layer
      rt2=nxb2
      call dgemv('t',neu1,neu2,1.d0,nxw2,neu1,ax,1,1.d0,rt2,1)
      do k=1,neu2
        bx(k)=dtanh(rt2(k))
      end do

      ! output layer
      vx=nxb3+ddot(neu2,nxw3,1,bx,1)

      !reverse map
      vx=vrange*(vx+1.d0)/2+nxout(1)

      if (dx.eq.1) then
      ! calculate first derivatives, dv/dr(i), dv(i)
        do i=1,ndim
          do k=1,neu2
            dvtm=0.d0
            do j=1,neu1
              dvtm=dvtm+nxw2(j,k)*nxw1(i,j)*(1-ax(j)**2)
            enddo
            dv(i)=dv(i)+nxw3(k)*dvtm*(1-bx(k)**2)
          enddo
          dv(i)=dv(i)*vrange/rrange(i)
        enddo
      endif

      return
      end subroutine nsim_2

c     potoh, o-h uccsd(t)-f12a/avtz energy in ev
      function potoh(r)
      real*8 r,potoh
      real*8 x,w1(5),b1(5),w2(5),b2,ra,rb,va,vb
      integer i
      data w1,b1,w2,b2,ra,rb,va,vb/
     &   6.4673602067509854e+000,
     &  -1.3623917123812161e+000,
     &   1.9692891975248517e+000,
     &  -3.0842317137227271e+000,
     &   2.8753386118410593e+000,
     &  -6.2995066766419541e+000,
     &   1.1338119520445564e-001,
     &   7.3524071060910179e-001,
     &  -1.9242209086024227e+000,
     &   4.6341234161417368e+000,
     &   4.4487144796737021e-003,
     &  -7.5921384799742464e-001,
     &   4.4369462093719220e-001,
     &  -8.7516096362305562e-002,
     &  -4.6326649659746060e+001,
     &   4.5791134903018026e+001,
     &   1.2500000000000000e+000,
     &   4.1000000000000000e+000,
     &   2.5300000000000002e-004,
     &   5.2943569999999998e+000/
      save w1,b1,w2,b2,ra,rb,va,vb

      potoh=0.d0
      x=2*(r-ra)/(rb-ra)-1
      do i=1,5
        potoh=potoh+w2(i)*tanh(b1(i)+w1(i)*x)
      enddo
      potoh=potoh+b2
      potoh=(potoh+1)*(vb-va)/2+va
      return
      end function potoh

c     pothh, h-h uccsd(t)-f12a/avtz energy in ev
      function pothh(r)
      real*8 r,pothh
      real*8 x,w1(5),b1(5),w2(5),b2,ra,rb,va,vb
      integer i
      data w1,b1,w2,b2,ra,rb,va,vb/
     &   1.6264981845926245e+000,
     &  -3.6036352665856217e+000,
     &  -1.4262429547486353e+000,
     &  -4.5389257076092262e+000,
     &  -1.0370365421774828e+001,
     &  -5.9327868896216229e-001,
     &   6.9006605801506959e-001,
     &  -9.0839635069711533e-001,
     &  -5.9790603462261309e+000,
     &  -1.3007124917554380e+001,
     &  -6.3219690048134394e-002,
     &   6.3788100794878316e-003,
     &  -1.0432656434523606e+000,
     &   2.0101618557956517e+001,
     &   4.0421850398756270e+001,
     &   5.9403385445541353e+001,
     &   6.0000000000000000e-001,
     &   6.1000000000000000e+000,
     &   1.8000000000000000e-005,
     &   1.1043358000000000e+001/
      save w1,b1,w2,b2,ra,rb,va,vb

      pothh=0.d0
      x=2*(r-ra)/(rb-ra)-1
      do i=1,5
        pothh=pothh+w2(i)*tanh(b1(i)+w1(i)*x)
      enddo
      pothh=pothh+b2
      pothh=(pothh+1)*(vb-va)/2+va
      return
      end function pothh

      end module nnsim

c=====================================================================
      module nnparam1

      implicit none
      save

      integer,parameter :: bondidx(4,4)=[-1,1,2,3,1,-1,4,5,
     &                                    2,4,-1,6,3,5,6,-1]
      integer,parameter :: ndim=6
      character*99 :: pesdir='./'
      character*99 :: n22f='NN1-22.txt'
      character*99 :: n40f='NN1-40.txt'
      character*99 :: n13f='NN1-13.txt'
      integer,parameter :: n22s1=25,n22s2=25
      integer,parameter :: n40s1=50,n40s2=50
      integer,parameter :: n13s1=20,n13s2=20

      real*8 :: n22w1(ndim,n22s1),n22b1(n22s1)
      real*8 :: n22w2(n22s1,n22s2),n22b2(n22s2)
      real*8 :: n22w3(n22s2),n22b3
      real*8 :: n22in(2,ndim),n22out(2)
      character*8 :: n22fcn(2)

      real*8 :: n40w1(ndim,n40s1),n40b1(n40s1)
      real*8 :: n40w2(n40s1,n40s2),n40b2(n40s2)
      real*8 :: n40w3(n40s2),n40b3
      real*8 :: n40in(2,ndim),n40out(2)
      character*8 :: n40fcn(2)

      real*8 :: n13w1(ndim,n13s1),n13b1(n13s1)
      real*8 :: n13w2(n13s1,n13s2),n13b2(n13s2)
      real*8 :: n13w3(n13s2),n13b3
      real*8 :: n13in(2,ndim),n13out(2)
      character*8 :: n13fcn(2)

      contains
      subroutine nninit
      implicit none
      integer :: i,j,k

      open(100,file=trim(pesdir)//trim(n22f),status='old',action='read')
      read(100,*)n22w1,n22b1,n22w2,n22b2,n22w3,n22b3,n22in,n22out,n22fcn
      close(100)

      open(100,file=trim(pesdir)//trim(n40f),status='old',action='read')
      read(100,*)n40w1,n40b1,n40w2,n40b2,n40w3,n40b3,n40in,n40out,n40fcn
      close(100)

      open(100,file=trim(pesdir)//trim(n13f),status='old',action='read')
      read(100,*)n13w1,n13b1,n13w2,n13b2,n13w3,n13b3,n13in,n13out,n13fcn
      close(100)

      n22out=n22out/27.2116d0
      n40out=n40out/27.2116d0
      n13out=n13out/27.2116d0

      end subroutine nninit
      end module nnparam1

c---------------------------------------------------------------------

      subroutine nninter1(rr0,vx,dx,dv)
c  dx=0 calculate energy only
c  dx=1 calculate 1st derivative
      use nnsim
      use nnparam1
      implicit none
      real*8 :: rr0(ndim),vx,dv(ndim)
      integer :: i,j,k,itmp,idx,dx
      integer :: iloc(1),sortidx(4)
      real*8 :: rr(ndim), rtmp,vtmp,df
      real*8 :: v13,v22,v40
      real*8 :: dv13(ndim),dv22(ndim),dv40(ndim)
      real*8 :: f13,f22,f40,fss
      real*8 :: z(6)
      real*8 :: logsig,x
      integer :: flag0,flag23,flag12
      integer,save :: init=0
      logsig(x)=1.d0/(1.d0+dexp(-x))
      if (init.eq.0) then
        init=1
        call nninit()
      end if
      flag0=0;flag23=0;flag12=0

c bond length permutation
c input: rr0(1:6)=r(oh1),r(oh2),r(oh3),r(h1h2),r(h1h3),r(h2h3)
c permute h1,h2,h3 to oh1.le.oh2.le.oh3
      sortidx(1)=1
      iloc=minloc(rr0(1:3))
      sortidx(2)=iloc(1) +1
      j=3
      do i=2,4
        if(i == sortidx(2)) cycle
        sortidx(j)=i
        j=j+1
      end do
      if(rr0(sortidx(3)-1) > rr0(sortidx(4)-1)) then
        itmp=sortidx(3)
        sortidx(3)=sortidx(4)
        sortidx(4)=itmp
      end if
      k=0
      do i=1,3
        do j=i+1,4
          k=k+1
          rr(k)=rr0( bondidx( sortidx(i),sortidx(j) ) )
        end do
      end do
c end bond length permutation

990   continue
      
      f22=logsig(50.d0*(rr(2)-3.5d0))
      f13=logsig(50.d0*(rr(3)-6.4d0))*(1.d0-f22)
      f40=1.d0-f22-f13
 
      v13=0.d0
      v22=0.d0
      v40=0.d0
      
      fss=1.0e-8
      
      if(f22.gt.fss) call nsim(rr,v22,ndim,n22s1,n22s2,n22fcn,n22in,
     &n22w1,n22b1,n22w2,n22b2,n22w3,n22b3,n22out,dx,dv22)
      if(f13.gt.fss) call nsim(rr,v13,ndim,n13s1,n13s2,n13fcn,n13in,
     &n13w1,n13b1,n13w2,n13b2,n13w3,n13b3,n13out,dx,dv13)
      if(f40.gt.fss) call nsim(rr,v40,ndim,n40s1,n40s2,n40fcn,n40in,
     &n40w1,n40b1,n40w2,n40b2,n40w3,n40b3,n40out,dx,dv40)
      
      vx=f22*v22+f13*v13+f40*v40
c     return

      if(flag0.eq.1) goto 991

      if (rr(3)-rr(2).lt.0.1.and.rr(3).ge.rr(2).and.flag23.eq.0) then
         rtmp=rr(3);rr(3)=rr(2);rr(2)=rtmp;
         rtmp=rr(4);rr(4)=rr(5);rr(5)=rtmp;
         vtmp=vx
         flag23=1
         goto 990
      endif

      if (flag23.eq.1) then
         df=0.5*(1+sin((rr(2)-rr(3))*5*pi))
         vx=vx*(1-df)+vtmp*df
         rtmp=rr(3);rr(3)=rr(2);rr(2)=rtmp;
         rtmp=rr(4);rr(4)=rr(5);rr(5)=rtmp;
      endif
991   flag0=1

      if (rr(2)-rr(1).lt.0.1.and.rr(2).ge.rr(1).and.flag12.eq.0) then
         rtmp=rr(1);rr(1)=rr(2);rr(2)=rtmp;
         rtmp=rr(6);rr(6)=rr(5);rr(5)=rtmp;
         vtmp=vx
         flag12=1
         goto 990
      endif

      if (flag12.eq.1) then
         df=0.5*(1+sin((rr(1)-rr(2))*5*pi))
         vx=vx*(1-df)+vtmp*df
         rtmp=rr(1);rr(1)=rr(2);rr(2)=rtmp;
         rtmp=rr(6);rr(6)=rr(5);rr(5)=rtmp;
      endif

      return
      end subroutine nninter1

c=====================================================================
      module nnparam2
      implicit none
      save
      integer,parameter :: bondidx(4,4)=[-1,1,2,3,1,-1,4,5,
     &                                    2,4,-1,6,3,5,6,-1]
      integer,parameter :: ndim=6

      character*99 :: pesdir='./'
      character*99 :: n01f='NN2.txt'
      integer,parameter :: n01s1=50,n01s2=80

      double precision :: n01w1(ndim,n01s1),n01b1(n01s1)
      double precision :: n01w2(n01s1,n01s2),n01b2(n01s2)
      double precision :: n01w3(n01s2),n01b3
      double precision :: n01in(2,ndim),n01out(2)
      character*8 :: n01fcn(2)

      contains
      subroutine nninit
      open(100,file=trim(pesdir)//trim(n01f),status='old',action='read')
      read(100,*)n01w1,n01b1,n01w2,n01b2,n01w3,n01b3,n01in,n01out,n01fcn
      close(100)
      n01out=n01out/27.2116d0
      end subroutine nninit

      end module nnparam2

c---------------------------------------------------------------------
      subroutine nninter2(rr0,vx,dx,dv)
!  dx=0 calculate energy only
!  dx=1 calculate 1st derivative
      use nnsim
      use nnparam2
      implicit none
      real*8 :: rr0(ndim),vx,dv(ndim)
      integer :: i,j,k,itmp,idx,dx
      integer :: iloc(1),sortidx(4)
      real*8 :: rr(ndim), rtmp,vtmp,df
      integer :: flag0,flag23,flag12
      integer,save :: init=0

      if (init.eq.0) then
        init=1
        call nninit()
      end if

      flag0=0;flag23=0;flag12=0

      sortidx(1)=1 
  
      iloc=minloc(rr0(1:3))
      sortidx(2)=iloc(1) +1
      j=3
      do i=2,4
        if(i == sortidx(2)) cycle
        sortidx(j)=i
        j=j+1
      end do
      if(rr0(sortidx(3)-1) > rr0(sortidx(4)-1)) then
        itmp=sortidx(3)
        sortidx(3)=sortidx(4)
        sortidx(4)=itmp
      end if
  
      k=0
      do i=1,3
        do j=i+1,4
          k=k+1
          rr(k)=rr0( bondidx( sortidx(i),sortidx(j) ) )
        end do
      end do

c 2+2 asy
      if (rr(2).gt.20.d0) then
        vx=potoh(rr(1))+pothh(rr(6))
        vx=vx/27.2116d0
        return
      endif

990   continue

      if ( rr(1).ge.3.0 ) then
! o----h , sort to rr(6)<=min(rr(5),rr(4))
        if ( rr(4).le.rr(6) ) then
        rtmp=rr(1);rr(1)=rr(3);rr(3)=rtmp;
        rtmp=rr(4);rr(4)=rr(6);rr(6)=rtmp;
        end if
        if ( rr(5).le.rr(6) ) then
        rtmp=rr(1);rr(1)=rr(2);rr(2)=rtmp;
        rtmp=rr(5);rr(5)=rr(6);rr(6)=rtmp;
        end if
        call nsim(rr,vx,ndim,n01s1,n01s2,n01fcn,n01in,n01w1,n01b1,
     &n01w2,n01b2,n01w3,n01b3,n01out,dx,dv)
        vx=max(vx,2.5)
        return
      elseif ( rr(2).gt.3.5.and.rr(6).gt.2.8) then
! h----h
        call nsim(rr,vx,ndim,n01s1,n01s2,n01fcn,n01in,n01w1,n01b1,
     &n01w2,n01b2,n01w3,n01b3,n01out,dx,dv) 
        vx=max(vx,2.5)
        return
      else
        call nsim(rr,vx,ndim,n01s1,n01s2,n01fcn,n01in,n01w1,n01b1,
     &n01w2,n01b2,n01w3,n01b3,n01out,dx,dv)
      endif

      if(flag0.eq.1) goto 991

      if (rr(3)-rr(2).lt.0.1.and.rr(3).ge.rr(2).and.flag23.eq.0) then
         rtmp=rr(3);rr(3)=rr(2);rr(2)=rtmp;
         rtmp=rr(4);rr(4)=rr(5);rr(5)=rtmp;
         vtmp=vx
         flag23=1
         goto 990
      endif

      if (flag23.eq.1) then
         df=0.5*(1+sin((rr(2)-rr(3))*5*pi))
         vx=vx*(1-df)+vtmp*df
         rtmp=rr(3);rr(3)=rr(2);rr(2)=rtmp;
         rtmp=rr(4);rr(4)=rr(5);rr(5)=rtmp;
      endif
991   flag0=1

      if (rr(2)-rr(1).lt.0.1.and.rr(2).ge.rr(1).and.flag12.eq.0) then
         rtmp=rr(1);rr(1)=rr(2);rr(2)=rtmp;
         rtmp=rr(6);rr(6)=rr(5);rr(5)=rtmp;
         vtmp=vx
         flag12=1
         goto 990
      endif

      if (flag12.eq.1) then
         df=0.5*(1+sin((rr(1)-rr(2))*5*pi))
         vx=vx*(1-df)+vtmp*df
         rtmp=rr(1);rr(1)=rr(2);rr(2)=rtmp;
         rtmp=rr(6);rr(6)=rr(5);rr(5)=rtmp;
      endif

      return
      end subroutine nninter2

c=====================================================================
      module nnparam3
      implicit none
      save
      integer,parameter :: bondidx(4,4)=[-1,1,2,3,1,-1,4,5,
     &                                    2,4,-1,6,3,5,6,-1]
      integer,parameter :: ndim=6

      character*99 :: pesdir='./'
      character*99 :: n01f='NN3.txt'
      integer,parameter :: n01s1=50,n01s2=80

      double precision :: n01w1(ndim,n01s1),n01b1(n01s1)
      double precision :: n01w2(n01s1,n01s2),n01b2(n01s2)
      double precision :: n01w3(n01s2),n01b3
      double precision :: n01in(2,ndim),n01out(2)
      character*8 :: n01fcn(2)

      contains
      subroutine nninit
      open(100,file=trim(pesdir)//trim(n01f),status='old',action='read')
      read(100,*)n01w1,n01b1,n01w2,n01b2,n01w3,n01b3,n01in,n01out,n01fcn
      close(100)
      n01out=n01out/27.2116d0
      end subroutine nninit

      end module nnparam3

c---------------------------------------------------------------------
      subroutine nninter3(rr0,vx,dx,dv)
!  dx=0 calculate energy only
!  dx=1 calculate 1st derivative
      use nnsim
      use nnparam3
      implicit none
      real*8 :: rr0(ndim),vx,dv(ndim)
      integer :: i,j,k,itmp,idx,dx
      integer :: iloc(1),sortidx(4)
      real*8 :: rr(ndim), rtmp,vtmp,df
      integer :: flag0,flag23,flag12
      integer,save :: init=0

      if (init.eq.0) then
        init=1
        call nninit()
      end if

      flag0=0;flag23=0;flag12=0

      sortidx(1)=1 
  
      iloc=minloc(rr0(1:3))
      sortidx(2)=iloc(1) +1
      j=3
      do i=2,4
        if(i == sortidx(2)) cycle
        sortidx(j)=i
        j=j+1
      end do
      if(rr0(sortidx(3)-1) > rr0(sortidx(4)-1)) then
        itmp=sortidx(3)
        sortidx(3)=sortidx(4)
        sortidx(4)=itmp
      end if
  
      k=0
      do i=1,3
        do j=i+1,4
          k=k+1
          rr(k)=rr0( bondidx( sortidx(i),sortidx(j) ) )
        end do
      end do

c 2+2 asy
      if (rr(2).gt.20.d0) then
        vx=potoh(rr(1))+pothh(rr(6))
        vx=vx/27.2116d0
        return
      endif

990   continue

      if ( rr(1).ge.3.0 ) then
! o----h , sort to rr(6)<=min(rr(5),rr(4))
        if ( rr(4).le.rr(6) ) then
        rtmp=rr(1);rr(1)=rr(3);rr(3)=rtmp;
        rtmp=rr(4);rr(4)=rr(6);rr(6)=rtmp;
        end if
        if ( rr(5).le.rr(6) ) then
        rtmp=rr(1);rr(1)=rr(2);rr(2)=rtmp;
        rtmp=rr(5);rr(5)=rr(6);rr(6)=rtmp;
        end if
        call nsim(rr,vx,ndim,n01s1,n01s2,n01fcn,n01in,n01w1,n01b1,
     &n01w2,n01b2,n01w3,n01b3,n01out,dx,dv)
        vx=max(vx,2.5)
        return
      elseif ( rr(2).gt.3.5.and.rr(6).gt.2.8) then
! h----h
        call nsim(rr,vx,ndim,n01s1,n01s2,n01fcn,n01in,n01w1,n01b1,
     &n01w2,n01b2,n01w3,n01b3,n01out,dx,dv) 
        vx=max(vx,2.5)
        return
      else
        call nsim(rr,vx,ndim,n01s1,n01s2,n01fcn,n01in,n01w1,n01b1,
     &n01w2,n01b2,n01w3,n01b3,n01out,dx,dv)
      endif

      if(flag0.eq.1) goto 991

      if (rr(3)-rr(2).lt.0.1.and.rr(3).ge.rr(2).and.flag23.eq.0) then
         rtmp=rr(3);rr(3)=rr(2);rr(2)=rtmp;
         rtmp=rr(4);rr(4)=rr(5);rr(5)=rtmp;
         vtmp=vx
         flag23=1
         goto 990
      endif

      if (flag23.eq.1) then
         df=0.5*(1+sin((rr(2)-rr(3))*5*pi))
         vx=vx*(1-df)+vtmp*df
         rtmp=rr(3);rr(3)=rr(2);rr(2)=rtmp;
         rtmp=rr(4);rr(4)=rr(5);rr(5)=rtmp;
      endif
991   flag0=1

      if (rr(2)-rr(1).lt.0.1.and.rr(2).ge.rr(1).and.flag12.eq.0) then
         rtmp=rr(1);rr(1)=rr(2);rr(2)=rtmp;
         rtmp=rr(6);rr(6)=rr(5);rr(5)=rtmp;
         vtmp=vx
         flag12=1
         goto 990
      endif

      if (flag12.eq.1) then
         df=0.5*(1+sin((rr(1)-rr(2))*5*pi))
         vx=vx*(1-df)+vtmp*df
         rtmp=rr(1);rr(1)=rr(2);rr(2)=rtmp;
         rtmp=rr(6);rr(6)=rr(5);rr(5)=rtmp;
      endif

      return
      end subroutine nninter3

c=====================================================================

