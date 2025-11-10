
subroutine test
  implicit none
  integer,parameter :: ns=16814,ndim=6
  real*8,parameter   :: eV=27.21138505d0
  real*4 :: xs0(ndim,ns),ys0(ns),ys(ns)
  real*4 :: cv1(ns),cv2(ns)
  integer :: i,fid
  fid=100223
  open(fid,file='/home/chenjun/oh3pes/g-gaussian-process/1-training/oh3-16814.rv',status='old',action='read')
  do i=1,ns
      read(fid,*) xs0(1:ndim,i),ys0(i)
  enddo
  close(fid)

  call system("date")
  call spotdriva_gp(ns,xs0,ys,cv1,cv2); ys=ys*eV
  call system("date")
  do i=ns-3,ns
    write(*,'(10es24.8)') ys0(i),abs(ys(i)-ys0(i)),cv1(i),cv2(i)
  enddo
end subroutine

subroutine spotdriva_gp(ns,rstore,enout,cv1,cv2)
  ! number of geometries: ns
  ! input:  r in Bohr, atoms in order of O H H H
  ! output: v in Hartree, v(OH+H2)=0
  implicit none
  real*8,parameter   :: eV=27.21138505d0
  integer,parameter  :: ndim=6
  integer,intent(in) :: ns
  real*4,intent(in)  :: rstore(ndim,ns)
  real*4,intent(out) :: enout(ns)
  real*8,optional,intent(out) :: cv1(ns),cv2(ns)

  real*8 :: xs(ndim,ns),ys(ns)
  integer :: i

  ! transfer input into double precision
  xs(1:ndim,1:ns)=rstore(1:ndim,1:ns)

  ! sort according to OH bond length
  do i=1,ns
      call sort_dist_oh3(xs(1:ndim,i))
  enddo

  ! do Gaussian Process prediction
  ! note: ( ns = nperbatch ) >= 1000 is efficient
  if (present(cv1)) then
  call pes_gp(ndim,ns,xs,ys,cv1,cv2)
  else
  call pes_gp(ndim,ns,xs,ys)
  endif

  ! transfer output into single precision
  enout=ys/eV

  return
end subroutine

subroutine sort_dist_oh3(r)
  implicit none
  real*8,intent(inout)  :: r(6)
  integer i,j
  real*8 tmp
  do i=1,3
  do j=i+1,3
    if(r(i).gt.r(j)) then
      tmp=r(i)
      r(i)=r(j)
      r(j)=tmp
      tmp=r(7-i)
      r(7-i)=r(7-j)
      r(7-j)=tmp
    endif
  enddo
  enddo
  return
end subroutine sort_dist_oh3

subroutine pes_gp(n0,ns,xs0,ys,cv1,cv2)
  implicit none
  ! --- variables used for a PES using GP model --------------------------------
  integer,save :: ndim,nt,n_mean,n_cov,covFunc_d
  real*8,allocatable,save :: x(:,:),y(:),hyp_mean(:),hyp_cov(:)
  real*8,save :: hyp_lik
  character(kind=1,len=20),save :: infFunc,meanFunc,covFunc,likFunc,xhandle
  real*8,allocatable,save :: alpha(:),sW(:),L(:,:),K(:,:),T(:,:)
  real*8,save :: sn2
  integer,save :: init=0
  character(kind=1,len=80),parameter :: finput="/home/chenjun/oh3pes/g-gaussian-process/1-training/GPinput-643.txt"
  integer :: info
  ! ----------------------------------------------------------------------------
  character(kind=1,len=80) :: CC ! Comments
  integer :: i,j,fid
  ! --- variables for unknown geometries, input @ xs, output @ ymu -------------
  integer,intent(in) :: n0,ns
  real*8,intent(in)  :: xs0(n0,ns)
  real*8 :: xs(n0,ns),sumw
  real*8,intent(out) :: ys(ns)
  real*8,optional,intent(out) :: cv1(ns),cv2(ns)
  real*8,allocatable :: Ks(:,:),W(:,:)
  ! --- loading variables for gaussian process at first call -------------------
  if(init.eq.0) then
    fid=101010
    open(fid,file=trim(finput),status='old',action='read')
    read(fid,*) CC
    read(fid,*) infFunc
    read(fid,*) meanFunc
    read(fid,*) covFunc
    if(trim(covFunc).eq.'covMaternard' .or. trim(covFunc).eq.'covMaterniso') read(fid,*) covFunc_d
    read(fid,*) likFunc

    read(fid,*) CC
    read(fid,*) ndim,nt

    if(trim(meanFunc).eq.'meanZero' .or. trim(meanFunc).eq.'meanOne') then
        n_mean=0 ! no need
    elseif(trim(meanFunc).eq.'meanConst') then
        n_mean=1
    elseif(trim(meanFunc).eq.'meanLinear') then
        n_mean=ndim
    else
        print*,"Unknown meanFunc: ",trim(meanFunc)
        stop
    endif

    if(trim(covFunc).eq.'covMaternard') then
        n_cov=ndim+1
    elseif(trim(covFunc).eq.'covMaterniso') then
        n_cov=2
    else
        print*,"Unknown covFunc: ",trim(covFunc)
    endif

    allocate(x(ndim,nt),y(nt),hyp_mean(max(1,n_mean)),hyp_cov(n_cov),alpha(nt),sW(nt),L(nt,nt),K(nt,nt),T(nt,nt))

    read(fid,*) CC
    if(n_mean.gt.0) then
    do i=1,n_mean
        read(fid,*) hyp_mean(i)
    enddo
    endif

    read(fid,*) CC
    do i=1,n_cov
        read(fid,*) hyp_cov(i)
    enddo

    read(fid,*) CC
    read(fid,*) hyp_lik

    read(fid,*) CC
    read(fid,*) xhandle
    do i=1,nt
        read(fid,*) x(1:ndim,i),y(i)
    enddo

    if(trim(xhandle).eq.'normal') then
        ! do nothing
    elseif(trim(xhandle).eq.'reverse') then
        do i=1,nt
        do j=1,ndim
        x(j,i)=1.d0/x(j,i)
        enddo
        enddo
    elseif(trim(xhandle).eq.'log') then
        do i=1,nt
        do j=1,ndim
        x(j,i)=log(x(j,i))
        enddo
        enddo
    endif

    close(fid)

    if(trim(infFunc).ne.'infExact') stop "infFunc.ne.infExact"
    if(trim(covFunc).ne.'covMaternard') stop "covFunc.ne.covMaternard"
    if(trim(meanFunc).ne.'meanZero') stop "meanFunc.ne.meanZero"
    if(trim(likFunc).ne.'likGauss') stop "likFunc.ne.likGauss"

    call covMaternard(covFunc_d,hyp_cov,ndim,x,nt,x,nt,0,K)
    sn2 = exp(2.d0*hyp_lik)

    !! T = (K/sn2 + I)^-1 / sn2
    T=K/sn2
    do i=1,nt
      T(i,i)=T(i,i)+1.d0
    enddo
    call dpotrf('U',nt,T,nt,info)
    if(info.ne.0) stop "Cholesky factorization failed"

    L=0.d0
    do i=1,nt
      L(i,i)=1.d0
    enddo
    call dpotrs('U',nt,nt,T,nt,L,nt,info)
    T=L/sn2

    !! alpha=T*E
    call dsymv('U',nt,1.d0,T,nt,y,1,0.d0,alpha,1)

    !! fill T
    do i=2,nt
    do j=1,i-1
      T(i,j)=T(j,i)
    enddo
    enddo

    init=1
  endif
  ! --- end of loading variables for gaussian process --------------------------

  ! --- do prediction using Gaussian Process -----------------------------------

  if(n0.ne.ndim) then
    print*,"input xs0 dimension wrong"
    stop
  endif
  xs=xs0

  if(trim(xhandle).eq.'normal') then
      ! do nothing
  elseif(trim(xhandle).eq.'reverse') then
      do i=1,ns
      do j=1,ndim
      xs(j,i)=1.d0/xs(j,i)
      enddo
      enddo
  elseif(trim(xhandle).eq.'log') then
      do i=1,ns
      do j=1,ndim
      xs(j,i)=log(xs(j,i))
      enddo
      enddo
  endif

  !! calculate potential
  allocate(Ks(ns,nt),W(ns,nt))
  call covMaternard(covFunc_d,hyp_cov,ndim,xs,ns,x,nt,0,Ks)
  call dgemv('N',ns,nt,1.d0,Ks,ns,alpha,1,0.d0,ys,1) !! ys=Ks*alpha, ys=Ks*T*y

! !! calculate covariance
! do i=1,ns
!     cv1(i)=0.d0
!     sumw=sum(Ks(i,1:nt))
!     do j=1,nt
!       cv1(i)=cv1(i)+(Ks(i,j)*alpha(j))**2-(Ks(i,j)*alpha(j)**2)
!     enddo
! enddo

! ! W=Ks*T
! call dgemm('N','N',ns,nt,nt,1.d0,Ks,ns,T,nt,0.d0,W,ns)
! call dgemv('N',ns,nt,1.d0,W,ns,y,1,0.d0,ys,1) !! ys=W*y

! do i=1,ns
!   cv2(i)=0.d0
!   sumw=sum(W(i,1:nt)) !! should be One
!   do j=1,nt
!     cv2(i)=cv2(i)+(W(i,j)*y(j))**2-(W(i,j)*y(j)**2)
!   enddo
! enddo

! i=987
! do j=1,nt
! write(1000,'(10es30.15)') W(i,j),Ks(i,j)
! enddo

  deallocate(Ks,W)

  ! ----------------------------------------------------------------------------
  return
end subroutine pes_gp

subroutine covMaternard(d, hyp, ndim, x, nx, z, nz, iderivatives, K)
  implicit none
  integer,intent(in) :: ndim,nx,nz,d,iderivatives
  real*8,intent(in)  :: hyp(ndim+1),x(ndim,nx),z(ndim,nz)
  real*8,intent(out) :: K(nx,nz)
  real*8 :: ell(ndim),sf2,kt(nx,nz),xt(ndim,nx),zt(ndim,nz),ki(nx,nz)
  integer :: i,j
  do i=1,ndim
    ell(i) = exp(hyp(i))
  enddo
  sf2 = exp(2.d0*hyp(ndim+1))

  if(d.ne.5) stop "covMaternard_d.ne.5"
  if (iderivatives.ne.0) stop "iderivatives.ne.0"

  ! precompute distances
  xt=x*dsqrt(d*1.d0)
  do j=1,nx
    xt(1:ndim,j)=xt(1:ndim,j)/ell
  enddo

  zt=z*dsqrt(d*1.d0)
  do j=1,nz
    zt(1:ndim,j)=zt(1:ndim,j)/ell
  enddo

! do j=1,nx
! do i=1,ndim
!   xt(i,j)=dsqrt(d*1.d0)/ell(i)*x(i,j)
! enddo
! enddo

! do j=1,nz
! do i=1,ndim
!   zt(i,j)=dsqrt(d*1.d0)/ell(i)*z(i,j)
! enddo
! enddo

  call sq_dist(ndim,xt,nx,zt,nz,K)

  K=sqrt(K)

  !m = @(t,f) f(t).*exp(-t); dm = @(t,f) df(t).*exp(-t);

  do i=1,nx
  do j=1,nz
    kt(i,j)=1.d0+K(i,j)*(1.d0+K(i,j)/3.d0)
  enddo
  enddo

  do i=1,nx
  do j=1,nz
    K(i,j)=sf2*kt(i,j)*exp(-K(i,j))
  enddo
  enddo

  return
end subroutine covMaternard

subroutine sq_dist(D,a,n,b,m,C)
  ! sq_dist - a function to compute a matrix of all pairwise squared distances
  ! between two sets of vectors, stored in the columns of the two matrices, a
  ! (of size D by n) and b (of size D by m). If only a single argument is given
  ! or the second matrix is empty, the missing matrix is taken to be identical
  ! to the first.
  ! a is of size Dxn, b is of size Dxm, C is of size nxm.
  !
  ! Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-12-13.
  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-26
  implicit none
  integer,intent(in) :: D,n,m
  real*8,intent(in)  :: a(D,n),b(D,m)
  real*8,intent(out) :: C(n,m)
  real*8 :: ax(D,n),bx(D,m),amean(D),bmean(D),mu(D),asum(n),bsum(m)
  integer :: i,j

  ! subtract mean
  do i=1,D
      amean(i)=sum(a(i,1:n))/(n*1.d0)
      bmean(i)=sum(b(i,1:m))/(m*1.d0)
  enddo
  mu = (1.d0*m/(n+m))*amean + (1.d0*n/(n+m))*bmean
  do i=1,n
      ax(1:D,i)=a(1:D,i)-mu(1:D)
  enddo
  do i=1,m
      bx(1:D,i)=b(1:D,i)-mu(1:D)
  enddo

  ! compute squared distances
  asum=0.d0
  do j=1,n
  do i=1,D
      asum(j)=asum(j)+ax(i,j)**2
  enddo
  enddo
  bsum=0.d0
  do j=1,m
  do i=1,D
      bsum(j)=bsum(j)+bx(i,j)**2
  enddo
  enddo

  call dgemm('t','n',n,m,D,-2.d0,ax,D,bx,D,0.d0,C,n)
  do j=1,m
  do i=1,n
      C(i,j)=C(i,j)+asum(i)+bsum(j)
  enddo
  enddo

  ! numerical noise can cause C to negative i.e. C > -1e-14
  C = max(C,0.d0)
  return
end subroutine sq_dist
