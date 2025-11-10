subroutine covMaternard(d, hyp, ndim, x, nx, z, nz, iderivatives, K)
  ! Matern covariance function with nu = d/2 and with Automatic Relevance
  ! Determination (ARD) distance measure. For d=1 the function is also known as
  ! the exponential covariance function or the Ornstein-Uhlenbeck covariance 
  ! in 1d. The covariance function is:
  !
  !   k(x^p,x^q) = sf^2 * f( sqrt(d)*r ) * exp(-sqrt(d)*r)
  !
  ! with f(t)=1 for d=1, f(t)=1+t for d=3 and f(t)=1+t+t²/3 for d=5.
  ! Here r is the distance sqrt((x^p-x^q)'*inv(P)*(x^p-x^q)), where the P matrix
  ! is diagonal with ARD parameters ell_1^2,...,ell_D^2, where D is the dimension
  ! of the input space and sf2 is the signal variance. The hyperparameters are:
  !
  ! hyp = [ log(ell_1)
  !         log(ell_2)
  !          ..
  !         log(ell_D)
  !         log(sf) ]
  !
  ! Copyright (c) by Hannes Nickisch, 2013-10-13.
  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-27
  implicit none
  integer,intent(in) :: ndim,nx,nz,d,iderivatives
  real*8,intent(in)  :: hyp(ndim+1),x(ndim,nx),z(ndim,nz)
  real*8,intent(out) :: K(nx,nz)

! real*8 :: ell(ndim),sf2,kt(nx,nz),xt(ndim,nx),zt(ndim,nz),ki(nx,nz)
  real*8 :: sf2
  real*8,allocatable :: ell(:),kt(:,:),xt(:,:),zt(:,:),ki(:,:)

  integer :: i,j

  allocate(ell(ndim),kt(nx,nz),xt(ndim,nx),zt(ndim,nz),ki(nx,nz))

  do i=1,ndim
    ell(i) = exp(hyp(i))
  enddo
  sf2 = exp(2.d0*hyp(ndim+1))

  if(d.ne.1 .and. d.ne.3 .and. d.ne.5) then
    print*,'only 1, 3 and 5 allowed for d'
    stop
  endif
  if(iderivatives.lt.0 .or. iderivatives.gt.ndim+1) then
    print*,'only 0 to ndim+1 allowed for iderivatives'
    stop
  endif
  ! precompute distances
  do j=1,nx
  do i=1,ndim
    xt(i,j)=dsqrt(d*1.d0)/ell(i)*x(i,j)
  enddo
  enddo

  do j=1,nz
  do i=1,ndim
    zt(i,j)=dsqrt(d*1.d0)/ell(i)*z(i,j)
  enddo
  enddo

  call sq_dist(ndim,xt,nx,zt,nz,K)
  do i=1,nx
  do j=1,nz
      K(i,j)=dsqrt(K(i,j))
  enddo
  enddo

  !m = @(t,f) f(t).*exp(-t); dm = @(t,f) df(t).*exp(-t);
  if (iderivatives.eq.0) then ! covariances
    call f_covMaternard(K,kt,nx,nz,d,0)
    do i=1,nx
    do j=1,nz
      K(i,j)=sf2*kt(i,j)*exp(-K(i,j))
    enddo
    enddo
  else                        ! derivatives
    if (iderivatives.le.ndim) then              ! length scale parameter
      call sq_dist(1,dsqrt(d*1.d0)/ell(iderivatives)*x(iderivatives,1:nx),nx,dsqrt(d*1.d0)/ell(iderivatives)*z(iderivatives,1:nz),nz,ki);
      call f_covMaternard(K,kt,nx,nz,d,1)
      do i=1,nx
      do j=1,nz
        K(i,j) = sf2*kt(i,j)*exp(-K(i,j))*Ki(i,j)
        if(Ki(i,j).le.1.d-12) K(i,j)=0.d0
      enddo
      enddo
    elseif (iderivatives.eq.(ndim+1)) then      ! magnitude parameter
      call f_covMaternard(K,kt,nx,nz,d,0)
      do i=1,nx
      do j=1,nz
        K(i,j)=2.d0*sf2*kt(i,j)*exp(-K(i,j))
      enddo
      enddo
    else
      print*,'Unknown hyperparameter'
      stop
    endif
  endif

  deallocate(ell,kt,xt,zt,ki)
  return
end subroutine covMaternard

subroutine f_covMaternard(t,f,na,nb,d,k)
  ! calculate f(t) or df(t), df(t)=(f(t)-f'(t))/t
  ! case 1, f = @(t) 1;               df = @(t) 1./t;
  ! case 3, f = @(t) 1 + t;           df = @(t) 1;
  ! case 5, f = @(t) 1 + t.*(1+t/3);  df = @(t) (1+t)/3;
  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-27
  implicit none
  integer,intent(in) :: na,nb,d,k
  real*8,intent(in)  :: t(na,nb)
  real*8,intent(out) :: f(na,nb)
  integer :: i,j
  if(d.ne.1 .and. d.ne.3 .and. d.ne.5) then
    print*,'only 1, 3 and 5 allowed for d'
    stop
  endif
  if(k.ne.0 .and. k.ne.1) then
    print*,'only 0 and 1 allowed for k, to calculate f or df'
    stop
  endif

  if (d.eq.1 .and. k.eq.0) then
    f=1.d0
  elseif (d.eq.3 .and. k.eq.0) then
    f=t+1.d0
  elseif (d.eq.5 .and. k.eq.0) then
    do i=1,na
    do j=1,nb
      f(i,j)=1.d0+t(i,j)*(1.d0+t(i,j)/3.d0)
    enddo
    enddo
  elseif (d.eq.1 .and. k.eq.1) then
    do i=1,na
    do j=1,nb
      f(i,j)=1.d0/t(i,j)
    enddo
    enddo
  elseif (d.eq.3 .and. k.eq.1) then
    f=1.d0
  elseif (d.eq.5 .and. k.eq.1) then
    do i=1,na
    do j=1,nb
      f(i,j)=(1.d0+t(i,j))/3.d0
    enddo
    enddo
  endif
  return
end subroutine f_covMaternard

subroutine test_covMaternard
  !% x=[ 1.3,3,2,4,5,6,1,5,4;...
  !%     3.2,3,4,2,6,5,7,2,7];
  !% z=[ 1.3,3,2,4,5,6,1,2;...
  !%     3.2,3,4,2,6,5,7,1];
  !% z=z+0.1;
  !% hyp=[0.1 0.5 0.2];
  !% K=covMaterniso(3,hyp,x',z')
  !% K=covMaterniso(3,hyp,x',z',1)
  !% K=covMaterniso(3,hyp,x',z',2)
  !% K=covMaterniso(3,hyp,x',z',3)
  implicit none
  integer,parameter :: ndim=2,nx=9,nz=8
  integer :: d,iderivatives
  real*8 :: hyp(ndim+1),x(ndim,nx),z(ndim,nz),K(nx,nz)
  integer :: i
  d=1
  hyp=[0.1d0,0.5d0,0.2d0]
  iderivatives=3
  x(1,1:9)=[1.3d0,3.d0,2.d0,4.d0,5.d0,6.d0,1.d0,5.d0,4.d0]
  x(2,1:9)=[3.2d0,3.d0,4.d0,2.d0,6.d0,5.d0,7.d0,2.d0,7.d0]
  z(1,1:8)=[1.3d0,3.d0,2.d0,4.d0,5.d0,6.d0,1.d0,2.d0]
  z(2,1:8)=[3.2d0,3.d0,4.d0,2.d0,6.d0,5.d0,7.d0,1.d0]
  z=z+0.1d0;
  call covMaternard(d, hyp, ndim, x, nx, z, nz, iderivatives, K)
  do i=1,nx
  write(*,'(<nz>f10.4)') K(i,1:nz)
  enddo
end subroutine
