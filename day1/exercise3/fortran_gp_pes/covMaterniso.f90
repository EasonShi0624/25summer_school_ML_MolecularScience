subroutine covMaterniso(d, hyp, ndim, x, nx, z, nz, iderivatives, K)
  ! Matern covariance function with nu = d/2 and isotropic distance measure. For
  ! d=1 the function is also known as the exponential covariance function or the 
  ! Ornstein-Uhlenbeck covariance in 1d. The covariance function is:
  !
  !   k(x^p,x^q) = sf^2 * f( sqrt(d)*r ) * exp(-sqrt(d)*r)
  !
  ! with f(t)=1 for d=1, f(t)=1+t for d=3 and f(t)=1+t+t²/3 for d=5.
  ! Here r is the distance sqrt((x^p-x^q)'*inv(P)*(x^p-x^q)), P is ell times
  ! the unit matrix and sf2 is the signal variance. The hyperparameters are:
  !
  ! hyp = [ log(ell)
  !         log(sf) ]
  !
  ! Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-26
  implicit none
  integer,intent(in) :: ndim,nx,nz,d,iderivatives
  real*8,intent(in)  :: hyp(2),x(ndim,nx),z(ndim,nz)
  real*8,intent(out) :: K(nx,nz)
  real*8 :: ell,sf2,kt(nx,nz)
  integer :: i,j 
  ell = exp(hyp(1))
  sf2 = exp(2.d0*hyp(2))

  if(d.ne.1 .and. d.ne.3 .and. d.ne.5) then
    print*,'only 1, 3 and 5 allowed for d'
    stop
  endif
  if(iderivatives.ne.0 .and. iderivatives.ne.1 .and. iderivatives.ne.2) then
    print*,'Unknown hyperparameter'
    print*,'only 0, 1 and 2 allowed for iderivatives'
    stop
  endif

  call sq_dist(ndim,dsqrt(d*1.d0)/ell*x,nx,dsqrt(d*1.d0)/ell*z,nz,K)
  do i=1,nx
  do j=1,nz
      K(i,j)=dsqrt(K(i,j))
  enddo
  enddo

  if (iderivatives.eq.0) then     ! covariances
    call f_covMaterniso(K,kt,nx,nz,d,0)
    do i=1,nx
    do j=1,nz
      K(i,j) = sf2*kt(i,j)*exp(-K(i,j))
    enddo
    enddo
  elseif (iderivatives.eq.1) then ! derivatives
    call f_covMaterniso(K,kt,nx,nz,d,1)
    do i=1,nx
    do j=1,nz
      K(i,j) = sf2*kt(i,j)*exp(-K(i,j))*K(i,j)
    enddo
    enddo
  elseif (iderivatives.eq.2) then
    call f_covMaterniso(K,kt,nx,nz,d,0)
    do i=1,nx
    do j=1,nz
      K(i,j) = 2.d0*sf2*kt(i,j)*exp(-K(i,j))
    enddo
    enddo
  else
    print*,'Unknown hyperparameter'
    stop
  endif
  return
end subroutine covMaterniso

subroutine f_covMaterniso(t,f,na,nb,d,k)
  ! calculate f(t) or df(t), df(t)=f(t)-f'(t)
  ! if d=1: f = 1; df = 1;
  ! if d=3: f = 1+t; df = t;
  ! if d=5: f = 1 + t.*(1+t/3); df = t.*(1+t)/3;
  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-26
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
  if (d.eq.1) then
    f=1.d0
  elseif (d.eq.3) then
    f=t
    if(k.eq.0) f=f+1.d0
  elseif (d.eq.5) then
    if(k.eq.0) then
      do i=1,na
      do j=1,nb
        f(i,j)=1.d0+t(i,j)*(1.d0+t(i,j)/3.d0)
      enddo
      enddo
    elseif(k.eq.1) then
      do i=1,na
      do j=1,nb
        f(i,j)=t(i,j)*(1.d0+t(i,j))/3.d0
      enddo
      enddo
    endif
  endif
  return
end subroutine f_covMaterniso

subroutine test_covMaterniso
  !% x=[ 1.3,3,2,4,5,6,1,5,4;...
  !%     3.2,3,4,2,6,5,7,2,7];
  !% z=[ 1.3,3,2,4,5,6,1,2;...
  !%     3.2,3,4,2,6,5,7,1];
  !% z=z+0.1;
  !% hyp=[0.1 0.5];
  !% K=covMaterniso(3,hyp,x',z')
  !% K=covMaterniso(3,hyp,x',z',1)
  !% K=covMaterniso(3,hyp,x',z',2)
  implicit none
  integer,parameter :: ndim=2,nx=9,nz=8
  integer :: d,iderivatives
  real*8 :: hyp(2),x(ndim,nx),z(ndim,nz),K(nx,nz)
  integer :: i
  d=1
  hyp=[0.1d0,0.5d0]
  iderivatives=0
  x(1,1:9)=[1.3d0,3.d0,2.d0,4.d0,5.d0,6.d0,1.d0,5.d0,4.d0]
  x(2,1:9)=[3.2d0,3.d0,4.d0,2.d0,6.d0,5.d0,7.d0,2.d0,7.d0]
  z(1,1:8)=[1.3d0,3.d0,2.d0,4.d0,5.d0,6.d0,1.d0,2.d0]
  z(2,1:8)=[3.2d0,3.d0,4.d0,2.d0,6.d0,5.d0,7.d0,1.d0]
  z=z+0.1d0;
  call covMaterniso(d, hyp, ndim, x, nx, z, nz, iderivatives, K)
  do i=1,nx
  write(*,'(<nz>f10.4)') K(i,1:nz)
  enddo
end subroutine
