subroutine likGauss_prediction(hyp,nt,y,mu,s2,lp,ymu,ys2)
  ! likGauss - Gaussian likelihood function for regression. The expression for the 
  ! likelihood is 
  !   likGauss(t) = exp(-(t-y)^2/2*sn^2) / sqrt(2*pi*sn^2),
  ! where y is the mean and sn is the standard deviation.
  !
  ! The hyperparameters are:
  !
  ! hyp = [  log(sn)  ]
  !
  ! Several modes are provided, for computing likelihoods, derivatives and moments
  ! respectively, see likFunctions.m for the details. In general, care is taken
  ! to avoid numerical issues when the arguments are extreme.
  !
  ! Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2015-07-13.
  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-28

  implicit none
  real*8,intent(in)  :: hyp
  integer,intent(in) :: nt
  real*8,intent(in)  :: y(nt),mu(nt),s2(nt)
  real*8,intent(out) :: lp(nt),ymu(nt),ys2(nt)
  real*8,parameter :: pi=3.14159265358979323846d0
  real*8 :: sn2
  integer :: i
  sn2=exp(2.d0*hyp)

  !%lZ = -(y-mu).^2./(sn2+s2)/2 - log(2*pi*(sn2+s2))/2; % log part function
  do i=1,nt
    lp(i) = -(y(i)-mu(i))**2/(sn2+s2(i))/2.d0 - log(2.d0*pi*(sn2+s2(i)))/2.d0
  enddo

  ymu=mu
  ys2=s2+sn2
  return
end subroutine likGauss_prediction
