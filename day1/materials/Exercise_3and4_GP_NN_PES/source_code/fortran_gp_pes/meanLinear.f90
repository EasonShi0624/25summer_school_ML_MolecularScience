subroutine meanLinear(ndim,hyp,x,nx,iderivatives,A)
  ! Linear mean function. The mean function is parameterized as:
  !
  ! A(x) = sum_i a_i * x_i;
  !
  ! The hyperparameter is:
  !
  ! hyp = [ a_1
  !         a_2
  !         ..
  !         a_D ]
  !
  ! Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-01-10.
  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-27
  implicit none
  integer,intent(in) :: ndim,nx,iderivatives
  real*8,intent(in)  :: hyp(ndim),x(ndim,nx)
  real*8,intent(out) :: A(nx)

  if(iderivatives.eq.0) then ! evaluate mean
    call dgemv('t',ndim,nx,1.d0,x,ndim,hyp,1,0.d0,A,1)
  elseif (iderivatives.le.ndim) then ! derivative
    A = x(iderivatives,1:nx)
  else
    A = 0.d0
  endif
  return
end subroutine meanLinear
