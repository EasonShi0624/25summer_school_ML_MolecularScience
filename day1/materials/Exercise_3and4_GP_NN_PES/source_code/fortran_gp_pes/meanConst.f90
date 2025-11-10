subroutine meanConst(ndim,hyp,x,nx,iderivatives,A)
  ! Constant mean function. The mean function is parameterized as:
  !
  ! A(x) = c
  !
  ! The hyperparameter is:
  !
  ! hyp = [ c ]
  !
  ! Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-08-04.
  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-27
  implicit none
  integer,intent(in) :: ndim,nx,iderivatives
  real*8,intent(in)  :: hyp,x(ndim,nx)
  real*8,intent(out) :: A(nx)
  if (iderivatives.eq.0) then  ! evaluate mean
    A = hyp
  elseif (iderivatives.eq.1) then !derivative
    A = 1.d0
  else
    A = 0.d0
  endif
  return
end subroutine meanConst
