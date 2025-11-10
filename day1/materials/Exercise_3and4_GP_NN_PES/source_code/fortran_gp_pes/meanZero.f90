subroutine meanZero(ndim,hyp,x,nx,iderivatives,A)
  ! Zero mean function. The mean function does not have any parameters.
  !
  ! A(x) = 0
  !
  ! Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-01-10.
  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-27
  implicit none
  integer,intent(in) :: ndim,nx,iderivatives
  real*8,intent(in)  :: hyp,x(ndim,nx)
  real*8,intent(out) :: A(nx)

  A = 0.d0
  return
end subroutine meanZero
