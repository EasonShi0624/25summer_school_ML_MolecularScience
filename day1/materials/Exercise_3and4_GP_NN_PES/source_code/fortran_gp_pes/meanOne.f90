subroutine meanOne(ndim,hyp,x,nx,iderivatives,A)
  ! One mean function. The mean function does not have any parameters.
  !
  ! A(x) = 1
  !
  ! Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-08-04.
  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-27
  implicit none
  integer,intent(in) :: ndim,nx,iderivatives
  real*8,intent(in)  :: hyp(ndim),x(ndim,nx)
  real*8,intent(out) :: A(nx)
  if(iderivatives.eq.0) then  ! evaluate mean
    A = 1.d0
  else                        ! derivative
    A = 0.d0
  endif
  return
end subroutine meanOne
