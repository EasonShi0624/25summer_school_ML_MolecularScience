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

subroutine test_sq_dist
  !% a=[ 1.3,3,2,4,5,6,1,5,4;...
  !%     3.2,3,4,2,6,5,7,2,7];
  !% b=[ 1.3,3,2,4,5,6,1,2;...
  !%     3.2,3,4,2,6,5,7,1];
  !% b=b+0.1;
  !% C=sq_dist(a,b);
  implicit none
  integer,parameter :: D=2,n=9,m=8
  real*8 :: a(D,n),b(D,m),C(n,m)
  integer :: i
  a(1,1:n)=[1.3,3.0,2.0,4.0,5.0,6.0,1.0,5.0,4.0]
  a(2,1:n)=[3.2,3.0,4.0,2.0,6.0,5.0,7.0,2.0,7.0]
  b(1,1:m)=[1.3,3.0,2.0,4.0,5.0,6.0,1.0,2.0]
  b(2,1:m)=[3.2,3.0,4.0,2.0,6.0,5.0,7.0,1.0]
  b=b+0.1d0
  call sq_dist(D,a,n,b,m,C)
  print*,C(2,5)
  do i=1,n
  write(*,'(<m>f10.4)') C(i,1:m)
  enddo
end subroutine test_sq_dist
