program test
  implicit none
  real*8,parameter :: ev=27.21138505d0
  integer,parameter :: nd=6,nt=50000
  real*8 :: x(nd,nt),rg(2,nd),v
  logical :: OH3DistTest
  integer :: i
  rg(1,:)=0.d0
  rg(2,:)=1.d0
  rg(:,1)=[1.350,2.750]
  rg(:,2)=[1.500,6.000]
  rg(:,3)=[1.600,6.000]
  rg(:,4)=[1.450,7.850]
  rg(:,5)=[1.000,8.900]
  rg(:,6)=[0.900,8.750]
  call lhs(nd,nt,rg,x)
  do i=1,nt
      if(OH3DistTest(x(:,i)).eq..false.) cycle
      if(x(1,i).gt.x(2,i)) cycle
      if(x(2,i).gt.x(3,i)) cycle
      call nnpes_oh3(x(:,i),v); v=v*ev
      if(v.gt.4.d0) cycle
      write(*,'(<nd+1>f10.5)') x(:,i),v
  enddo
end program

function OH3DistTest(r) result(res)
  implicit none
  real*8,intent(in) :: r(6)
  logical :: res
  real*8 :: x1,y1,x2,y2,v
  res=.true.
  if(r(1).gt.r(2)+r(4)) goto 321
  if(r(1).gt.r(3)+r(5)) goto 321
  if(r(1).lt.abs(r(2)-r(4))) goto 321
  if(r(1).lt.abs(r(3)-r(5))) goto 321
  x1=(r(1)**2+r(3)**2-r(5)**2)/2.d0/r(1)
  y1=sqrt(r(3)**2-x1**2)
  x2=(r(1)**2+r(2)**2-r(4)**2)/2.d0/r(1)
  y2=sqrt(r(2)**2-x2**2)
  if(r(6).gt.sqrt((x1-x2)**2+(y1+y2)**2)) goto 321
  if(r(6).lt.sqrt((x1-x2)**2+(y1-y2)**2)) goto 321
  return
  321 res=.false.
  return
end function

subroutine lhs(nd,nt,rg,x)
  ! latin hypercube sampling
  ! nd: dimension
  ! nt: total points
  implicit none
  integer,intent(in) :: nd,nt
  real*8,intent(in) :: rg(2,nd)
  real*8,intent(out) :: x(nd,nt)
  real*8 :: xp(nd,nt),xd(nd,nt)
  integer :: ip(1),i,j

  call random_seed()
  call random_number(xp)
  call random_number(xd)

  do i=1,nd
    do j=1,nt
      ip=minloc(xp(i,1:nt))
      xp(i,ip(1))=2.d0
      x(i,j)=(nt-ip(1)+1-xd(i,j))/nt*rg(1,i)+(ip(1)-1+xd(i,j))/nt*rg(2,i)
    enddo
  enddo

  return
end subroutine
