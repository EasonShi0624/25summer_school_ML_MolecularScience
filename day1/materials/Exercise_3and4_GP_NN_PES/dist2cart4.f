      program main
      implicit none
      real*8 r(6),c(3,4),rtest(6),v
      integer :: id=0

      do while(.true.)
        read(*,*,end=99) r,v
        id=id+1
        call dist2cart4(r,c)
        write(*,509) id,v,c*0.529177d0

      call cart2dist4(c,rtest)
      if(dot_product(r-rtest,r-rtest).gt.1.d-5) print*,id

      enddo
   99 continue
      stop
  509 format(' 4',/,' OH3 # ',i6.6," E(eV) = ",f16.8,/,
     &' O ',3(' ',f14.8),/,
     &' H ',3(' ',f14.8),/,
     &' H ',3(' ',f14.8),/,
     &' H ',3(' ',f14.8)
     &)
      end

      subroutine dist2cart4(r,c)
      implicit none
      real*8,intent(in) :: r(6)
      real*8,intent(out) :: c(3,4)
      real*8 angle,r1,r2,r3
      integer i,j,k
      real*8 cab,cx,ch,dab,dx,dh,cdh,di
      angle(r1,r2,r3)=dacos(  max(-1.d0,min(1.d0,(r1**2+r2**2-r3**2)/(2.d0*r1*r2))) )
      c=0.d0
      c(1,2)=r(1)
      cab=angle(r(1),r(2),r(4))
      c(1,3)=r(2)*dcos(cab)
      c(2,3)=r(2)*dsin(cab)
      cx=c(1,3)
      ch=c(2,3)
      dab=angle(r(1),r(3),r(5))
      c(1,4)=r(3)*dcos(dab)
      dx=c(1,4)
      dh=r(3)*dsin(dab)
      cdh=dsqrt(r(6)**2-(cx-dx)**2)
      di=angle(ch,dh,cdh)
      c(2,4)=dh*dcos(di)
      c(3,4)=dh*dsin(di)
      return
      end subroutine

      subroutine cart2dist4(c,r)
      implicit none
      integer,parameter :: natom=4
      real*8,intent(in) :: c(3,natom)
      real*8,intent(out) :: r(natom*(natom-1)/2)
      integer i,j,k,id
      id=0
      do i=1,natom-1
      do j=i+1,natom
        id=id+1
        r(id)=0.d0
        do k=1,3
        r(id)=r(id)+(c(k,i)-c(k,j))**2
        enddo
        r(id)=dsqrt(r(id))
      enddo
      enddo
      return
      end subroutine

