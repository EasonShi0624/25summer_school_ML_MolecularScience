! --------------------------------------------------------------------
      program main
      implicit none
      real*8 r(6),vab,v1,v2,v3
      integer id

      open(101,file='test',action='read')
      do while(.true.)
        read(101,*,end=999) id,r,vab

        call OH3PES('NN1',1,r,v1)
        call OH3PES('NN2',1,r,v2)
        call OH3PES('NN3',1,r,v3)

        write(*,'(i6,10f10.5)') id,vab,[v1,v2,v3]*27.2116d0
      enddo
999   close(101)
      stop
      end

! --------------------------------------------------------------------
