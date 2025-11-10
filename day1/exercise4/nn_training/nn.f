!  _   _  _____  _   _  ____      _     _     
! | \ | || ____|| | | ||  _ \    / \   | |    
! |  \| ||  _|  | | | || |_) |  / _ \  | |    
! | |\  || |___ | |_| ||  _ <  / ___ \ | |___ 
! |_| \_||_____| \___/ |_| \_\/_/   \_\|_____|
!  _   _  _____  _____ __        __ ___   ____   _  __ ____  
! | \ | || ____||_   _|\ \      / // _ \ |  _ \ | |/ // ___| 
! |  \| ||  _|    | |   \ \ /\ / /| | | || |_) || ' / \___ \ 
! | |\  || |___   | |    \ V  V / | |_| ||  _ < | . \  ___) |
! |_| \_||_____|  |_|     \_/\_/   \___/ |_| \_\|_|\_\|____/ 
!  ____    ___   _  _____ 
! |___ \  / _ \ / ||___ / 
!   __) || | | || |  |_ \ 
!  / __/ | |_| || | ___) |
! |_____| \___/ |_||____/ 
!  ___            ___  _                    _             
! |   \  _ _     / __|| |_   ___  _ _    _ | | _  _  _ _  
! | |) || '_|_  | (__ | ' \ / -_)| ' \  | || || || || ' \ 
! |___/ |_| (_)  \___||_||_|\___||_||_|  \__/  \_,_||_||_|
!
! Training Neural Network with Levenberg-Marquardt Method
! CHEN JUN, 2013-3-1
! BLAS or LAPACK subroutines: dsyrk, dgemv, dgesv
! --------------------------------------------------------------------
      module nnmod
      save
! ----these parameters defines the NN structure-----------------------
      integer :: ndim ! input dimension
      integer,parameter :: nout=1 ! output dimension
      integer :: nhid ! number of hidden layers
      integer,allocatable :: nl(:) ! neuron numbers of each layer
      integer :: mnl ! maximum number of neurons-ndim
      integer :: nw ! total number of weights and biases
! ----these parameters defines the training set-----------------------
      real*8 :: tRatio ! nt=floor(tRatio*ntot)
      character*99 :: re_file ! file name containing ab initio points
      integer,parameter :: nmax=1000000
      integer :: ntot ! total number of ab initio points
      integer :: nt ! number of training set
      integer :: nv ! number of validation set
      real*8,allocatable :: x(:,:),y(:) ! stores all points
      real*8,allocatable :: xmin(:),xmax(:),xrange(:)
      real*8 :: ymin,ymax,yrange ! stores min max and range of each dimension
      logical,allocatable :: m00(:) ! select which points are used for training
      real*8,allocatable :: xt(:,:),yt(:),xv(:,:),yv(:) ! stores normalized training and validation set
! ----these parameters defines the training progress------------------
      integer :: nloop ! maximum nloop epoches
      integer :: ncycle ! fit ncycle times
      integer :: max_fail=6
      real*8,allocatable :: w(:,:,:),w_back(:,:,:),w_save(:,:,:)
      real*8,allocatable :: errt(:),errv(:) ! stores training and validation error of each epoch
      real*8,allocatable :: zt(:,:,:),zv(:,:,:) ! stores input and output of each neuron each sample point, large, (mnl+1)*(nhid+2)*(nt or nv)
      real*8,allocatable :: et(:),ev(:) ! et(nt),fitting error
      real*8,allocatable :: dedw(:,:,:) ! stores de/dw of each NN param each training sample
      real*8,allocatable :: JAC(:,:),JTJ(:,:),JTJM(:,:) ! JAC(nt,nw),JTJ(nw,nw), large
      real*8,allocatable :: dw(:),dw_back(:) ! dw(nw)
      integer,allocatable :: ipiv(:) ! ipiv(nw), used in dgesv
      integer,allocatable :: index_table(:,:)
      integer :: ierr
      end module nnmod

! --------------------------------------------------------------------
      program neural_networks
      use nnmod
      implicit none
      integer icycle,ep,ifail
      real*8 mu,rmse

      call random_seed
      call echo_time
      call system("echo `whoami` $HOST")
      call read_input_re ! read input and points, allocate arrays

      do 200 icycle=1,ncycle ! repeat fitting ncycle times
        write(*,*) 
        call echo_time
        write(*,'(" Fitting Count = ",i3.1)') icycle
        call random_divide(icycle)
        call init_zt_zv ! zt,zv stores input&output of each neuron
        call init_wb

        mu=1.d-1
        do 300 ep=0,nloop ! training nloop epoches until overfitting
          call feedforward('a') ! calculate zt and zv
          call performance(ep)

          ! early stop, check validation & max_fail, else go on training
          if(ep.gt.0.and.errv(ep).gt.errv(ep-1)) then
             ifail=ifail+1
          else
             ifail=0
             w_save=w
          endif

          write(*,501) ep,nloop,mu,errt(ep),errv(ep),ifail,max_fail
          if(ifail.ge.max_fail) goto 301 ! early stopping
          if(ep.eq.nloop) goto 301 ! max epoch stopping

          call backward_propagation ! calc d(error)/d(w,b), write to JAC
          w_back=w; dw_back=dw
          do 400 while(.true.)
            dw=dw_back; w=w_back
            call get_delta_w(mu)
            call update_w
            call feedforward('t')
            call performance(ep+1)
            if(errt(ep+1).lt.errt(ep)) then
              mu=max(mu/9.d0,1.d-10)
              w_back=w
              exit ! exit do loop 400
            else
              mu=mu*8.d0
            endif
            if(mu.gt.1.d10) then
              print*, "MU.gt.1.d10"
              goto 301 ! end this fitting process
            endif
400       enddo

300     enddo ! end of one training process
301     continue

        rmse=errt(ep-ifail)**2*tRatio+errv(ep-ifail)**2*(1-tRatio)
        rmse=dsqrt(rmse)
        write(*,502) icycle,rmse,errt(ep-ifail),errv(ep-ifail)
        call savenet(icycle,nhid,nl,mnl,w_save,xmin,xmax,ymin,ymax,rmse)

200   enddo

      call deallocate_all
      call echo_time
      stop
501   format(1x,"Epoch ",i4,"/",i4,"; Mu=",es7.0,
     &"; Perform=",es9.2," /",es9.2," meV; Val=",i2,"/",i2)
502   format(1x,'Total rmse_',i2.2,'=',f12.4,' meV; ',
     &'train=',f12.4,'; validation=',f12.4)
      end

! --------------------------------------------------------------------
      subroutine echo_time
      implicit integer(a-z)
      call gettim(hh,mi,ss,ms)
      call getdat(yy,mm,dd)
      write(*,777)yy,mm,dd,hh,mi,ss,ms
  777 format(1x,i4,2('/',i2.2),2x,2(i2.2,':'),i2.2,'.',i2.2)
      return
      end

! --------------------------------------------------------------------
      subroutine read_input_re
      use nnmod
      implicit none
      integer i

      print*,"Training Neural Network with Levenberg-Marquardt Method"
! ----read input file ------------------------------------------------
      open(1,file='input',status='old',action='read')

      read(1,*) ndim          ! input dimension
      read(1,*) nhid          ! number of hidden layers
      allocate(nl(0:nhid+1))
      nw=0                    ! total number of weights and biases
      nl(0)=ndim              ! input dimension
      nl(nhid+1)=nout         ! output dimension
      do i=1,nhid
        read(1,*) nl(i)       ! neurons of hidden layers
        nw=nw+nl(i-1)*nl(i)+nl(i)
      enddo
      nw=nw+nl(nhid)*nl(nhid+1)+nl(nhid+1)
      mnl=maxval(nl)

      read(1,*) tRatio        ! ratio for training
      read(1,*) nloop         ! max epoches
      read(1,*) ncycle        ! fit ncycle times
      read(1,*) max_fail      ! validation test fails max times, early stoping
      read(1,*) re_file       ! ab initio data file
      close(1)

      write(*,'(1x,"NN structure is ",i3.1,\)') nl(0)
      do i=1,nhid+1
        write(*,'(" -",i3.1,\)') nl(i)
      enddo
      write(*,'(" with ",i5," weight and bias")') nw

! ----load all ab initio points --------------------------------------
      allocate(x(ndim,nmax),y(nmax),stat=ierr)
      ntot=0
      open(1,file=trim(re_file),status='old',action='read')
      do while (.true.)
        read(1,*,end=901) x(1:ndim,ntot+1),y(ntot+1)
        ntot=ntot+1
      enddo
901   close(1)

      nt=floor(ntot*tRatio) ! number of training data
      nv=ntot-nt              ! number of validation data

      print*,"Load ",ntot," ab initio data from ",trim(re_file)
      write(*,'("  Training set ",i6," , Validation set ",i6)') nt,nv

! ----minmax each dimension into -1~1 --------------------------------
      allocate(xmin(ndim),xmax(ndim),xrange(ndim))
      do i=1,ndim
      xmin(i)=minval(x(i,1:ntot))
      xmax(i)=maxval(x(i,1:ntot))
      xrange(i)=xmax(i)-xmin(i)
      if (xrange(i).ne.0.d0) then
      x(i,1:ntot)=(x(i,1:ntot)-xmin(i))/xrange(i)*2.d0-1.d0
      endif
      enddo
      ymin=minval(y(1:ntot))
      ymax=maxval(y(1:ntot))
      yrange=ymax-ymin
      if (yrange.ne.0.d0) then
      y(1:ntot)=(y(1:ntot)-ymin)/yrange*2.d0-1.d0
      endif

! ----some arrays to allocate ----------------------------------------
      allocate(m00(ntot))
      allocate(xt(ndim,nt),yt(nt),xv(ndim,nv),yv(nv),stat=ierr)
      allocate(w(0:mnl,mnl,nhid+1),w_back(0:mnl,mnl,nhid+1),stat=ierr)
      allocate(w_save(0:mnl,mnl,nhid+1),stat=ierr)
      allocate(errt(0:nloop+1),errv(0:nloop+1),stat=ierr)
      allocate(zt(0:mnl,0:nhid+1,nt),stat=ierr) ! large
      allocate(dedw(0:mnl,0:nhid+1,nt),stat=ierr) ! same as zt
      allocate(zv(0:mnl,0:nhid+1,nv),stat=ierr)
      allocate(JAC(nt,nw),stat=ierr) ! largest matrix
      allocate(JTJ(nw,nw),stat=ierr) ! JTJ=JAC'*JAC
      allocate(JTJM(nw,nw),stat=ierr) ! JTJM=JTJ+mu*I
      allocate(dw(nw),dw_back(nw),stat=ierr)
      allocate(et(nt),ev(nv),stat=ierr)
      allocate(ipiv(nw),stat=ierr)
      allocate(index_table(nw,3),stat=ierr)
      write(*,*) "totally allocated memory: ",(nhid+2+nw)/2+
     &(ndim+1)*nmax+3*ndim+(ndim+1)*ntot+(mnl+1)*mnl*(nhid+1)*3+
     &(nloop+2)*2+(mnl+1)*(nhid+2)*nt*2+(mnl+1)*(nhid+2)*nv+
     &nt*nw+nw*nw*2+nw*2+ntot+nw*3/2,' words'
      return
      end

! --------------------------------------------------------------------
      subroutine deallocate_all
      use nnmod
      implicit none
      deallocate(nl)                ! (nhid+2)*int
      deallocate(x,y)               ! (ndim+1)*nmax*word
      deallocate(xmin,xmax,xrange)  ! 3*ndim*word
      deallocate(m00)               ! ntot*logical
      deallocate(xt,yt,xv,yv)       ! (ndim+1)*ntot*word
      deallocate(w,w_back,w_save)   ! (mnl+1)*mnl*(nhid+1)*3*word
      deallocate(errt,errv)         ! (nloop+2)*2*word
      deallocate(zt)                ! (mnl+1)*(nhid+2)*nt*word
      deallocate(dedw)              ! (mnl+1)*(nhid+2)*nt*word
      deallocate(zv)                ! (mnl+1)*(nhid+2)*nv*word
      deallocate(JAC)               ! nt*nw*word
      deallocate(JTJ)               ! nw*nw*word
      deallocate(JTJM)              ! nw*nw*word
      deallocate(dw,dw_back)        ! nw*2*word
      deallocate(et,ev)             ! ntot*word
      deallocate(ipiv)              ! nw*int
      deallocate(index_table)       ! nw*3*int
! totally allocated memory:
!     ntot logical
!     (nhid+2+nw) integer
!     (ndim+1)*nmax+3*ndim+(ndim+1)*ntot+(mnl+1)*mnl*(nhid+1)*3+(nloop+2)*2+(mnl+1)*(nhid+2)*nt*2+(mnl+1)*(nhid+2)*nv+nt*nw+nw*nw*2+nw*2+nt word
      return
      end

! --------------------------------------------------------------------
      subroutine random_divide(icycle)
      use nnmod
      implicit none
      logical :: mtmp
      integer :: i,j,k
      real*8 :: tmp
      integer :: icycle
      character*2 :: acycle
      write(acycle,'(i2.2)') icycle
! x y are kept unchanged (normalized)
! refresh and save m00
! refresh xt yt, xv yv
 
      m00(1:nt)=.true.
      m00(nt+1:nt+nv)=.false.
 
! reorder randomly
      do i=1,ntot-1
        call random_number(tmp); j=floor(tmp*(ntot-i))+i
c       j=i !!!
        mtmp=m00(j)
        m00(j)=m00(i)
        m00(i)=mtmp
      enddo
! refresh training set and validation set
      j=0
      k=0
      do i=1,ntot
      if(m00(i).eq..true.) then
      j=j+1
      xt(1:ndim,j)=x(1:ndim,i)  ! training set
      yt(j)=y(i)
      else
      k=k+1
      xv(1:ndim,k)=x(1:ndim,i)  ! validation set
      yv(k)=y(i)
      endif
      enddo
c     write(*,'(" Training set ",i5," , Validation set ",i5)') j,k
! save which points are used as train set
      open(101,file='M'//acycle)
      do i=1,ntot
      if(m00(i).eq..true.) then
      write(101,'(i1)') 1
      else
      write(101,'(i1)') 0
      endif
      enddo
      close(101)

      return
      end

! --------------------------------------------------------------------
      subroutine init_zt_zv
! zt stores input and output of each neuron for each training sample
! zt(0:mnl,0:nhid+1,nt)
! after randomly selecting training set, zt should be initialized
! put xt(1:ndim,1:nt) into zt(1:ndim,0,1:nt)
! put biases=1.d0 into zt(0,0:nhid+1,1:nt)
! the same for validation set zv
      use nnmod
      implicit none

      zt(1:ndim,0,1:nt)=xt(1:ndim,1:nt)
      zt(0,0:nhid+1,1:nt)=1.d0

      zv(1:ndim,0,1:nv)=xv(1:ndim,1:nv)
      zv(0,0:nhid+1,1:nv)=1.d0

      return
      end

! --------------------------------------------------------------------
      subroutine savenet(icycle,nhid,nl,mnl,w,xmin,xmax,ymin,ymax,rmse)
      implicit none
      integer icycle,nhid,nl(0:nhid+1),mnl
      real*8 w(0:mnl,mnl,nhid+1),xmin(nl(0)),xmax(nl(0)),ymin,ymax,rmse
      character*2 cc
      integer ilayer,jneuron,jp,i

      write(cc,'(i2.2)') icycle
      open(123,file='W'//cc//'.txt',status='unknown')

      ! write NN structure

c     write(123,'(1x,i3.1,/)') nhid       !!!!
c     write(123,'(1x,i3.1,\)') nl(0)      !!!!
c     do i=1,nhid+1                       !!!!
c       write(123,'(" ",i3.1,\)') nl(i)   !!!!
c     enddo                               !!!!
c     write(123,*) ""                     !!!!

      do ilayer=1,nhid+1

        ! write weights
        do jneuron=1,nl(ilayer)
        do jp=1,nl(ilayer-1)
          write(123,*) w(jp,jneuron,ilayer)
        enddo
        enddo
        write(123,*) ""

        ! write biases
        do jneuron=1,nl(ilayer)
          write(123,*) w(0,jneuron,ilayer)
        enddo
        write(123,*) ""

      enddo

      ! write xminmax
      do i=1,nl(0)
        write(123,*) xmin(i),xmax(i)
      enddo
      write(123,*) ""

      ! write yminmax
      write(123,*) ymin,ymax
      write(123,*) ""

      ! write transfer functions
      do i=1,nhid
        write(123,'(a8,\)') 'tansig'
      enddo
      write(123,*) ""

      ! write NN structure
      write(123,'(1x,i3.1,\)') nl(0)
      do i=1,nhid+1
        write(123,'(" -",i3.1,\)') nl(i)
      enddo
      write(123,*) ""

      write(123,"('rmse=',f12.4,' meV')") rmse
      close(123)

      return
      end subroutine savenet

! --------------------------------------------------------------------
      function random(xmin,xmax)
      implicit none
      real*8 random,xmin,xmax
      call random_number(random)
      random=random*(xmax-xmin)+xmin
      return
      end

! --------------------------------------------------------------------
      subroutine init_wb
      use nnmod
      implicit none
      real*8,external :: random
      integer i,j,k,inw

      index_table=0
      w=0.d0

      inw=0
      do i=1,nhid+1
      do j=1,nl(i)
      do k=0,nl(i-1)
        inw=inw+1
        w(k,j,i)=random(-1.d0,1.d0)
        index_table(inw,1:3)=[k,j,i]
      enddo
      enddo
      enddo

      return
      end

! --------------------------------------------------------------------
      subroutine feedforward(s)
      use nnmod
      implicit none
      real*8 tmp
      integer ilayer,jneuron,ksample
      character*1 s
! parameters to use: nhid,nl,w,zt,nt

! zt(0:mnl,0:nhid+1,1:nt)
! w(0:nl(i-1),1:nl(i),i) ilayer=1:nhid+1

!c$omp parallel default(shared)
!c$omp& private(ilayer,jneuron,ksample,tmp)
!c$omp do
      do ilayer=1,nhid+1
      do jneuron=1,nl(ilayer)
      do ksample=1,nt
        tmp=dot_product(   w(0:nl(ilayer-1),jneuron,ilayer)   ,
     &                    zt(0:nl(ilayer-1),ilayer-1,ksample) )
        if (ilayer.le.nhid) then
          zt(jneuron,ilayer,ksample)=dtanh(tmp) ! hidden layers, transferFcn = tanh
        else
          zt(jneuron,ilayer,ksample)=tmp ! output layer, transferFcn = purelin
        endif
      enddo
      enddo
      enddo
!c$omp end do nowait
!c$omp end parallel

! s=='t' only calculate training set
! s=='a' calculate training set and validation set
      if(s.eq.'t') return

! for validation set
c$omp parallel default(shared)
!c$omp& private(ilayer,jneuron,ksample,tmp)
c$omp do
      do ilayer=1,nhid+1
      do jneuron=1,nl(ilayer)
      do ksample=1,nv
        tmp=dot_product(  w(0:nl(ilayer-1),jneuron,ilayer)   ,
     &                   zv(0:nl(ilayer-1),ilayer-1,ksample) )
        if (ilayer.le.nhid) then
          zv(jneuron,ilayer,ksample)=dtanh(tmp)
        else
          zv(jneuron,ilayer,ksample)=tmp
        endif
      enddo
      enddo
      enddo
c$omp end do nowait
c$omp end parallel

      return
      end

! --------------------------------------------------------------------
      subroutine performance(iepoch)
      use nnmod
      implicit none
      integer iepoch,i
      real*8 rmse

      et(1:nt)=zt(1,nhid+1,1:nt)-yt(1:nt)
      rmse=dsqrt(dot_product(et,et)/nt)*yrange/2.d0*1000.d0
      errt(iepoch)=rmse

      ev(1:nv)=zv(1,nhid+1,1:nv)-yv(1:nv)
      if(nv.ge.1) then
      rmse=dsqrt(dot_product(ev,ev)/nv)*yrange/2.d0*1000.d0
      else
      rmse=0.d0
      endif
      errv(iepoch)=rmse

      return
      end

! --------------------------------------------------------------------
      subroutine backward_propagation
      use nnmod
      implicit none
      integer ilayer,jneuron,ksample,jn,jp
      integer i,j,id
      real*8 tmp
! parameters to use: dedw JAC JTJ
! refresh dedw usint backward propagation algorithm
! for output layer
      dedw(1,nhid+1,1:nt)=-1.d0

! for hidden layers
      do ilayer=nhid,1,-1
      do jneuron=1,nl(ilayer)
c$omp parallel default(shared)
c$omp& private(ksample,tmp)
c$omp do
      do ksample=1,nt
        tmp=dot_product( dedw(1:nl(ilayer+1),ilayer+1,ksample)  ,
     &                      w(jneuron,1:nl(ilayer+1),ilayer+1)  )
      dedw(jneuron,ilayer,ksample)=tmp*(1-zt(jneuron,ilayer,ksample)**2)
      enddo
c$omp end do nowait
c$omp end parallel
      enddo
      enddo

! save the 3-dimension matrix dedw into the 2-dimension matrix JAC

c$omp parallel default(shared)
c$omp& private(ksample,id,jp,jneuron,ilayer)
c$omp do
      do id=1,nw
      do ksample=1,nt
        call locate(id,jp,jneuron,ilayer)
        JAC(ksample,id)=dedw(jneuron,ilayer,ksample)
     &                 *zt(jp,ilayer-1,ksample)
      enddo
      enddo
c$omp end do nowait
c$omp end parallel

! calculate JAC'*JAC, most time-consuming step
      call dsyrk('u','t',nw,nt,1.0d0,JAC,nt,0.0d0,JTJ,nw) !! blas
      do i=2,nw
      do j=1,i-1
      JTJ(i,j)=JTJ(j,i)
      enddo
      enddo

! calculate dw from Newton's method
      call dgemv('t',nt,nw,1.d0,JAC,nt,et,1,0.d0,dw,1) !! blas

      return
      end

! --------------------------------------------------------------------
      subroutine locate(id,jp,jneuron,ilayer)
!                         V  V  V
c     subroutine locate(nhid,nl,id,jp,jneuron,ilayer)
!                                   V    V      V
! nhid and nl defines the structure of NN, totally nw parameters
! these nw parameters are stored in w(ia,ib,ilayer) and mapped into (nw) line
! provide the location id, return jp,jneuron,ilayer in w(0:jp,jneuron,ilayer)
      use nnmod
      implicit none
c     integer nhid
      integer id,jp,jneuron,ilayer,xx
c     integer nl(0:nhid+1)

      jp=index_table(id,1)
      jneuron=index_table(id,2)
      ilayer=index_table(id,3)

      return
      end

! --------------------------------------------------------------------
      subroutine get_delta_w(mu)
      use nnmod
      implicit none
      real*8 mu
      integer i,info

      JTJM(1:nw,1:nw)=JTJ(1:nw,1:nw)
      do i=1,nw
      JTJM(i,i)=JTJM(i,i)+mu
      enddo

      call dgesv(nw,1,JTJM,nw,ipiv,dw,nw,info) !! lapack
      if(info.ne.0) stop 'delta_w solve failed in dgesv'
      return
      end

! --------------------------------------------------------------------
      subroutine update_w
      use nnmod
      implicit none
      integer id,jp,jneuron,ilayer
      !w=w+dw
!c$omp parallel default(shared)
!c$omp& private(id,jp,jneuron,ilayer)
!c$omp do
      do id=1,nw
      call locate(id,jp,jneuron,ilayer)
      w(jp,jneuron,ilayer)=w(jp,jneuron,ilayer)+dw(id)
      enddo
!c$omp end do nowait
!c$omp end parallel
      return
      end

! --------------------------------------------------------------------
