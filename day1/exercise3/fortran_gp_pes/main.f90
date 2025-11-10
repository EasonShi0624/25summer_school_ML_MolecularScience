program main
  implicit none

  ! --- variables used for a PES using GP model --------------------------------
  integer,save :: ndim,nt,n_mean,n_cov,covFunc_d
  real*8,allocatable,save :: x(:,:),y(:),hyp_mean(:),hyp_cov(:)
  real*8,save :: hyp_lik
  character(kind=1,len=20),save :: infFunc,meanFunc,covFunc,likFunc,xhandle
  integer,save :: init=0
  ! ----------------------------------------------------------------------------
  character(kind=1,len=80) :: CC ! Comments
  integer :: i,j,fid

  integer :: ns
  real*8,allocatable :: xs(:,:),ys(:),ymu(:),ys2(:),fmu(:),fs2(:),lp(:)

  ! --- loading variables for gaussian process at first call -------------------
  if(init.eq.0) then
    fid=101
    open(fid,file='GPinput.txt',status='old',action='read')
    read(fid,*) CC
    read(fid,*) infFunc
    read(fid,*) meanFunc
    read(fid,*) covFunc
    if(trim(covFunc).eq.'covMaternard' .or. trim(covFunc).eq.'covMaterniso') read(fid,*) covFunc_d
    read(fid,*) likFunc

    read(fid,*) CC
    read(fid,*) ndim,nt

    if(trim(meanFunc).eq.'meanZero' .or. trim(meanFunc).eq.'meanOne') then
        n_mean=0 ! no need
    elseif(trim(meanFunc).eq.'meanConst') then
        n_mean=1
    elseif(trim(meanFunc).eq.'meanLinear') then
        n_mean=ndim
    else
        print*,"Unknown meanFunc: ",trim(meanFunc)
        stop
    endif

    if(trim(covFunc).eq.'covMaternard') then
        n_cov=ndim+1
    elseif(trim(covFunc).eq.'covMaterniso') then
        n_cov=2
    else
        print*,"Unknown covFunc: ",trim(covFunc)
    endif

    allocate(x(ndim,nt),y(nt),hyp_mean(max(1,n_mean)),hyp_cov(n_cov))
    read(fid,*) CC
    if(n_mean.gt.0) then
    do i=1,n_mean
        read(fid,*) hyp_mean(i)
    enddo
    endif

    read(fid,*) CC
    do i=1,n_cov
        read(fid,*) hyp_cov(i)
    enddo

    read(fid,*) CC
    read(fid,*) hyp_lik

    read(fid,*) CC
    read(fid,*) xhandle
    do i=1,nt
        read(fid,*) x(1:ndim,i),y(i)
    enddo

    if(trim(xhandle).eq.'normal') then
        ! do nothing
    elseif(trim(xhandle).eq.'reverse') then
        do i=1,nt
        do j=1,ndim
        x(j,i)=1.d0/x(j,i)
        enddo
        enddo
    elseif(trim(xhandle).eq.'log') then
        do i=1,nt
        do j=1,ndim
        x(j,i)=log(x(j,i))
        enddo
        enddo
    endif

    close(fid)
    init=1
  endif
  ! --- end of loading variables for gaussian process --------------------------

  ! --- do prediction using Gaussian Process -----------------------------------
  ns=1000
  allocate(xs(ndim,ns),ys(ns),ymu(ns),ys2(ns),fmu(ns),fs2(ns),lp(ns))
  fid=102
  open(fid,file='GPtest.txt',status='old',action='read')
  do i=1,ns
      read(fid,*) xs(1:ndim,i),ys(i)
  enddo
  if(trim(xhandle).eq.'normal') then
      ! do nothing
  elseif(trim(xhandle).eq.'reverse') then
      do i=1,ns
      do j=1,ndim
      xs(j,i)=1.d0/xs(j,i)
      enddo
      enddo
  elseif(trim(xhandle).eq.'log') then
      do i=1,ns
      do j=1,ndim
      xs(j,i)=log(xs(j,i))
      enddo
      enddo
  endif
  close(fid)

  call gp_prediction(n_mean,n_cov,hyp_mean,hyp_cov,hyp_lik,&
            infFunc,meanFunc,covFunc,covFunc_d,likFunc,&
            ndim,nt,x,y,ns,xs,ymu,ys2,fmu,fs2,lp)

  do i=1,ns
      print*,ys(i),ymu(i)
  enddo
  ! ----------------------------------------------------------------------------
end program
