subroutine gp_prediction(&
      n_mean,n_cov,hyp_mean,hyp_cov,hyp_lik,&
      infFunc,meanFunc,covFunc,covFunc_d,likFunc,&
      ndim,nt,x,y,ns,xs,ymu,ys2,fmu,fs2,lp)

  ! njuchenjun@gmail.com, 2015-11-28
  implicit none
  integer,intent(in) :: n_mean,n_cov
  real*8,intent(in) :: hyp_mean(n_mean),hyp_cov(n_cov),hyp_lik
  character(kind=1,len=20) :: infFunc,meanFunc,covFunc,likFunc
  integer,intent(in) :: covFunc_d
  integer,intent(in) :: ndim,nt,ns
  real*8,intent(in)  :: x(ndim,nt),y(nt),xs(ndim,ns)
  real*8,intent(out) :: ymu(ns),ys2(ns),fmu(ns),fs2(ns),lp(ns)

  real*8 :: alpha(nt),sW(nt),L(nt,nt)

  integer,parameter :: nperbatch=1000
! real*8 :: xp(ndim,nperbatch),Kss(nperbatch),Ks(nt,nperbatch),ms(nperbatch),&
!           ymup(nperbatch),ys2p(nperbatch),fmup(nperbatch),fs2p(nperbatch),lpp(nperbatch),&
!           V(nt,nperbatch),Lt(nt,nt),ys(nperbatch)
  real*8,allocatable :: xp(:,:),Kss(:),Ks(:,:),ms(:),&
                        ymup(:),ys2p(:),fmup(:),fs2p(:),lpp(:),&
                        V(:,:),Lt(:,:),ys(:)

  integer :: nact,ifirst,ilast,iflag,i,j,ipiv(nt),info

  integer,parameter :: lwork=1000
  real*8 :: work(lwork)


  allocate(xp(ndim,nperbatch),Kss(nperbatch),Ks(nt,nperbatch),ms(nperbatch),&
            ymup(nperbatch),ys2p(nperbatch),fmup(nperbatch),fs2p(nperbatch),lpp(nperbatch),&
            V(nt,nperbatch),Lt(nt,nt),ys(nperbatch))

  if(trim(infFunc).eq.'infExact') then
      call infExact(n_mean,n_cov,hyp_mean,hyp_cov,hyp_lik,&
            meanFunc,covFunc,covFunc_d,likFunc,ndim,nt,x,y,&
            alpha,sW,L)
  else
      print*,"Unknown infFunc: ",trim(infFunc)
  endif

  nact=0
  xp=0.d0
  ymu=0.d0; ys2=0.d0; fmu=0.d0; fs2=0.d0; lp=0.d0
  do while(nact.lt.ns)
      ifirst=nact+1; ilast=nact+nperbatch; iflag=0
      if(ilast.gt.ns) then
        iflag=1
      endif
      if(iflag.eq.0) then
        xp=xs(1:ndim,ifirst:ilast)
      else
        xp(1:ndim,1:(ns-nact))=xs(1:ndim,ifirst:ns)
      endif
      !! --- start gaussian process prediction
      ! evaluate covariance matrix
      if(trim(covFunc).eq.'covMaternard') then
          Kss=0.d0
          call covMaternard(covFunc_d,hyp_cov,ndim,x,nt,xp,nperbatch,0,Ks)
      elseif(trim(covFunc).eq.'covMaterniso') then
          Kss=0.d0
          call covMaterniso(covFunc_d,hyp_cov,ndim,x,nt,xp,nperbatch,0,Ks)
      else
          print*,"Unknown covFunc: ",trim(covFunc)
          stop
      endif
      ! evaluate mean vector
      if(trim(meanFunc).eq.'meanZero') then
          call meanZero(ndim,hyp_mean,xp,nperbatch,0,ms)
      elseif(trim(meanFunc).eq.'meanOne') then
          call meanOne(ndim,hyp_mean,xp,nperbatch,0,ms)
      elseif(trim(meanFunc).eq.'meanConst') then
          call meanConst(ndim,hyp_mean,xp,nperbatch,0,ms)
      elseif(trim(meanFunc).eq.'meanLinear') then
          call meanLinear(ndim,hyp_mean,xp,nperbatch,0,ms)
      else
          print*,"Unknown meanFunc: ",trim(meanFunc)
          stop
      endif

      !% fmup=ms+Ks'*alpha
      fmup=ms
      call dgemv('T',nt,nperbatch,1.d0,Ks,nt,alpha,1,1.d0,fmup,1)
      if(iflag.eq.0) then
        fmu(ifirst:ilast)=fmup !% predictive means
      else
        fmu(ifirst:ns)=fmup(1:(ns-nact))
      endif

      !% V = L'\([sW...]*Ks)
      do j=1,nperbatch
      do i=1,nt
        V(i,j)=Ks(i,j)*sw(i)
      enddo
      enddo
      Lt=0.d0
      do i=1,nt
      do j=1,i
        Lt(i,j)=L(j,i)
      enddo
      enddo
      call dsysv('L',nt,nperbatch,Lt,nt,ipiv,V,nt,work,lwork,info)
      if(info.ne.0) stop "Solve dsysv failed"

      !% fs2(id) = kss - sum(V.*V,1)'
      do i=1,nperbatch
        fs2p(i)=kss(i)
        do j=1,nt
          fs2p(i)=fs2p(i)-V(j,i)*V(j,i)
        enddo
        fs2p(i)=max(fs2p(i),0.d0) !% remove numerical noise i.e. negative variances
      enddo

      if(iflag.eq.0) then
        fs2(ifirst:ilast)=fs2p !% predictive variances
      else
        fs2(ifirst:ns)=fs2p(1:(ns-nact))
      endif

      if(trim(likFunc).eq.'likGauss') then
        ys=0.d0
        call likGauss_prediction(hyp_lik,nperbatch,ys,fmup,fs2p,  lpp,ymup,ys2p)
      else
        print*,"Unknown likFunc: ",trim(likFunc)
        stop
      endif

      if(iflag.eq.0) then
        ymu(ifirst:ilast)=ymup !% predictive mean ys|y
        ys2(ifirst:ilast)=ys2p !% predictive variance ys|y
         lp(ifirst:ilast)= lpp !% log probability; sample averaging
      else
        ymu(ifirst:ns)=ymup(1:(ns-nact))
        ys2(ifirst:ns)=ys2p(1:(ns-nact))
         lp(ifirst:ns)= lpp(1:(ns-nact))
      endif
      !! --- end gaussian process prediction

      nact=ilast
  enddo

  deallocate(xp,Kss,Ks,ms,ymup,ys2p,fmup,fs2p,lpp,V,Lt,ys)


  return
end subroutine gp_prediction
