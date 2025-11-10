subroutine spotdriva_gp(ns,rstore,enout)
  ! number of geometries: ns
  ! input:  r in Bohr, atoms in order of O H H H
  ! output: v in Hartree, v(OH+H2)=0
  implicit none
  real*8,parameter   :: eV=27.21138505d0
  integer,parameter  :: ndim=6
  integer,intent(in) :: ns
  real*4,intent(in)  :: rstore(ndim,ns)
  real*4,intent(out) :: enout(ns)

  real*8 :: xs(ndim,ns),ys(ns)
  integer :: i

  ! transfer input into double precision
  xs(1:ndim,1:ns)=rstore(1:ndim,1:ns)

  ! sort according to OH bond length
  do i=1,ns
      call sort_dist_oh3(xs(1:ndim,i))
  enddo

  ! do Gaussian Process prediction
  ! note: ( ns = nperbatch ) >= 1000 is most efficient
  call pes_gp(ndim,ns,xs,ys)

  ! transfer output into single precision
  enout=ys/eV

  return
end subroutine

subroutine sort_dist_oh3(r)
  implicit none
  real*8,intent(inout)  :: r(6)
  integer i,j
  real*8 tmp
  do i=1,3
  do j=i+1,3
    if(r(i).gt.r(j)) then
      tmp=r(i)
      r(i)=r(j)
      r(j)=tmp
      tmp=r(7-i)
      r(7-i)=r(7-j)
      r(7-j)=tmp
    endif
  enddo
  enddo
  return
end subroutine sort_dist_oh3

subroutine pes_gp(n0,ns,xs0,ymu)
  implicit none
  ! --- variables used for a PES using GP model --------------------------------
  integer,save :: ndim,nt,n_mean,n_cov,covFunc_d
  real*8,allocatable,save :: x(:,:),y(:),hyp_mean(:),hyp_cov(:)
  real*8,save :: hyp_lik
  character(kind=1,len=20),save :: infFunc,meanFunc,covFunc,likFunc,xhandle
  real*8,allocatable,save :: alpha(:),sW(:),L(:,:)
  integer,save :: init=0
  character(kind=1,len=80),parameter :: finput="/home/chenjun/oh3pes/g-gaussian-process/1-training/GPinput.txt"
  ! ----------------------------------------------------------------------------
  character(kind=1,len=80) :: CC ! Comments
  integer :: i,j,fid
  ! --- variables for unknown geometries, input @ xs, output @ ymu -------------
  integer,intent(in) :: n0,ns
  real*8,intent(in)  :: xs0(n0,ns)
  real*8,intent(out) :: ymu(ns)
  real*8 :: fmu(ns)
  real*8 :: xs(n0,ns),ys(ns),ys2(ns),fs2(ns),lp(ns)
  ! --- loading variables for gaussian process at first call -------------------
  if(init.eq.0) then
    fid=101010
    open(fid,file=trim(finput),status='old',action='read')
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

    allocate(x(ndim,nt),y(nt),hyp_mean(max(1,n_mean)),hyp_cov(n_cov),alpha(nt),sW(nt),L(nt,nt))

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

    if(trim(infFunc).eq.'infExact') then
        call infExact(n_mean,n_cov,hyp_mean,hyp_cov,hyp_lik,&
              meanFunc,covFunc,covFunc_d,likFunc,ndim,nt,x,y,&
              alpha,sW,L)
    else
        print*,"Unknown infFunc: ",trim(infFunc)
    endif

    do i=2,nt
    do j=1,i-1
      L(i,j)=L(j,i)
    enddo
    enddo

    init=1
  endif
  ! --- end of loading variables for gaussian process --------------------------

  ! --- do prediction using Gaussian Process -----------------------------------

  if(n0.ne.ndim) then
    print*,"input xs0 dimension wrong"
    stop
  endif
  xs=xs0

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

  call gp_prediction(n_mean,n_cov,hyp_mean,hyp_cov,hyp_lik,&
            infFunc,meanFunc,covFunc,covFunc_d,likFunc,&
            ndim,nt,x,y,ns,xs,ymu,ys2,fmu,fs2,lp,alpha,sW,L)

  ! ----------------------------------------------------------------------------
  return
end subroutine pes_gp

subroutine gp_prediction(&
      n_mean,n_cov,hyp_mean,hyp_cov,hyp_lik,&
      infFunc,meanFunc,covFunc,covFunc_d,likFunc,&
      ndim,nt,x,y,ns,xs,ymu,ys2,fmu,fs2,lp,alpha,sW,L)

  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-28
  implicit none
  integer,intent(in) :: n_mean,n_cov
  real*8,intent(in) :: hyp_mean(n_mean),hyp_cov(n_cov),hyp_lik
  character(kind=1,len=20) :: infFunc,meanFunc,covFunc,likFunc
  integer,intent(in) :: covFunc_d
  integer,intent(in) :: ndim,nt,ns
  real*8,intent(in)  :: x(ndim,nt),y(nt),xs(ndim,ns)
  real*8,intent(in) :: alpha(nt),sW(nt),L(nt,nt) !! new
  real*8,intent(out) :: ymu(ns),ys2(ns),fmu(ns),fs2(ns),lp(ns)

  integer,parameter :: nperbatch=20*19*19
  real*8 :: xp(ndim,nperbatch),Kss(nperbatch),Ks(nt,nperbatch),ms(nperbatch),&
            ymup(nperbatch),ys2p(nperbatch),fmup(nperbatch),fs2p(nperbatch),lpp(nperbatch),&
            V(nt,nperbatch),Lt(nt,nt),ys(nperbatch)
  integer :: nact,ifirst,ilast,iflag,i,j,ipiv(nt),info

  ! integer,parameter :: lwork=107648
  ! real*8 :: work(lwork)

  integer,parameter :: lwork=-1
  real*8 :: work(1)

  ! if(trim(infFunc).eq.'infExact') then
  !     call infExact(n_mean,n_cov,hyp_mean,hyp_cov,hyp_lik,&
  !           meanFunc,covFunc,covFunc_d,likFunc,ndim,nt,x,y,&
  !           alpha,sW,L)
  ! else
  !     print*,"Unknown infFunc: ",trim(infFunc)
  ! endif

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
      goto 2345 !!! only calculate fmu, set fmu as output !!! NOTICE ===========

      !% V = L'\([sW...]*Ks)
      do j=1,nperbatch
      do i=1,nt
        V(i,j)=Ks(i,j)*sw(i)
      enddo
      enddo

      Lt=L !! -- Lt should contain lower triangle and diagonal of L, or Lt=L'
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

      2345 continue
      nact=ilast
  enddo

  ymu=fmu !! 2015/11/30
  return
end subroutine gp_prediction

subroutine infExact(    n_mean,n_cov,hyp_mean,hyp_cov,hyp_lik,&
                        meanFunc,covFunc,covFunc_d,likFunc,ndim,nt,x,y,&
                        post_alpha,post_sW,post_L)
  ! Exact inference for a GP with Gaussian likelihood. Compute a parametrization
  ! of the posterior, the negative log marginal likelihood and its derivatives
  ! w.r.t. the hyperparameters. See also "help infMethods".
  !
  ! Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2015-07-13.
  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-28
  ! ==> derivatives not implemented now
  implicit none
  integer,intent(in) :: n_mean,n_cov,ndim,nt,covFunc_d
  real*8,intent(in)  :: hyp_mean(n_mean),hyp_cov(n_cov),hyp_lik,x(ndim,nt),y(nt)
  character(kind=1,len=20),intent(in) :: meanFunc,covFunc,likFunc
  real*8,intent(out) :: post_alpha(nt),post_sW(nt),post_L(nt,nt) !! ??? unknown size

  real*8 :: K(nt,nt),m(nt),L(nt,nt),sn2,sl,ym(nt,1)
  integer :: i,info

  !! Exact inference only possible with Gaussian likelihood
  if(trim(likFunc).ne.'likGauss') then
      print*,"Exact inference only possible with Gaussian likelihood"
      stop
  endif
  !! evaluate covariance matrix
  if(trim(covFunc).eq.'covMaternard') then
      call covMaternard(covFunc_d,hyp_cov,ndim,x,nt,x,nt,0,K)
  elseif(trim(covFunc).eq.'covMaterniso') then
      call covMaterniso(covFunc_d,hyp_cov,ndim,x,nt,x,nt,0,K)
  else
      print*,"Unknown covFunc: ",trim(covFunc)
      stop
  endif
  !! evaluate mean vector
  if(trim(meanFunc).eq.'meanZero') then
      call meanZero(ndim,hyp_mean,x,nt,0,m)
  elseif(trim(meanFunc).eq.'meanOne') then
      call meanOne(ndim,hyp_mean,x,nt,0,m)
  elseif(trim(meanFunc).eq.'meanConst') then
      call meanConst(ndim,hyp_mean,x,nt,0,m)
  elseif(trim(meanFunc).eq.'meanLinear') then
      call meanLinear(ndim,hyp_mean,x,nt,0,m)
  else
      print*,"Unknown meanFunc: ",trim(meanFunc)
      stop
  endif

  !! noise variance of likGauss
  sn2 = exp(2.d0*hyp_lik)

  if(sn2.lt.1.d-6) then !! very tiny sn2 can lead to numerical trouble
      ! Cholesky factor of covariance with noise
      ! L = -inv(K+inv(sW^2))
      L=K
      do i=1,nt
        L(i,i)=L(i,i)+sn2
      enddo
      call dpotrf('U',nt,L,nt,info)
      if(info.ne.0) stop "Cholesky factorization failed 1"
      sl=1.d0
      post_L=0.d0
      do i=1,nt
        post_L(i,i)=1.d0
      enddo
      call dpotrs('U',nt,nt,L,nt,post_L,nt,info)
      if(info.ne.0) stop "Solve Cholesky factorization failed 1"
      post_L=-1.d0*post_L

  else
      L=K/sn2
      do i=1,nt
        L(i,i)=L(i,i)+1.d0
      enddo
      call dpotrf('U',nt,L,nt,info)
      if(info.ne.0) stop "Cholesky factorization failed 2"
      sl=sn2
      post_L=L
  endif

  post_alpha=y-m
  call dpotrs('U',nt,1,L,nt,post_alpha,nt,info)
  if(info.ne.0) stop "Solve Cholesky factorization failed 2"
  post_alpha = post_alpha/sl
  post_sW = 1.d0/dsqrt(sn2) ! sqrt of noise precision vector

  !! to calculate derivatives:
  !% if nargout>1                               % do we want the marginal likelihood?
  !%   nlZ = (y-m)'*alpha/2 + sum(log(diag(L))) + n*log(2*pi*sl)/2;   % -log marg lik
  !%   if nargout>2                                         % do we want derivatives?
  !%     dnlZ = hyp;                                 % allocate space for derivatives
  !%     Q = solve_chol(L,eye(n))/sl - alpha*alpha';     % precompute for convenience
  !%     for i = 1:numel(hyp.cov)
  !%       dnlZ.cov(i) = sum(sum(Q.*feval(cov{:}, hyp.cov, x, [], i)))/2;
  !%     end
  !%     dnlZ.lik = sn2*trace(Q);
  !%     for i = 1:numel(hyp.mean)
  !%       dnlZ.mean(i) = -feval(mean{:}, hyp.mean, x, i)'*alpha;
  !%     end
  !%   end
  !% end

  return
end subroutine infExact

subroutine covMaternard(d, hyp, ndim, x, nx, z, nz, iderivatives, K)
  ! Matern covariance function with nu = d/2 and with Automatic Relevance
  ! Determination (ARD) distance measure. For d=1 the function is also known as
  ! the exponential covariance function or the Ornstein-Uhlenbeck covariance 
  ! in 1d. The covariance function is:
  !
  !   k(x^p,x^q) = sf^2 * f( sqrt(d)*r ) * exp(-sqrt(d)*r)
  !
  ! with f(t)=1 for d=1, f(t)=1+t for d=3 and f(t)=1+t+t²/3 for d=5.
  ! Here r is the distance sqrt((x^p-x^q)'*inv(P)*(x^p-x^q)), where the P matrix
  ! is diagonal with ARD parameters ell_1^2,...,ell_D^2, where D is the dimension
  ! of the input space and sf2 is the signal variance. The hyperparameters are:
  !
  ! hyp = [ log(ell_1)
  !         log(ell_2)
  !          ..
  !         log(ell_D)
  !         log(sf) ]
  !
  ! Copyright (c) by Hannes Nickisch, 2013-10-13.
  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-27
  implicit none
  integer,intent(in) :: ndim,nx,nz,d,iderivatives
  real*8,intent(in)  :: hyp(ndim+1),x(ndim,nx),z(ndim,nz)
  real*8,intent(out) :: K(nx,nz)
  real*8 :: ell(ndim),sf2,kt(nx,nz),xt(ndim,nx),zt(ndim,nz),ki(nx,nz)
  integer :: i,j
  do i=1,ndim
    ell(i) = exp(hyp(i))
  enddo
  sf2 = exp(2.d0*hyp(ndim+1))

  if(d.ne.1 .and. d.ne.3 .and. d.ne.5) then
    print*,'only 1, 3 and 5 allowed for d'
    stop
  endif
  if(iderivatives.lt.0 .or. iderivatives.gt.ndim+1) then
    print*,'only 0 to ndim+1 allowed for iderivatives'
    stop
  endif
  ! precompute distances
  do j=1,nx
  do i=1,ndim
    xt(i,j)=dsqrt(d*1.d0)/ell(i)*x(i,j)
  enddo
  enddo

  do j=1,nz
  do i=1,ndim
    zt(i,j)=dsqrt(d*1.d0)/ell(i)*z(i,j)
  enddo
  enddo

  call sq_dist(ndim,xt,nx,zt,nz,K)
  do i=1,nx
  do j=1,nz
      K(i,j)=dsqrt(K(i,j))
  enddo
  enddo

  !m = @(t,f) f(t).*exp(-t); dm = @(t,f) df(t).*exp(-t);
  if (iderivatives.eq.0) then ! covariances
    call f_covMaternard(K,kt,nx,nz,d,0)
    do i=1,nx
    do j=1,nz
      K(i,j)=sf2*kt(i,j)*exp(-K(i,j))
    enddo
    enddo
  else                        ! derivatives
    if (iderivatives.le.ndim) then              ! length scale parameter
      call sq_dist(1,dsqrt(d*1.d0)/ell(iderivatives)*x(iderivatives,1:nx),nx,dsqrt(d*1.d0)/ell(iderivatives)*z(iderivatives,1:nz),nz,ki);
      call f_covMaternard(K,kt,nx,nz,d,1)
      do i=1,nx
      do j=1,nz
        K(i,j) = sf2*kt(i,j)*exp(-K(i,j))*Ki(i,j)
        if(Ki(i,j).le.1.d-12) K(i,j)=0.d0
      enddo
      enddo
    elseif (iderivatives.eq.(ndim+1)) then      ! magnitude parameter
      call f_covMaternard(K,kt,nx,nz,d,0)
      do i=1,nx
      do j=1,nz
        K(i,j)=2.d0*sf2*kt(i,j)*exp(-K(i,j))
      enddo
      enddo
    else
      print*,'Unknown hyperparameter'
      stop
    endif
  endif

  return
end subroutine covMaternard

subroutine f_covMaternard(t,f,na,nb,d,k)
  ! calculate f(t) or df(t), df(t)=(f(t)-f'(t))/t
  ! case 1, f = @(t) 1;               df = @(t) 1./t;
  ! case 3, f = @(t) 1 + t;           df = @(t) 1;
  ! case 5, f = @(t) 1 + t.*(1+t/3);  df = @(t) (1+t)/3;
  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-27
  implicit none
  integer,intent(in) :: na,nb,d,k
  real*8,intent(in)  :: t(na,nb)
  real*8,intent(out) :: f(na,nb)
  integer :: i,j
  if(d.ne.1 .and. d.ne.3 .and. d.ne.5) then
    print*,'only 1, 3 and 5 allowed for d'
    stop
  endif
  if(k.ne.0 .and. k.ne.1) then
    print*,'only 0 and 1 allowed for k, to calculate f or df'
    stop
  endif

  if (d.eq.1 .and. k.eq.0) then
    f=1.d0
  elseif (d.eq.3 .and. k.eq.0) then
    f=t+1.d0
  elseif (d.eq.5 .and. k.eq.0) then
    do i=1,na
    do j=1,nb
      f(i,j)=1.d0+t(i,j)*(1.d0+t(i,j)/3.d0)
    enddo
    enddo
  elseif (d.eq.1 .and. k.eq.1) then
    do i=1,na
    do j=1,nb
      f(i,j)=1.d0/t(i,j)
    enddo
    enddo
  elseif (d.eq.3 .and. k.eq.1) then
    f=1.d0
  elseif (d.eq.5 .and. k.eq.1) then
    do i=1,na
    do j=1,nb
      f(i,j)=(1.d0+t(i,j))/3.d0
    enddo
    enddo
  endif
  return
end subroutine f_covMaternard

subroutine covMaterniso(d, hyp, ndim, x, nx, z, nz, iderivatives, K)
  ! Matern covariance function with nu = d/2 and isotropic distance measure. For
  ! d=1 the function is also known as the exponential covariance function or the 
  ! Ornstein-Uhlenbeck covariance in 1d. The covariance function is:
  !
  !   k(x^p,x^q) = sf^2 * f( sqrt(d)*r ) * exp(-sqrt(d)*r)
  !
  ! with f(t)=1 for d=1, f(t)=1+t for d=3 and f(t)=1+t+t²/3 for d=5.
  ! Here r is the distance sqrt((x^p-x^q)'*inv(P)*(x^p-x^q)), P is ell times
  ! the unit matrix and sf2 is the signal variance. The hyperparameters are:
  !
  ! hyp = [ log(ell)
  !         log(sf) ]
  !
  ! Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-26
  implicit none
  integer,intent(in) :: ndim,nx,nz,d,iderivatives
  real*8,intent(in)  :: hyp(2),x(ndim,nx),z(ndim,nz)
  real*8,intent(out) :: K(nx,nz)
  real*8 :: ell,sf2,kt(nx,nz)
  integer :: i,j 
  ell = exp(hyp(1))
  sf2 = exp(2.d0*hyp(2))

  if(d.ne.1 .and. d.ne.3 .and. d.ne.5) then
    print*,'only 1, 3 and 5 allowed for d'
    stop
  endif
  if(iderivatives.ne.0 .and. iderivatives.ne.1 .and. iderivatives.ne.2) then
    print*,'Unknown hyperparameter'
    print*,'only 0, 1 and 2 allowed for iderivatives'
    stop
  endif

  call sq_dist(ndim,dsqrt(d*1.d0)/ell*x,nx,dsqrt(d*1.d0)/ell*z,nz,K)
  do i=1,nx
  do j=1,nz
      K(i,j)=dsqrt(K(i,j))
  enddo
  enddo

  if (iderivatives.eq.0) then     ! covariances
    call f_covMaterniso(K,kt,nx,nz,d,0)
    do i=1,nx
    do j=1,nz
      K(i,j) = sf2*kt(i,j)*exp(-K(i,j))
    enddo
    enddo
  elseif (iderivatives.eq.1) then ! derivatives
    call f_covMaterniso(K,kt,nx,nz,d,1)
    do i=1,nx
    do j=1,nz
      K(i,j) = sf2*kt(i,j)*exp(-K(i,j))*K(i,j)
    enddo
    enddo
  elseif (iderivatives.eq.2) then
    call f_covMaterniso(K,kt,nx,nz,d,0)
    do i=1,nx
    do j=1,nz
      K(i,j) = 2.d0*sf2*kt(i,j)*exp(-K(i,j))
    enddo
    enddo
  else
    print*,'Unknown hyperparameter'
    stop
  endif
  return
end subroutine covMaterniso

subroutine f_covMaterniso(t,f,na,nb,d,k)
  ! calculate f(t) or df(t), df(t)=f(t)-f'(t)
  ! if d=1: f = 1; df = 1;
  ! if d=3: f = 1+t; df = t;
  ! if d=5: f = 1 + t.*(1+t/3); df = t.*(1+t)/3;
  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-26
  implicit none
  integer,intent(in) :: na,nb,d,k
  real*8,intent(in)  :: t(na,nb)
  real*8,intent(out) :: f(na,nb)
  integer :: i,j
  if(d.ne.1 .and. d.ne.3 .and. d.ne.5) then
    print*,'only 1, 3 and 5 allowed for d'
    stop
  endif
  if(k.ne.0 .and. k.ne.1) then
    print*,'only 0 and 1 allowed for k, to calculate f or df'
    stop
  endif
  if (d.eq.1) then
    f=1.d0
  elseif (d.eq.3) then
    f=t
    if(k.eq.0) f=f+1.d0
  elseif (d.eq.5) then
    if(k.eq.0) then
      do i=1,na
      do j=1,nb
        f(i,j)=1.d0+t(i,j)*(1.d0+t(i,j)/3.d0)
      enddo
      enddo
    elseif(k.eq.1) then
      do i=1,na
      do j=1,nb
        f(i,j)=t(i,j)*(1.d0+t(i,j))/3.d0
      enddo
      enddo
    endif
  endif
  return
end subroutine f_covMaterniso

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

subroutine meanLinear(ndim,hyp,x,nx,iderivatives,A)
  ! Linear mean function. The mean function is parameterized as:
  !
  ! A(x) = sum_i a_i * x_i;
  !
  ! The hyperparameter is:
  !
  ! hyp = [ a_1
  !         a_2
  !         ..
  !         a_D ]
  !
  ! Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-01-10.
  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-27
  implicit none
  integer,intent(in) :: ndim,nx,iderivatives
  real*8,intent(in)  :: hyp(ndim),x(ndim,nx)
  real*8,intent(out) :: A(nx)

  if(iderivatives.eq.0) then ! evaluate mean
    call dgemv('t',ndim,nx,1.d0,x,ndim,hyp,1,0.d0,A,1)
  elseif (iderivatives.le.ndim) then ! derivative
    A = x(iderivatives,1:nx)
  else
    A = 0.d0
  endif
  return
end subroutine meanLinear

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

subroutine likGauss_prediction(hyp,nt,y,mu,s2,lp,ymu,ys2)
  ! likGauss - Gaussian likelihood function for regression. The expression for the 
  ! likelihood is 
  !   likGauss(t) = exp(-(t-y)^2/2*sn^2) / sqrt(2*pi*sn^2),
  ! where y is the mean and sn is the standard deviation.
  !
  ! The hyperparameters are:
  !
  ! hyp = [  log(sn)  ]
  !
  ! Several modes are provided, for computing likelihoods, derivatives and moments
  ! respectively, see likFunctions.m for the details. In general, care is taken
  ! to avoid numerical issues when the arguments are extreme.
  !
  ! Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2015-07-13.
  ! Fortran by Jun Chen (Email: NJUChenJun@gmail.com), 2015-11-28

  implicit none
  real*8,intent(in)  :: hyp
  integer,intent(in) :: nt
  real*8,intent(in)  :: y(nt),mu(nt),s2(nt)
  real*8,intent(out) :: lp(nt),ymu(nt),ys2(nt)
  real*8,parameter :: pi=3.14159265358979323846d0
  real*8 :: sn2
  integer :: i
  sn2=exp(2.d0*hyp)

  !%lZ = -(y-mu).^2./(sn2+s2)/2 - log(2*pi*(sn2+s2))/2; % log part function
  do i=1,nt
    lp(i) = -(y(i)-mu(i))**2/(sn2+s2(i))/2.d0 - log(2.d0*pi*(sn2+s2(i)))/2.d0
  enddo

  ymu=mu
  ys2=s2+sn2
  return
end subroutine likGauss_prediction

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
