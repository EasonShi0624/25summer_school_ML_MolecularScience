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
  real*8,intent(out) :: post_alpha(nt),post_sW(nt),post_L(nt,nt)

! real*8 :: K(nt,nt),m(nt),L(nt,nt),sn2,sl,ym(nt,1)
  real*8 :: sn2,sl
  real*8,allocatable :: K(:,:),m(:),L(:,:),ym(:,:)

  integer :: i,info

  allocate(K(nt,nt),m(nt),L(nt,nt),ym(nt,1))

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

  deallocate(K,m,L,ym)
  return
end subroutine infExact
