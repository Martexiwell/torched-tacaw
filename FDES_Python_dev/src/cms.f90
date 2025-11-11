module cms

  implicit none

  double complex, allocatable :: prop(:,:)  ! propagator
  !double complex, allocatable :: trans_array(:,:,:) ! transmission function array, ONLY for TORCHED_TACAW

 contains

  !--------------------------------------------------------------------------------------------------------------------------
  subroutine init_cms(dz) ! allocate arrays and prepare propagator and a mask for bandwidth limiting

    use fdes_const, only: pi
    use fftw
    use global
    use atoms
  
    implicit none

    double precision dz

    integer ix, iy
    double precision kx, ky, kth2

    allocate(prop(ntx,nty))

    kth2 = min ( dble(nx)/lat(1), dble(ny)/lat(2) ) / 3.d0
    kth2 = kth2*kth2

    bw_mask = 1  ! the idea of bw_mask is that we can do X=X*bw_mask in k-space to bw-limit array
    prop = 0.d0

    do ix=1, ntx
      kx = getkx(ix)
      do iy=1, nty
        ky = getky(iy)
        if(bandlimit .and. (kx*kx+ky*ky)>kth2) then
          bw_mask(ix,iy) = 0
        else
          prop(ix,iy) = exp(-dcmplx(0,1)*pi*lambda*dz*(kx*kx+ky*ky))
        endif
      enddo
    enddo
!    call output_array(prop,'(a,i0,a)','P_',0,'.dat')

  end subroutine init_cms

  !--------------------------------------------------------------------------------------------------------------------------
  subroutine cms_step ! assuming that transmission function has been prepared, it applies transmission and propagator

    use global
    use fftw
    use potential

    implicit none

    wf = trans*wf         ! Real space
    call fft2d(wf,kwf)    ! FFT
    kwf = prop*kwf        ! Reciprocal space
    call bfft2d(kwf,wf)   ! FFTI

  end subroutine cms_step

  !--------------------------------------------------------------------------------------------------------------------------
  subroutine msfdes_step(isl) ! assuming that transmission function has been prepared, it applies transmission and propagator

    use global
    use fdes_const, only  : echarge, rplanck !, pi, a0_Ang, rydberg  !,eps0, a0,
    use atoms, only : I, natypes   ! I is the image I(x,y) of the slice built with sq_deposit
    use fftw
    use potential

    implicit none

    integer, intent(in) :: isl
    integer :: itype, ix, iy 
    double precision :: magfac = (echarge/rplanck)*1.d-20 ! this is e/hbar with a conversion Å^2/m^2

    !if(.not.allocated(trans_array))  allocate(trans_array(ntx,nty,ntz))

    allocate(fzk(ntx,nty),pot(ntx,nty),trans(ntx,nty),F(ntx,nty), I_cplx(ntx,nty))
    
    fzk=0.d0
    pot = 0.d0
    trans = 1.d0
    F=0.0d0

    !print *, 'msfdes_step: arrays allocated '

    !print *, 'msfdes_step: using natypes=', natypes
    do itype = 1, natypes
      !print *, ' Generating image for itype : ', itype, ' in slice ', isl
      
      ! electric component
      I_cplx = dcmplx(1.0d0, 0.0d0)*I(:,:,0, itype) ! Image from the current slice, electric component
      call fft2d(I_cplx,F)    ! FFT of image I, F is now in k space
      pot = Vk_grid(:,:,itype)
      fzk = fzk + F*pot*S_inv ! is this okay to reuse that much?

      ! z magnetic component of A
      ! mx times ky for magnetic contribution
      I_cplx = dcmplx(1.0d0, 0.0d0)*I(:,:,1, itype) ! Image from the current slice, x component of magnetic moment
      call fft2d(I_cplx,F)    ! FFT of image I, F is now in k space
      pot = (magfac/sigma)*Ak_grid(:,:,itype)
      !fzk = fzk - (magfac/sigma)*pot*( mxF*ky-myF*kx )
      do iy = 1, nty
        do ix = 1, ntx
          fzk(ix,iy) = fzk(ix,iy) - F(ix,iy)*ky_arr(iy)*pot(ix,iy)*S_inv(ix,iy) 
        end do
      end do

      !print *, 'msfdes_step: electric pot created' 

      ! z magnetic component of A
      ! my times kx for magnetic contribution
      I_cplx = dcmplx(1.0d0, 0.0d0)*I(:,:,2, itype) ! Image from the current slice, y component of magnetic moment
      call fft2d(I_cplx,F)    ! FFT of image I, F is now in k space
      !pot = (magfac/sigma)*Ak_grid(:,:,itype)
      !fzk = fzk - (magfac/sigma)*pot*( mxF*ky-myF*kx )
      do iy = 1, nty
        do ix = 1, ntx
          fzk(ix,iy) = fzk(ix,iy) + F(ix,iy)*kx_arr(ix)*pot(ix,iy)*S_inv(ix,iy)
        end do
      end do

    end do

    pot = 0.d0 ! pot will be used to store the real space potential
    
    !print *, 'fzk=', sum(fzk)
    !call bfft2d(F,pot)      ! FFTI to get proj pot in real space
    call bfft2d(fzk,pot)      ! FFTI to get proj pot in real space
    if(bandlimit) call bw_limit(pot) !Should I do bw_limit????
    !if (sum(I(:,:,isl,0,1)) > 0 .and. isl==print_psi_k) call output_array(pot,'(a,i0,a)','results/potAz_r_',isl,'_fdes.dat')
    !if (sum(I(:,:,isl,0,1)) > 0 .and. isl==print_psi_k) call output_array(pot,'(a,i0,a)','results/potE_r_',isl,'_fdes.dat')
    !if (isl==print_psi_k) call output_array(pot,'(a,i0,a)','results/potE_r_',isl,'_fdes.dat')
    trans = exp( dcmplx(0.d0,1.d0)*sigma*pot  )    ! I think moder fortran compilers allow this implementation!
    if(bandlimit) call bw_limit(trans) !Should I do bw_limit????
    !trans_array(:,:,isl) = trans
    wf = trans*wf         ! Real space
    call fft2d(wf,kwf)    ! FFT
    kwf = prop*kwf        ! Reciprocal space
    call bfft2d(kwf,wf)   ! FFTI

    deallocate(fzk,pot,trans,F,I_cplx)

  end subroutine msfdes_step

  !--------------------------------------------------------------------------------------------------------------------------
  subroutine msfdes_step_2D(isl) ! assuming that transmission function has been prepared, it applies transmission and propagator

    use global
    use fdes_const, only  : echarge, rplanck !, pi, a0_Ang, rydberg  !,eps0, a0,
    use atoms, only : I, natypes   ! I is the image I(x,y) of the slice built with sq_deposit
    use fftw
    use potential

    implicit none

    integer, intent(in) :: isl
    integer :: itype, ix, iy 
    double precision :: magfac = (echarge/rplanck)*1.d-20 ! this is e/hbar with a conversion Å^2/m^2

    !if(.not.allocated(trans_array))  allocate(trans_array(ntx,nty,ntz))

    allocate(fzk(ntx,nty),pot(ntx,nty),F(ntx,nty), I_cplx(ntx,nty))
    if (.not. allocated(trans)) allocate(trans(ntx,nty))
    
    fzk=0.d0
    pot = 0.d0
    trans = 1.d0
    F=0.0d0

    !print *, 'msfdes_step: arrays allocated '

    !print *, 'msfdes_step: using natypes=', natypes
    do itype = 1, natypes
      !print *, ' Generating image for itype : ', itype, ' in slice ', isl
      
      ! electric component
      I_cplx = dcmplx(1.0d0, 0.0d0)*I(:,:,0, itype) ! Image from the current slice, electric component
      call fft2d(I_cplx,F)    ! FFT of image I, F is now in k space
      pot = Vk_grid(:,:,itype)
      fzk = fzk + F*pot*S_inv ! is this okay to reuse that much?

      ! z magnetic component of A
      ! mx times ky for magnetic contribution
      I_cplx = dcmplx(1.0d0, 0.0d0)*I(:,:,1, itype) ! Image from the current slice, x component of magnetic moment
      call fft2d(I_cplx,F)    ! FFT of image I, F is now in k space
      pot = (magfac/sigma)*Ak_grid(:,:,itype)
      !fzk = fzk - (magfac/sigma)*pot*( mxF*ky-myF*kx )
      do iy = 1, nty
        do ix = 1, ntx
          fzk(ix,iy) = fzk(ix,iy) - F(ix,iy)*ky_arr(iy)*pot(ix,iy)*S_inv(ix,iy) 
        end do
      end do

      !print *, 'msfdes_step: electric pot created' 

      ! z magnetic component of A
      ! my times kx for magnetic contribution
      I_cplx = dcmplx(1.0d0, 0.0d0)*I(:,:,2, itype) ! Image from the current slice, y component of magnetic moment
      call fft2d(I_cplx,F)    ! FFT of image I, F is now in k space
      !pot = (magfac/sigma)*Ak_grid(:,:,itype)
      !fzk = fzk - (magfac/sigma)*pot*( mxF*ky-myF*kx )
      do iy = 1, nty
        do ix = 1, ntx
          fzk(ix,iy) = fzk(ix,iy) + F(ix,iy)*kx_arr(ix)*pot(ix,iy)*S_inv(ix,iy)
        end do
      end do

    end do

    pot = 0.d0 ! pot will be used to store the real space potential
    
    !print *, 'fzk=', sum(fzk)
    !call bfft2d(F,pot)      ! FFTI to get proj pot in real space
    call bfft2d(fzk,pot)      ! FFTI to get proj pot in real space
    if(bandlimit) call bw_limit(pot) !Should I do bw_limit????
    !if (sum(I(:,:,isl,0,1)) > 0 .and. isl==print_psi_k) call output_array(pot,'(a,i0,a)','results/potAz_r_',isl,'_fdes.dat')
    !if (sum(I(:,:,isl,0,1)) > 0 .and. isl==print_psi_k) call output_array(pot,'(a,i0,a)','results/potE_r_',isl,'_fdes.dat')
    !if (isl==print_psi_k) call output_array(pot,'(a,i0,a)','results/potE_r_',isl,'_fdes.dat')
    trans = exp( dcmplx(0.d0,1.d0)*sigma*pot  )    ! I think moder fortran compilers allow this implementation!
    if(bandlimit) call bw_limit(trans) !Should I do bw_limit????
    !trans_array(:,:,isl) = trans
    !wf = trans*wf         ! Real space
    !call fft2d(wf,kwf)    ! FFT
    !kwf = prop*kwf        ! Reciprocal space
    !call bfft2d(kwf,wf)   ! FFTI

    deallocate(fzk,pot,F,I_cplx)

  end subroutine msfdes_step_2D

end module cms
