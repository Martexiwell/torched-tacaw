module fftw

  use, intrinsic :: iso_c_binding

  implicit none
  
  include 'fftw3.f03'

  integer, allocatable :: bw_mask(:,:)      ! mask for bandwidth limiting, public
  
  integer fwdfft, backfft                   ! private variables for FFT plans
!  type(C_PTR) :: fwdfft, backfft

  
 contains
 
  subroutine init_fft
  
    use global, only : ntx, nty
  
    implicit none
    
    !integer iret
    double complex, allocatable :: in(:,:), out(:,:)

    allocate(in(ntx,nty),out(ntx,nty),bw_mask(ntx,nty))

    bw_mask(:,:) = 1 ! we initialize with no BW limiting

    !call dfftw_init_threads(iret)
    !print *, iret
    !call dfftw_plan_with_nthreads(8)
    call dfftw_plan_dft_2d(fwdfft, ntx, nty, in, out, FFTW_FORWARD, FFTW_MEASURE)
    call dfftw_plan_dft_2d(backfft, ntx, nty, in, out, FFTW_BACKWARD, FFTW_MEASURE)
    
    deallocate(in,out)
    
  end subroutine init_fft
  
  subroutine fft2d(inp,out)
  
    use global, only : ntx, nty

    implicit none
    
    double complex inp(ntx,nty), out(ntx,nty)
    
    call dfftw_execute_dft(fwdfft,inp,out)
    out = out/sqrt(dble(ntx*nty))
    
  end subroutine fft2d

  subroutine bfft2d(inp,out)
  
    use global, only : ntx, nty

    implicit none
    
    double complex inp(ntx,nty), out(ntx,nty)
    
    call dfftw_execute_dft(backfft,inp,out)
    out = out/sqrt(dble(ntx*nty))
    
  end subroutine bfft2d
  
  double precision function getkx(ix)
  
    use atoms, only : lat
    use global
    
    implicit none
    
    integer ix
    
    if(ix<=(ntx+1)/2) then
      getkx = dble(ix-1)/(nscx*lat(1))
    else
      getkx = dble(ix-ntx-1)/(nscx*lat(1))
    endif
    
  end function getkx

  double precision function getky(iy)
  
    use atoms, only : lat
    use global
    
    implicit none
    
    integer iy
    
    if(iy<=(nty+1)/2) then
      getky = dble(iy-1)/(nscy*lat(2))
    else
      getky = dble(iy-nty-1)/(nscy*lat(2))
    endif
    
  end function getky

end module fftw
