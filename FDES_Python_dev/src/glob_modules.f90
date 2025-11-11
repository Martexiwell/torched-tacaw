module fdes_const

  IMPLICIT NONE

  DOUBLE PRECISION, PARAMETER :: PI=3.141592653589793238462643383279502884197D0
  DOUBLE PRECISION, PARAMETER :: EPS0 = 8.8541878188d-12            ! in SI units: F/m
  DOUBLE PRECISION, PARAMETER :: A0 = 5.29177210544d-11             ! in SI units: m
  DOUBLE PRECISION, PARAMETER :: A0_Ang = 0.529177210544d0          ! in Ångström
  DOUBLE PRECISION, PARAMETER :: ECHARGE=1.602176634d-19            ! in SI units: C
  DOUBLE PRECISION, PARAMETER :: EGAMMA=0.57721566490153286060651209008240243104215933593992d0 ! Euler's gamma
  DOUBLE PRECISION, PARAMETER :: RYDBERG=13.60569253                ! in the electron-Volts
  DOUBLE PRECISION, PARAMETER :: RPLANCK=1.054571817D-34            ! in the unit of J.s
  
  DOUBLE PRECISION, PARAMETER :: EMASS=9.10938188D-31               !in the unit of kg
  DOUBLE PRECISION, PARAMETER :: PLANCK=6.626068D-34                !in the unit of J.s
  DOUBLE PRECISION, PARAMETER :: PLANCKbar=6.626068D-34/(2*PI)                !in the unit of J.s
  DOUBLE PRECISION, PARAMETER :: LSPEED=2.99792458D8                !in the unit of m/s

  DOUBLE PRECISION, PARAMETER :: eVtoJ=1.602176565D-19              !electronvolt to joule
  DOUBLE PRECISION, PARAMETER :: Atom=1.0D-10                       !angstrom to meter
  DOUBLE PRECISION, PARAMETER :: JtoRy=4.587420897D17               !joul to rydberg
  DOUBLE PRECISION, PARAMETER :: mtoBohr=1.889725989D10             !meter to Bohr
  DOUBLE PRECISION, PARAMETER :: Bohrtonm=0.052917725D0             !bohr to nm

  DOUBLE PRECISION, PARAMETER :: BOHRMAG=9.2741D-24 		    !Joule per Tesla 
  DOUBLE PRECISION, PARAMETER :: VACPERM=PI*4D-7 		    ! N/A^2

end module fdes_const

module global

  implicit none
  
  integer nx, ny, nz       ! mesh within a unit cell (of dimension lat(1) x lat(2), nz is number of slices within unit cell
  integer nscx, nscy, nscz ! supercell size, nscz defines maximum thickness
  integer ntx, nty, ntz    ! =nx*nscx, ny*nscy and nz*nscz
  
  double precision coff     ! cut-off for potential (A)
  
  double precision accv     ! acceleration voltage in keV
  double precision conv     ! convergence angle in mrad
  integer          angm     ! angular momentum
  integer          shx, shy ! shift of the beam in pixels
  integer          tiltx, tilty ! beam tilt in pixels (shift in k-space)

  double precision df, c12a, c12b, c21a, c21b, c23a, c23b, c3, c32a, c32b, c34a, c34b, c41a, c41b, c43a, c43b, c45a, c45b, c5, c52a, c52b, c54a, c54b, c56a, c56b ! aberration coefficients

  double precision sigma    ! interaction parameter
  double precision lambda   ! wavelength in Angstrom
  double precision k        ! wave-vector size (1/A)
  
  double precision Lz       ! z-component of OAM
  
!  double precision, allocatable :: Az(:,:), Anp(:,:)  ! vector potential from atoms and the eventual non-periodic part -> now in potential.f90
  double precision mag(3)   ! macroscopic magnetization
  double complex, allocatable :: wf(:,:), kwf(:,:) ! wf slices in real space and k-space
  double precision mass ! relativistically corrected mass 

  logical debug
  logical print_psi_0       ! output the initial wave-function
  integer print_psi_r       ! output real-space WF every "print_psi_r" slices
  integer print_psi_k       ! output k-space WF every "print_psi_k" slices
  integer print_diffpatt    ! output diffraction pattern every "print_diffpatt" slices

  logical savepot           ! for nscx,nsxy,nscz different from 1, it can save computing time to save the potentials from unit cell (it costs memory instead)

  logical bandlimit         ! should we apply bandwidth limiting?
  logical window_apply      ! should we apply windowing function (maybe only on A_np?? -> to decide)
  double precision window_alpha  ! alpha parameter for windowing function

  !
  ! ADDED BY LUIS BRISEÑO
  !

  double precision, allocatable :: kx_arr(:), ky_arr(:)

 contains
 
  subroutine output_array(arr,fmt,prefix,val,suffix)
  
    implicit none
    
    double complex, intent(in)   :: arr(ntx,nty)
    character(len=*), intent(in) :: fmt, prefix, suffix
    integer, intent(in)          :: val
    
    integer ix, iy
    character(len=80)  fname

    write(fname,fmt) trim(prefix),val,trim(suffix)
    open(77,file=trim(fname),form='formatted')
    do ix=1, ntx
      do iy=1, nty
        write(77,'(2i6,2e20.12)') ix, iy, arr(ix,iy)
      enddo
      write(77,*)
    enddo
    close(77)
  
  end subroutine output_array

  subroutine output_array_bin(arr,fmt,prefix,val,suffix)
  
    implicit none
    
    double complex, intent(in)   :: arr(ntx,nty)
    character(len=*), intent(in) :: fmt, prefix, suffix
    integer, intent(in)          :: val
    
    integer ix, iy
    character(len=80)  fname

    write(fname,fmt) trim(prefix),val,trim(suffix)
    !open(77,file=trim(fname),form='binary')			   	                   ! JLBG changed this line becase his compiler did not suppor the binnary
    open(77, file=trim(fname), form='unformatted', access='stream')      ! JLBG changed this line becase his compiler did not suppor the binnary
    do ix=1, ntx
      do iy=1, nty
        write(77) arr(ix,iy)
      enddo
    enddo
    close(77)
  
  end subroutine output_array_bin

  subroutine output_array_shift(arr,fmt,prefix,val,suffix)
  
    implicit none
    
    double complex, intent(in)   :: arr(ntx,nty)
    character(len=*), intent(in) :: fmt, prefix, suffix
    integer, intent(in)          :: val
    
    integer ix, iy, i2x, i2y
    character(len=80)  fname

    write(fname,fmt) trim(prefix),val,trim(suffix)
    open(77,file=trim(fname),form='formatted')
    do ix=1, ntx
      do iy=1, nty
        i2x = mod(ix-1+ntx/2,ntx)+1
        i2y = mod(iy-1+nty/2,nty)+1
        write(77,'(2i6,2e20.12)') ix, iy, arr(i2x,i2y)
      enddo
      write(77,*)
    enddo
    close(77)
  
  end subroutine output_array_shift
  
  subroutine output_array_real(arr,fmt,prefix,val,suffix)

    implicit none

    double precision, intent(in) :: arr(ntx,nty)
    character(len=*), intent(in) :: fmt, prefix, suffix
    integer, intent(in)          :: val

    integer ix, iy
    character(len=80)  fname

    write(fname,fmt) trim(prefix),val,trim(suffix)
    open(77,file=trim(fname),form='formatted')
    do ix=1, ntx
      do iy=1, nty
        write(77,'(2i6,1e20.12)') ix, iy, arr(ix,iy)
      enddo
      write(77,*)
    enddo
    close(77)

  end subroutine output_array_real

end module global
