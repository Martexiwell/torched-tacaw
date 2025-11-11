module cms_interface
  use fdes_const
  use global
  use atoms
  use fftw
  use potential
  use cms
  implicit none

  ! Expose the following global variables to Python for dimensions/state
  !integer, public :: ntx, nty, ntz

contains

  ! In the 'contains' section of cms_interface.f90

  integer function get_ntx()
    !f2py intent(out)
    use global
    implicit none
    get_ntx = ntx
  end function get_ntx

  integer function get_nty()
    !f2py intent(out)
    use global
    implicit none
    get_nty = nty
  end function get_nty
  
  integer function get_ntz()
    !f2py intent(out)
    use global
    implicit none
    get_ntz = ntz
  end function get_ntz

  subroutine initialize_simulation(input_file)
    !! Initializes the simulation, reading the input file and setting up all modules.
    character(len=*), intent(in) :: input_file
    integer :: ios
    double precision zstp
    logical file_exists

    ! NAMELIST declaration must be here or in a USEd module
    namelist /inputs/ debug, coff, nx, ny, nz, nscx, nscy, nscz, accv, shx, shy, tiltx, tilty, conv, angm, bandlimit, savepot, mag, print_psi_0, print_psi_r, print_psi_k, print_diffpatt, window_apply, window_alpha, df, c12a, c12b, c21a, c21b, c23a, c23b, c3, c32a, c32b, c34a, c34b, c41a, c41b, c43a, c43b, c45a, c45b, c5, c52a, c52b, c54a, c54b, c56a, c56b

    ! --- FIX: Reset file pointer before reading ---
    INQUIRE(FILE=input_file, EXIST=file_exists)
    IF (file_exists) THEN
        ! Ensure the file pointer is at the beginning before reading
        REWIND(UNIT=10, IOSTAT=ios)  ! Use the correct unit number (e.g., 10)
    END IF
    ! ---------------------------------------------

    ! --- 1. Read Input ---
    open(unit=10, file=trim(input_file), status='old', action='read', iostat=ios)
    if (ios /= 0) then
      print *, "Error opening input file: ", trim(input_file)
      stop
    endif
    read(10, nml=inputs)
    close(10)

    ! --- 2. Initialize Parameters and Modules ---
    ntx = nx*nscx
    nty = ny*nscy
    ntz = nz*nscz

    px = 1.0d0/dble(ntx)
    py = 1.0d0/dble(nty)
    pz = 1.0d0/dble(ntz)

    call load_structure
    call build_Sk
    call init_fft
    !call init_wf
    call init_scattering_factors
    zstp = lat(3)/dble(nz)
    call init_cms(zstp) ! This loads prop
    !call sq_deposit

    ! Note: 'trans_array' allocation and final storage are commented out
    ! in your original plan, as Python will likely manage the data flow.

  end subroutine initialize_simulation


  ! WE DON'T NEED THIS INTO TORCH-TACAW
  subroutine msfdes_step_python(iz)
    !! Executes a single multislice step at slice index iz
    integer, intent(in) :: iz
    !double precision norm

    call sq_deposit_2D(iz)
    call msfdes_step_2D(iz)

    ! Optional: Print norm for monitoring (can be removed if Python handles it)
    ! norm = dble(sum(wf*conjg(wf)))
    ! write(*,'(a,f12.5)') 'Wavefunction norm:', norm

  end subroutine msfdes_step_python


  ! Simple getters for Python
  subroutine get_prop_python(arr, N1, N2)
    !! Returns the current prop array (complex(8) -> np.complex128)
    integer, intent(in) :: N1, N2 ! New inputs for dimensions
!f2py intent(out) :: arr(N1, N2)   ! Use the inputs for dimensioning
    complex(8), intent(out) :: arr(N1, N2) ! Must use explicit dimensions
    
    ! This assumes 'prop' is allocated as (ntx, nty) and N1=ntx, N2=nty
    arr = prop(1:N1, 1:N2)
  end subroutine get_prop_python
  
  !subroutine get_wf_python(arr, N1, N2)
   ! integer, intent(in) :: N1, N2
!f2py intent(out) :: arr(N1, N2)
    !complex(8), intent(out) :: arr(N1, N2)
    !arr = wf(1:N1, 1:N2)
  !end subroutine get_wf_python

  !subroutine get_kwf_python(arr, N1, N2)
    !integer, intent(in) :: N1, N2
!f2py intent(out) :: arr(N1, N2)
    !complex(8), intent(out) :: arr(N1, N2)
    !arr = kwf(1:N1, 1:N2)
  !end subroutine get_kwf_python
  
  subroutine get_trans_python(arr, N1, N2)
    integer, intent(in) :: N1, N2
!f2py intent(out) :: arr(N1, N2, N3)
    complex(8), intent(out) :: arr(N1, N2)
    arr = trans(1:N1, 1:N2)
  end subroutine get_trans_python

  !pure subroutine safe_deallocate(x)
  !  class(*), allocatable, intent(inout) :: x
  !  if (allocated(x)) deallocate(x)
  !end subroutine safe_deallocate

  subroutine deallocate_all

    IF (ALLOCATED(z)) DEALLOCATE(z)
    IF (ALLOCATED(pos)) DEALLOCATE(pos)
    IF (ALLOCATED(magv)) DEALLOCATE(magv)
    IF (ALLOCATED(magm)) DEALLOCATE(magm)
    IF (ALLOCATED(occ)) DEALLOCATE(occ)
    IF (ALLOCATED(dwfac)) DEALLOCATE(dwfac)
    IF (ALLOCATED(cube)) DEALLOCATE(cube)
    IF (ALLOCATED(I)) DEALLOCATE(I)
    IF (ALLOCATED(prop)) DEALLOCATE(prop)
    IF (ALLOCATED(bwlim)) DEALLOCATE(bwlim)
    IF (ALLOCATED(bw_mask)) DEALLOCATE(bw_mask)
    IF (ALLOCATED(wf)) DEALLOCATE(wf)
    IF (ALLOCATED(kwf)) DEALLOCATE(kwf)
    IF (ALLOCATED(kx_arr)) DEALLOCATE(kx_arr)
    IF (ALLOCATED(ky_arr)) DEALLOCATE(ky_arr)
    IF (ALLOCATED(trans)) DEALLOCATE(trans)
    IF (ALLOCATED(Az)) DEALLOCATE(Az)
    IF (ALLOCATED(pot)) DEALLOCATE(pot)
    IF (ALLOCATED(r_V)) DEALLOCATE(r_V)
    IF (ALLOCATED(r_Az)) DEALLOCATE(r_Az)
    IF (ALLOCATED(radAz)) DEALLOCATE(radAz)
    IF (ALLOCATED(radV)) DEALLOCATE(radV)
    IF (ALLOCATED(fzk)) DEALLOCATE(fzk)
    IF (ALLOCATED(Vk_grid)) DEALLOCATE(Vk_grid)
    IF (ALLOCATED(Ak_grid)) DEALLOCATE(Ak_grid)
    IF (ALLOCATED(S_k)) DEALLOCATE(S_k)
    IF (ALLOCATED(S_inv)) DEALLOCATE(S_inv)
    IF (ALLOCATED(F)) DEALLOCATE(F)
    IF (ALLOCATED(I_cplx)) DEALLOCATE(I_cplx)

  end subroutine deallocate_all

end module cms_interface
