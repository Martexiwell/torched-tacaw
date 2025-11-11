!===========================================================
! ak_mod.f90 : Module for ak interpolation with
!                  asymptotic extension for k→0 and k→∞
!===========================================================
MODULE ak_mod
   
  IMPLICIT NONE

  ! kinds
  integer, parameter :: dp = selected_real_kind(15, 307)

  PRIVATE
  PUBLIC :: ak_setup, ak_interp, ak_finalize, init_mparams

  ! We aim to interpolate some function a(k)
  REAL(dp), ALLOCATABLE :: ka(:), aa(:), af(:)
  REAL(dp) :: kmin, kmax
  LOGICAL :: initialized = .FALSE.

  ! Parameters for asymptotic forms 
  ! Based on the fin by Lyon
  REAL(dp) :: a0,a1,a2,a3,b0,b1

  real(dp) :: pi = acos(-1.0_dp)   ! π from the inverse cosine

  ! mparams(atom,Z_index) : we allocate for 118 elements and 10 params
  ! based on Keenan's fits of atomic magnetic moments
  real(dp), dimension(118,10) :: mparams

CONTAINS

  !---------------------------------------------------------
  SUBROUTINE ak_setup(filename, z_nat, eps_threshold_in)
    ! Reads data (k, ak) from file and prepares ak
    CHARACTER(LEN=*), INTENT(IN) :: filename
    !REAL(dp), INTENT(IN) :: a0_in,a1_in,a2_in,a3_in,b0_in,b1_in,eps_threshold_in
    REAL(dp), INTENT(IN) :: eps_threshold_in
    INTEGER, INTENT(IN)  :: z_nat

    INTEGER :: n, i, ios, unit
    REAL(dp), ALLOCATABLE :: tmpk(:), tmpa(:)
    CHARACTER(LEN=512) :: line, line_trim

    ! Store asymptotic params
    !a0 = a0_in
    !a1 = a1_in
    !a2 = a2_in
    !a3 = a3_in
    !b0 = b0_in
    !b1 = b1_in
    !assumes init_mparms has been called elsewhere
    a0 = mparams(z_nat,1)
    a1 = mparams(z_nat,3)
    a2 = mparams(z_nat,5)
    a3 = mparams(z_nat,7)
  
    b0 = mparams(z_nat,2)
    b1 = mparams(z_nat,4)

    ! Kmin and Kmax values calculated according in which Kmin and Kmax 
    ! generates a relative error less than eps_threshold based in the subsequent 
    ! asymptoctic terms
    !kmin = ( (3.0d0*a0)*eps_threshold_in / ( a1*( (2.0d0*pi)**0.5d0 ) ) )**2.0d0
    !kmax = ( 1365.0d0*a1*(b0**2.0d0)*(pi**0.5d0) / ( 2048.0d0*a0*(b1**2.0d0)*eps_threshold_in ) )**(2.0d0/3.0d0)
    ! compute kmin/kmax from asymptotic error threshold (keep kinds consistent)
    kmin = ( (3.0_dp*a0)*eps_threshold_in / ( a1*( (2.0_dp*pi)**0.5_dp ) ) )**2.0_dp
    kmax = ( 1365.0_dp*a1*(b0**2.0_dp)*(pi**0.5_dp) / ( 2048.0_dp*a0*(b1**2.0_dp)*eps_threshold_in ) )**(2.0_dp/3.0_dp)

    ! --- First pass: open file and count data lines (skip comments/blanks)
    open(newunit=unit, file=filename, status='old', action='read', iostat=ios)
    if (ios /= 0) then
       write(*,*) 'ak_setup: ERROR: cannot open file ', trim(filename), ' iostat=', ios
       stop 1
    end if

    ! Skip the first line (the header "k a(k)")
    read(unit,*)
    ! Count lines in file
    n = 0
    do
       read(unit,'(A)', iostat=ios) line
       if (ios /= 0) exit
       line_trim = adjustl(line)
       if (len_trim(line_trim) == 0) cycle                 ! skip blank
       if (line_trim(1:1) == '#') cycle                    ! skip comments starting with '#'
       ! At this point we assume the line contains data -> count it
       n = n + 1
    end do

    if (n < 2) then
       write(*,*) 'ak_setup: ERROR: not enough data lines in ', trim(filename)
       close(unit)
       stop 1
    end if
    rewind(unit)

    ! Allocate
    ALLOCATE(tmpk(n), tmpa(n))

    ! Skip the first line (the header "k a(k)")
    read(unit,*)
    ! second pass: read only data lines into tmpk/tmpa
    i = 0 
    do
       read(unit,'(A)', iostat=ios) line
       if (ios /= 0) exit
       line_trim = adjustl(line)
       if (len_trim(line_trim) == 0) cycle
       if (line_trim(1:1) == '#') cycle
       ! now attempt to parse two reals from the trimmed line
       read(line_trim, *, iostat=ios) tmpk(i+1), tmpa(i+1)
       if (ios /= 0) then
          write(*,*) 'ak_setup: WARNING: failed to parse line ', trim(line_trim)
          cycle
       end if
       i = i + 1
       if (i == n) exit
    end do
    CLOSE(unit)

    ! Copy into module arrays
    if (allocated(ka)) deallocate(ka)
    if (allocated(aa)) deallocate(aa)
    if (allocated(af)) deallocate(af)

    ALLOCATE(ka(n), aa(n), af(n))
    ka = tmpk
    aa = tmpa

    ! ensure kmin/kmax lie inside the data bounds
    !IF (kmin < ka(1)) THEN
       kmin = ka(1)
    !ELSEIF (kmax > ka(n)) THEN
       kmax = ka(n)
    !END IF

    ! Compute second derivatives
    CALL ak_derivative(ka, aa, n, 1.0D30, 1.0D30, af)

    initialized = .TRUE.

    ! clean up temporaries
    deallocate(tmpk, tmpa)

    write(*,*) 'ak_setup: read ', n, ' points;' 
    write(*,*) 'kmin=', kmin, ', kmax=', kmax
    !write(*,*) ' points; k1=', ka(1), ', kn=', ka(n)
    !write(*,*) ' points; k1=', ka(2), ', kn=', ka(n-1)
  END SUBROUTINE ak_setup

  !---------------------------------------------------------
  FUNCTION ak_interp(k) RESULT(ak)
    REAL(dp), INTENT(IN) :: k
    REAL(dp) :: ak
    REAL(dp) :: prefac
    INTEGER :: n

    IF (.NOT. initialized) THEN
       PRINT *, "Error: ak not initialized. Call ak_setup first."
       STOP
    END IF

    prefac = 4*pi/k

    n = SIZE(ka)

    IF (k < kmin) THEN
       ! Small-k asymptotic form: a0 + a_small * k^p_small
       ak = a0 + (a1/3.0d0)*((2.0d0*pi*k)**0.5d0) + (a2*pi/4.0d0)*k + (2.0d0*a3/5.0d0)*((2*pi*(k**3.0d0))**0.5d0)
       ak = prefac*ak
    ELSEIF (k > kmax) THEN
       ! Large-k asymptotic form: a_big * k^(-p_big)
       ak = (-144.0d0*a0/(b0**2)) * k**(-6.0d0) -12285.0d0*(pi**2.0d0)*a1*( k**(-7.5d0) ) / ( 64*( b1**2.0d0 )*(2.0d0**0.5d0) )
       ak = prefac*ak 
    ELSE
       CALL splint(ka, aa, af, n, k, ak)
    END IF
  END FUNCTION ak_interp

  !---------------------------------------------------------
  SUBROUTINE ak_finalize()
    IF (ALLOCATED(ka)) DEALLOCATE(ka, aa, af)
    initialized = .FALSE.
  END SUBROUTINE ak_finalize

  !---------------------------------------------------------
  SUBROUTINE ak_derivative(k,ak,n,ap1,apn,af)
    INTEGER, INTENT(IN) :: n
    REAL(dp), INTENT(IN) :: k(n), ak(n), ap1, apn
    REAL(dp), INTENT(OUT) :: af(n)
    INTEGER :: i,j
    REAL(dp) :: p,qn,sig,un
    REAL(dp), ALLOCATABLE :: u(:)

    ALLOCATE(u(n-1))
    IF (ap1 > 0.99d30) THEN
       af(1)=0.0d0
       u(1)=0.0d0
    ELSE
       af(1)=-0.5d0
       u(1)=(3.0d0/(k(2)-k(1)))*((ak(2)-ak(1))/(k(2)-k(1))-ap1)
    END IF
    DO i=2,n-1
       sig=(k(i)-k(i-1))/(k(i+1)-k(i-1))
       p=sig*af(i-1)+2.0d0
       af(i)=(sig-1.0d0)/p
       u(i)=(6.0d0*((ak(i+1)-ak(i))/(k(i+1)-k(i))-(ak(i)-ak(i-1))/(k(i)-k(i-1))) /(k(i+1)-k(i-1))-sig*u(i-1))/p
    END DO
    IF (apn > 0.99d30) THEN
       qn=0.0d0
       un=0.0d0
    ELSE
       qn=0.5d0
       un=(3.0d0/(k(n)-k(n-1)))*(apn-(ak(n)-ak(n-1))/(k(n)-k(n-1)))
    END IF
    af(n)=(un-qn*u(n-1))/(qn*af(n-1)+1.0d0)
    DO j=n-1,1,-1
       af(j)=af(j)*af(j+1)+u(j)
    END DO
    DEALLOCATE(u)
  END SUBROUTINE ak_derivative

  !---------------------------------------------------------
  SUBROUTINE splint(ka,aa,af,n,k,ak)
    INTEGER, INTENT(IN) :: n
    REAL(dp), INTENT(IN) :: ka(n), aa(n), af(n), k
    REAL(dp), INTENT(OUT) :: ak
    INTEGER :: klo,khi,k_avg
    REAL(dp) :: h,a,b, prefac

    klo=1
    khi=n
    DO WHILE (khi-klo > 1)
       k_avg=(khi+klo)/2
       IF(ka(k_avg) > k) THEN
          khi=k_avg
       ELSE
          klo=k_avg
       END IF
    END DO
    h=ka(khi)-ka(klo)
    a=(ka(khi)-k)/h
    b=(k-ka(klo))/h
    ak=a*aa(klo)+b*aa(khi)+((a**3-a)*af(klo)+(b**3-b)*af(khi))*(h*h)/6.0d0

    ! We added the factor "( ( pi / (2.0d0*k) )**0.5d0 )" manually that comes from the Hankel Transfor
    ! (4*pi/k) comes to make ak the reciprocal proj pot
    prefac = (4*pi/k)*( ( pi / (2.0d0*k) )**0.5d0 ) 
    ak = prefac*ak
  END SUBROUTINE splint

  ! based on Keenan's fits of atomic magnetic moments
  !---------------------------------------------------------
  subroutine init_mparams
    implicit none

    ! initialize to zero
    mparams = 0.0_dp ! most atoms won't generate A-/B-fields

    !mparms(z, i) expects z and i
    ! where z is the atomic number and the index i corresponds to 
    ! i= 1 -> a0
    ! i= 2 -> b0
    ! i= 3 -> a1
    ! i= 4 -> b1
    ! i= 5 -> a2
    ! i= 6 -> b2
    ! i= 7 -> a3
    ! i= 8 -> b3
    ! i= 9 -> a4
    ! i=10 -> b4

    ! Sc atom fit
     mparams(21, 1) =       0.96270000d0
     mparams(21, 2) =       1.29800000d0
     mparams(21, 3) =      -0.01195000d0
     mparams(21, 4) =       0.00188700d0
     mparams(21, 5) =       0.00320700d0
     mparams(21, 6) =       0.00072230d0
     mparams(21, 7) =       0.17460000d0
     mparams(21, 8) =       0.26240000d0
     mparams(21, 9) =       0.02062000d0
     mparams(21,10) =       0.03740000d0

     ! Ti atom fit
     mparams(22, 1) =       0.98650000d0
     mparams(22, 2) =       2.00000000d0
     mparams(22, 3) =      -0.00256500d0
     mparams(22, 4) =       0.00144200d0
     mparams(22, 5) =       0.39060000d0
     mparams(22, 6) =       0.25280000d0
     mparams(22, 7) =      -0.52380000d0
     mparams(22, 8) =       6.22800000d0
     mparams(22, 9) =      -0.17310000d0
     mparams(22,10) =       0.38690000d0

     ! V atom fit
     mparams(23, 1) =       6.59400000d0
     mparams(23, 2) =     463.80000000d0
     mparams(23, 3) =       2.00700000d0
     mparams(23, 4) =       0.47610000d0
     mparams(23, 5) =     -41.02000000d0
     mparams(23, 6) =    3962.00000000d0
     mparams(23, 7) =      -1.16500000d0
     mparams(23, 8) =       2.72000000d0
     mparams(23, 9) =      -0.77930000d0
     mparams(23,10) =       0.49110000d0

     ! Cr atom fit
     mparams(24, 1) =       1.20100000d0
     mparams(24, 2) =       4.65500000d0
     mparams(24, 3) =       1.30400000d0
     mparams(24, 4) =       0.27830000d0
     mparams(24, 5) =      -5.86700000d0
     mparams(24, 6) =       1.04500000d0
     mparams(24, 7) =       3.57700000d0
     mparams(24, 8) =       5.46200000d0
     mparams(24, 9) =       3.88000000d0
     mparams(24,10) =       1.31200000d0

     ! Mn atom fit
     mparams(25, 1) =       0.81820000d0
     mparams(25, 2) =       4.15600000d0
     mparams(25, 3) =       2.39000000d0
     mparams(25, 4) =       0.26630000d0
     mparams(25, 5) =      -6.85800000d0
     mparams(25, 6) =       3.05500000d0
     mparams(25, 7) =       6.22200000d0
     mparams(25, 8) =       4.96000000d0
     mparams(25, 9) =      -0.75400000d0
     mparams(25,10) =       0.24500000d0

     ! Fe atom fit
     mparams(26, 1) =       6.96700000d0
     mparams(26, 2) =     435.70000000d0
     mparams(26, 3) =       1.79900000d0
     mparams(26, 4) =       0.18940000d0
     mparams(26, 5) =     -44.85000000d0
     mparams(26, 6) =    3955.00000000d0
     mparams(26, 7) =      -0.66650000d0
     mparams(26, 8) =       1.29400000d0
     mparams(26, 9) =      -0.51910000d0
     mparams(26,10) =       0.16930000d0

     ! Co atom fit
     mparams(27, 1) =       5.32400000d0
     mparams(27, 2) =     271.60000000d0
     mparams(27, 3) =       1.89400000d0
     mparams(27, 4) =       0.16380000d0
     mparams(27, 5) =     -31.58000000d0
     mparams(27, 6) =    2386.00000000d0
     mparams(27, 7) =      -0.82530000d0
     mparams(27, 8) =       0.95670000d0
     mparams(27, 9) =      -0.52110000d0
     mparams(27,10) =       0.12850000d0

     ! Ni atom fit
     mparams(28, 1) =       2.12300000d0
     mparams(28, 2) =       0.21090000d0
     mparams(28, 3) =       5.42300000d0
     mparams(28, 4) =       6.53500000d0
     mparams(28, 5) =     -42.32000000d0
     mparams(28, 6) =       0.95410000d0
     mparams(28, 7) =      70.10000000d0
     mparams(28, 8) =       1.13900000d0
     mparams(28, 9) =     -30.31000000d0
     mparams(28,10) =       1.32500000d0

     ! Cu atom fit
     mparams(29, 1) =       1.68400000d0
     mparams(29, 2) =       0.01976000d0
     mparams(29, 3) =      -1.09300000d0
     mparams(29, 4) =       0.05604000d0
     mparams(29, 5) =      -0.07235000d0
     mparams(29, 6) =       0.00205800d0
     mparams(29, 7) =      -0.11530000d0
     mparams(29, 8) =       0.00449900d0
     mparams(29, 9) =      -0.44340000d0
     mparams(29,10) =       0.57740000d0

     ! Y atom fit
     mparams(39, 1) =       1.80800000d0
     mparams(39, 2) =       0.00182400d0
     mparams(39, 3) =      -0.58160000d0
     mparams(39, 4) =       0.09904000d0
     mparams(39, 5) =      -1.82400000d0
     mparams(39, 6) =       0.00030880d0
     mparams(39, 7) =       1.04500000d0
     mparams(39, 8) =       0.00013210d0
     mparams(39, 9) =      -0.17140000d0
     mparams(39,10) =       0.00005709d0

     ! Zr atom fit
     mparams(40, 1) =       0.00295400d0
     mparams(40, 2) =       0.00035820d0
     mparams(40, 3) =       1.25100000d0
     mparams(40, 4) =      20.98000000d0
     mparams(40, 5) =       0.96150000d0
     mparams(40, 6) =       5.49600000d0
     mparams(40, 7) =      -0.10080000d0
     mparams(40, 8) =       0.24810000d0
     mparams(40, 9) =       0.19460000d0
     mparams(40,10) =       0.57400000d0

     ! Nb atom fit
     mparams(41, 1) =       0.98500000d0
     mparams(41, 2) =       0.06480000d0
     mparams(41, 3) =       0.33540000d0
     mparams(41, 4) =       3.14600000d0
     mparams(41, 5) =      -0.31610000d0
     mparams(41, 6) =       0.02979000d0
     mparams(41, 7) =      -1.07100000d0
     mparams(41, 8) =       0.26020000d0
     mparams(41, 9) =       0.66190000d0
     mparams(41,10) =       0.29390000d0

     ! Mo atom fit
     mparams(42, 1) =       1.15000000d0
     mparams(42, 2) =       0.70320000d0
     mparams(42, 3) =       0.20980000d0
     mparams(42, 4) =       0.01704000d0
     mparams(42, 5) =      -1.33100000d0
     mparams(42, 6) =       0.03853000d0
     mparams(42, 7) =       0.70770000d0
     mparams(42, 8) =       0.03072000d0
     mparams(42, 9) =       0.18150000d0
     mparams(42,10) =       0.17240000d0

     ! Ru atom fit
     mparams(44, 1) =       1.14100000d0
     mparams(44, 2) =       0.75940000d0
     mparams(44, 3) =       0.25080000d0
     mparams(44, 4) =       0.01774000d0
     mparams(44, 5) =      -1.32900000d0
     mparams(44, 6) =       0.04251000d0
     mparams(44, 7) =       0.58500000d0
     mparams(44, 8) =       0.16240000d0
     mparams(44, 9) =       0.38490000d0
     mparams(44,10) =       0.02449000d0

     ! Rh atom fit
     mparams(45, 1) =       0.02283000d0
     mparams(45, 2) =       0.00162500d0
     mparams(45, 3) =       2.76900000d0
     mparams(45, 4) =       1.72400000d0
     mparams(45, 5) =      -2.18700000d0
     mparams(45, 6) =       2.57300000d0
     mparams(45, 7) =      -0.14440000d0
     mparams(45, 8) =       0.04847000d0
     mparams(45, 9) =       0.14020000d0
     mparams(45,10) =       0.06810000d0

     ! Pd atom fit
     mparams(46, 1) =       0.00762300d0
     mparams(46, 2) =       0.00040480d0
     mparams(46, 3) =      -3.55800000d0
     mparams(46, 4) =      96.47000000d0
     mparams(46, 5) =       9.94200000d0
     mparams(46, 6) =     169.10000000d0
     mparams(46, 7) =       2.20300000d0
     mparams(46, 8) =      11.02000000d0
     mparams(46, 9) =       0.23910000d0
     mparams(46,10) =       0.49860000d0

     ! Ag atom fit
     mparams(47, 1) =       4.34900000d0
     mparams(47, 2) =      39.61000000d0
     mparams(47, 3) =      -8.31200000d0
     mparams(47, 4) =     153.70000000d0
     mparams(47, 5) =       0.46110000d0
     mparams(47, 6) =       0.26750000d0
     mparams(47, 7) =      -0.61890000d0
     mparams(47, 8) =       0.34810000d0
     mparams(47, 9) =       2.32000000d0
     mparams(47,10) =      21.11000000d0

     ! Hf atom fit
     mparams(72, 1) =       0.86020000d0
     mparams(72, 2) =       1.20100000d0
     mparams(72, 3) =       0.26970000d0
     mparams(72, 4) =       0.06587000d0
     mparams(72, 5) =      -0.19460000d0
     mparams(72, 6) =       1.34500000d0
     mparams(72, 7) =      -0.17150000d0
     mparams(72, 8) =       0.05218000d0
     mparams(72, 9) =      -0.11490000d0
     mparams(72,10) =       0.33840000d0

     ! Ta atom fit
     mparams(73, 1) =       0.97820000d0
     mparams(73, 2) =       1.50200000d0
     mparams(73, 3) =       0.21600000d0
     mparams(73, 4) =       0.05528000d0
     mparams(73, 5) =      -0.67570000d0
     mparams(73, 6) =       0.11350000d0
     mparams(73, 7) =       0.39000000d0
     mparams(73, 8) =       1.21000000d0
     mparams(73, 9) =       0.25660000d0
     mparams(73,10) =       0.12250000d0

     ! W atom fit
     mparams(74, 1) =       0.87050000d0
     mparams(74, 2) =       1.17100000d0
     mparams(74, 3) =       0.95430000d0
     mparams(74, 4) =       0.06933000d0
     mparams(74, 5) =      -1.54600000d0
     mparams(74, 6) =       0.07871000d0
     mparams(74, 7) =       0.27100000d0
     mparams(74, 8) =       0.23820000d0
     mparams(74, 9) =       0.31970000d0
     mparams(74,10) =       0.05784000d0

     ! Re atom fit
     mparams(75, 1) =       2.07000000d0
     mparams(75, 2) =     669.70000000d0
     mparams(75, 3) =       1.51800000d0
     mparams(75, 4) =       0.09766000d0
     mparams(75, 5) =       0.49920000d0
     mparams(75, 6) =      13.97000000d0
     mparams(75, 7) =      -0.56300000d0
     mparams(75, 8) =       0.04738000d0
     mparams(75, 9) =      -0.42180000d0
     mparams(75,10) =       0.23850000d0

     ! Os atom fit
     mparams(76, 1) =       1.04200000d0
     mparams(76, 2) =       1.46900000d0
     mparams(76, 3) =       0.34990000d0
     mparams(76, 4) =       0.05149000d0
     mparams(76, 5) =      -1.51400000d0
     mparams(76, 6) =       0.11530000d0
     mparams(76, 7) =       1.08600000d0
     mparams(76, 8) =       0.94170000d0
     mparams(76, 9) =       0.70260000d0
     mparams(76,10) =       0.11420000d0

     ! Ir atom fit
     mparams(77, 1) =       0.02514000d0
     mparams(77, 2) =       0.02202000d0
     mparams(77, 3) =      11.39000000d0
     mparams(77, 4) =     101.10000000d0
     mparams(77, 5) =     -21.54000000d0
     mparams(77, 6) =     299.20000000d0
     mparams(77, 7) =       2.05000000d0
     mparams(77, 8) =       6.56200000d0
     mparams(77, 9) =      -0.02924000d0
     mparams(77,10) =       0.04927000d0

     ! Pt atom fit
     mparams(78, 1) =       2.00100000d0
     mparams(78, 2) =       0.19250000d0
     mparams(78, 3) =      -3.30500000d0
     mparams(78, 4) =       0.20580000d0
     mparams(78, 5) =       1.18500000d0
     mparams(78, 6) =       8.15800000d0
     mparams(78, 7) =       3.71600000d0
     mparams(78, 8) =       0.28570000d0
     mparams(78, 9) =      -2.08800000d0
     mparams(78,10) =       0.31530000d0

     ! Au atom fit
     mparams(79, 1) =       0.61000000d0
     mparams(79, 2) =       1.74700000d0
     mparams(79, 3) =       1.06900000d0
     mparams(79, 4) =       2.51100000d0
     mparams(79, 5) =       0.62580000d0
     mparams(79, 6) =       0.27640000d0
     mparams(79, 7) =      -3.43300000d0
     mparams(79, 8) =       0.63400000d0
     mparams(79, 9) =       2.10200000d0
     mparams(79,10) =       0.77600000d0
  end subroutine init_mparams

END MODULE ak_mod
