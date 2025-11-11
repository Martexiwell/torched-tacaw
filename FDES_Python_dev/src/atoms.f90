 module atoms

  implicit none
  
  integer                          nat            ! number of atoms
  integer, allocatable          :: z(:)           ! proton number of each atom
  double precision, allocatable :: pos(:,:)       ! x,y,z of each atom in fractional coordinates
  double precision, allocatable :: magv(:,:)      ! mx, my, mz of each atom (normalized unit vector)
  double precision, allocatable :: magm(:)        ! magnitude of magnetic moment in Bohr magnetons
  double precision, allocatable :: occ(:)         ! fractional site occupation
  double precision, allocatable :: dwfac(:)       ! the Debye-Waller factor given in DrProbe way
  double precision                 lat(3)         ! cell parameters in Angstrom
  integer                          nc(3)          ! number of cubes in each dimension
  integer                          atpres(103)    ! to flag, which atoms are present in the structure file and index them
  integer                          natypes        ! how many atom types there are
  integer, allocatable          :: cube(:,:,:,:)  ! cubated atoms
  integer                          ncx, ncy, ncz  ! number of sub-intervals in cubation
  !
  !   ADDED BY LUIS BRISEÑO
  !
  double precision, allocatable :: I(:,:,:,:) ! Image offset function I(x,y,z, i, natype) for sq_deposit
  double precision                 px, py, pz     ! pixel pitch (meters or same units as atoms) maybe should use coff?

  contains

    !--------------------------------------------------------------------------------------------------------------------------
    subroutine load_structure
    
      use global

      implicit none
      
      integer i, maxnat, ios
      double precision alpha, beta, gamma
      character(len=2) elname
      character(len=10) filename
      logical file_exists
      
      filename = 'data/atoms'
      ! --- FIX: Reset file pointer before reading ---
      INQUIRE(FILE=filename, EXIST=file_exists)
      IF (file_exists) THEN
        ! Ensure the file pointer is at the beginning before reading
        REWIND(UNIT=20, IOSTAT=ios)  ! Use the correct unit number (e.g., 10)
      END IF
      ! ---------------------------------------------
      open(20, file='data/atoms', status='old', action='read')

      if(debug) print *, 'data/atoms opened'
      
      read(20,*) ! just a string to be ignored
      read(20,*) i, lat(1), lat(2), lat(3), alpha, beta, gamma
      lat = lat*10.d0 ! we convert from nm in cel files to Angstroms internally used by this code
      if(abs(alpha-90.d0)>1.0d-5 .or. abs(beta-90.d0)>1.0d-5 .or. abs(gamma-90.d0)>1.0d-5) then
         print *, "ERROR: can't handle non-orthogonal cells yet"
         stop
      endif

      maxnat = nint(lat(1)*lat(2)*lat(3)) ! practically all atoms need more than 1A^3 space, so this should be a safe upper margin
      natypes = 0 !! initialize counter of natypes
      atpres = 0
      allocate(z(maxnat),pos(maxnat,3),magm(maxnat),magv(maxnat,3),occ(maxnat),dwfac(maxnat))
      do i=1, maxnat
        read(20,*,iostat=ios) elname, pos(i,1), pos(i,2), pos(i,3), occ(i), dwfac(i), magv(i,1), magv(i,2), magv(i,3)
        if(ios/=0) goto 99
        z(i) = get_atomic_number(elname)
        do while(pos(i,1)<0.d0); pos(i,1)=pos(i,1)+1.d0; enddo
        do while(pos(i,1)>1.d0); pos(i,1)=pos(i,1)-1.d0; enddo
        do while(pos(i,2)<0.d0); pos(i,2)=pos(i,2)+1.d0; enddo
        do while(pos(i,2)>1.d0); pos(i,2)=pos(i,2)-1.d0; enddo
        do while(pos(i,3)<0.d0); pos(i,3)=pos(i,3)+1.d0; enddo
        do while(pos(i,3)>1.d0); pos(i,3)=pos(i,3)-1.d0; enddo
        if (atpres(z(i))==0) then
          natypes = natypes + 1
          atpres(z(i)) = natypes
        endif
        !if(debug) print *, elname, z(i)
        magm(i) = sqrt(dot_product(magv(i,:),magv(i,:)))
        if(magm(i)>1.0d-8) magv(i,:)=magv(i,:)/magm(i)
        !print *, ' Ak_grid=', magv(i,1), magv(i,2)
      enddo
      print *, "ERROR: hardcoded maxnat seems to be too small - please check your structure or increase maxnat"
      stop
      99 nat = i-1
      print *, "Loaded number of atoms: ", nat

    end subroutine load_structure
    
    
    !--------------------------------------------------------------------------------------------------------------------------
    subroutine calc_cubes
    
      use global
    
      implicit none
      
      integer iat, icx, icy, icz, icat
      integer, parameter :: nat_max_cube = 100
      double precision r(3)
      
      ncx = floor(lat(1)/coff)
      ncy = floor(lat(2)/coff)
      ncz = nz
      if(debug) print *, 'Cubation: ', ncx, ncy, ncz, lat(1)/ncx, lat(2)/ncy
      
      allocate(cube(0:ncx+1,0:ncy+1,ncz,0:nat_max_cube)) ! zero as the last index contains number of atoms in the cube
      do icx=0, ncx+1
        do icy=0, ncy+1
          do icz=1, ncz
            cube(icx,icy,icz,0) = 0
          enddo
        enddo
      enddo

      natypes = 0
      atpres = 0
      do iat=1, nat

        if (atpres(z(iat))==0) then
          natypes = natypes + 1
          atpres(z(iat)) = natypes
        endif
        
        r = pos(iat,:)
        call get_cube_indices(r,icx,icy,icz)

        icat = cube(icx,icy,icz,0) + 1
        cube(icx,icy,icz,0) = icat     ! increase number of atoms in the cube
        cube(icx,icy,icz,icat) = iat   ! store index of the atom, which sits there

      enddo
      
      ! take care of corners
      cube(0,0,:,:)         = cube(ncx,ncy,:,:)        ! 1) (0,0)              1 8 7
      cube(0,1:ncy,:,:)     = cube(ncx,1:ncy,:,:)      ! 2) (0,interior)       2   6
      cube(0,ncy+1,:,:)     = cube(ncx,1,:,:)          ! 3) (0,max)            3 4 5
      cube(1:ncx,ncy+1,:,:) = cube(1:ncx,1,:,:)        ! 4) (interior,max)
      cube(ncx+1,ncy+1,:,:) = cube(1,1,:,:)            ! 5) (max,max)
      cube(ncx+1,1:ncy,:,:) = cube(1,1:ncy,:,:)        ! 6) (max,interior)
      cube(ncx+1,0,:,:)     = cube(1,ncy,:,:)          ! 7) (max,0)
      cube(1:ncx,0,:,:)     = cube(1:ncx,ncy,:,:)      ! 8) (interior,0)
    
    end subroutine calc_cubes
    
    
    !--------------------------------------------------------------------------------------------------------------------------
    subroutine get_atom_coord(i,j,k,iat,r) ! output "r" is in fractional coordinates
    
      implicit none
      
      integer, intent(in)           :: i, j, k, iat
      double precision, intent(out) :: r(3)
      
      r = pos(cube(i,j,k,iat),:)
      ! since we copied to buffer cubes only the atom index, not a translated position, we need to do the translation here
      if(i==0)     r(1)=r(1)-1.d0
      if(i==ncx+1) r(1)=r(1)+1.d0
      if(j==0)     r(2)=r(2)-1.d0
      if(j==ncy+1) r(2)=r(2)+1.d0
    
    end subroutine get_atom_coord
    
    
    !--------------------------------------------------------------------------------------------------------------------------
    subroutine get_cube_indices(r,i,j,k) ! atom enters in fractional coordinates between (0,1)
  
      implicit none
    
      double precision, intent(in) :: r(3)
      integer, intent(out)         :: i, j, k
      
      i = ceiling((r(1)+1.23e-12)*dble(ncx))
      if(i>ncx) i=ncx
      j = ceiling((r(2)+1.23e-12)*dble(ncy))
      if(j>ncy) j=ncy
      k = ceiling((r(3)+1.23e-12)*dble(ncz))
      if(k>ncz) k=ncz
    
    end subroutine get_cube_indices
    
    !--------------------------------------------------------------------------------------------------------------------------
    function is_magnetic(Z) result(out)
    
      implicit none
      integer, intent(in) :: Z
      logical :: out
      
      if( ((Z>20).and.(Z<30)) .or. ((Z>38).and.(Z<48)) .or. ((Z>71).and.(Z<80)) ) then
        out = .true.
      else
        out = .false.
      endif
    
    end function is_magnetic
    
    
 !--------------------------------------------------------------------------------------------------------------------------
 FUNCTION get_atomic_number(element_name) RESULT(Z) ! generated by ChatGPT
  IMPLICIT NONE
  CHARACTER(LEN=*), INTENT(IN) :: element_name
  INTEGER :: Z
  CHARACTER(LEN=3) :: name
  INTEGER :: i, ichar_lower, ichar_upper

  ! Convert input to uppercase for case-insensitive comparison
  name = TRIM(ADJUSTL(element_name))
  DO i = 1, LEN(name)
    ichar_lower = ICHAR(name(i:i))
    IF (ichar_lower >= ICHAR('a') .AND. ichar_lower <= ICHAR('z')) THEN
      ichar_upper = ichar_lower - 32
      name(i:i) = CHAR(ichar_upper)
    END IF
  END DO

  SELECT CASE (name)
    CASE ('H')
      Z = 1
    CASE ('HE')
      Z = 2
    CASE ('LI')
      Z = 3
    CASE ('BE')
      Z = 4
    CASE ('B')
      Z = 5
    CASE ('C')
      Z = 6
    CASE ('N')
      Z = 7
    CASE ('O')
      Z = 8
    CASE ('F')
      Z = 9
    CASE ('NE')
      Z = 10
    CASE ('NA')
      Z = 11
    CASE ('MG')
      Z = 12
    CASE ('AL')
      Z = 13
    CASE ('SI')
      Z = 14
    CASE ('P')
      Z = 15
    CASE ('S')
      Z = 16
    CASE ('CL')
      Z = 17
    CASE ('AR')
      Z = 18
    CASE ('K')
      Z = 19
    CASE ('CA')
      Z = 20
    CASE ('SC')
      Z = 21
    CASE ('TI')
      Z = 22
    CASE ('V')
      Z = 23
    CASE ('CR')
      Z = 24
    CASE ('MN')
      Z = 25
    CASE ('FE')
      Z = 26
    CASE ('CO')
      Z = 27
    CASE ('NI')
      Z = 28
    CASE ('CU')
      Z = 29
    CASE ('ZN')
      Z = 30
    CASE ('GA')
      Z = 31
    CASE ('GE')
      Z = 32
    CASE ('AS')
      Z = 33
    CASE ('SE')
      Z = 34
    CASE ('BR')
      Z = 35
    CASE ('KR')
      Z = 36
    CASE ('RB')
      Z = 37
    CASE ('SR')
      Z = 38
    CASE ('Y')
      Z = 39
    CASE ('ZR')
      Z = 40
    CASE ('NB')
      Z = 41
    CASE ('MO')
      Z = 42
    CASE ('TC')
      Z = 43
    CASE ('RU')
      Z = 44
    CASE ('RH')
      Z = 45
    CASE ('PD')
      Z = 46
    CASE ('AG')
      Z = 47
    CASE ('CD')
      Z = 48
    CASE ('IN')
      Z = 49
    CASE ('SN')
      Z = 50
    CASE ('SB')
      Z = 51
    CASE ('TE')
      Z = 52
    CASE ('I')
      Z = 53
    CASE ('XE')
      Z = 54
    CASE ('CS')
      Z = 55
    CASE ('BA')
      Z = 56
    CASE ('LA')
      Z = 57
    CASE ('CE')
      Z = 58
    CASE ('PR')
      Z = 59
    CASE ('ND')
      Z = 60
    CASE ('PM')
      Z = 61
    CASE ('SM')
      Z = 62
    CASE ('EU')
      Z = 63
    CASE ('GD')
      Z = 64
    CASE ('TB')
      Z = 65
    CASE ('DY')
      Z = 66
    CASE ('HO')
      Z = 67
    CASE ('ER')
      Z = 68
    CASE ('TM')
      Z = 69
    CASE ('YB')
      Z = 70
    CASE ('LU')
      Z = 71
    CASE ('HF')
      Z = 72
    CASE ('TA')
      Z = 73
    CASE ('W')
      Z = 74
    CASE ('RE')
      Z = 75
    CASE ('OS')
      Z = 76
    CASE ('IR')
      Z = 77
    CASE ('PT')
      Z = 78
    CASE ('AU')
      Z = 79
    CASE ('HG')
      Z = 80
    CASE ('TL')
      Z = 81
    CASE ('PB')
      Z = 82
    CASE ('BI')
      Z = 83
    CASE ('PO')
      Z = 84
    CASE ('AT')
      Z = 85
    CASE ('RN')
      Z = 86
    CASE ('FR')
      Z = 87
    CASE ('RA')
      Z = 88
    CASE ('AC')
      Z = 89
    CASE ('TH')
      Z = 90
    CASE ('PA')
      Z = 91
    CASE ('U')
      Z = 92
    CASE ('NP')
      Z = 93
    CASE ('PU')
      Z = 94
    CASE ('AM')
      Z = 95
    CASE ('CM')
      Z = 96
    CASE ('BK')
      Z = 97
    CASE ('CF')
      Z = 98
    CASE ('ES')
      Z = 99
    CASE ('FM')
      Z = 100
    CASE DEFAULT
      Z = -1  ! Return -1 if element is not found
  END SELECT

 END FUNCTION get_atomic_number


 !--------------------------------------------------------------------------------------------------------------------------
 FUNCTION get_element_name(atomic_number) RESULT(element_name) ! generated by ChatGPT
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: atomic_number
  CHARACTER(LEN=20) :: element_name

  CHARACTER(LEN=20), DIMENSION(118) :: element_names =  &
  [ CHARACTER(LEN=20) ::  &
    'hydrogen  ', 'helium    ', 'lithium   ', 'beryllium ', 'boron     ', 'carbon    ', 'nitrogen  ', 'oxygen    ', 'fluorine  ', 'neon      ', &
    'sodium    ', 'magnesium ', 'aluminum  ', 'silicon   ', 'phosphorus', 'sulfur    ', 'chlorine  ', 'argon     ', 'potassium ', 'calcium   ', &
    'scandium  ', 'titanium  ', 'vanadium  ', 'chromium  ', 'manganese ', 'iron      ', 'cobalt    ', 'nickel    ', 'copper    ', 'zinc      ', &
    'gallium   ', 'germanium ', 'arsenic   ', 'selenium  ', 'bromine   ', 'krypton   ', 'rubidium  ', 'strontium ', 'yttrium   ', 'zirconium ', &
    'niobium   ', 'molybdenum', 'technetium', 'ruthenium ', 'rhodium   ', 'palladium ', 'silver    ', 'cadmium   ', 'indium    ', 'tin       ', &
    'antimony  ', 'tellurium ', 'iodine    ', 'xenon     ', 'cesium    ', 'barium    ', 'lanthanum ', 'cerium    ', 'praseodymium', 'neodymium ', &
    'promethium', 'samarium  ', 'europium  ', 'gadolinium', 'terbium   ', 'dysprosium', 'holmium   ', 'erbium    ', 'thulium   ', 'ytterbium ', &
    'lutetium  ', 'hafnium   ', 'tantalum  ', 'tungsten  ', 'rhenium   ', 'osmium    ', 'iridium   ', 'platinum  ', 'gold      ', 'mercury   ', &
    'thallium  ', 'lead      ', 'bismuth   ', 'polonium  ', 'astatine  ', 'radon     ', 'francium  ', 'radium    ', 'actinium  ', 'thorium   ', &
    'protactinium', 'uranium   ', 'neptunium ', 'plutonium ', 'americium ', 'curium    ', 'berkelium ', 'californium', 'einsteinium', 'fermium   ', &
    'mendelevium', 'nobelium  ', 'lawrencium', 'rutherfordium', 'dubnium ', 'seaborgium', 'bohrium   ', 'hassium   ', 'meitnerium', 'darmstadtium', &
    'roentgenium', 'copernicium', 'nihonium ', 'flerovium ', 'moscovium ', 'livermorium', 'tennessine', 'oganesson ' ]

  IF (atomic_number >= 1 .AND. atomic_number <= 118) THEN
    element_name = element_names(atomic_number)
  ELSE
    element_name = 'unknown'
  END IF

 END FUNCTION get_element_name


 !--------------------------------------------------------------------------------------------------------------------------
 FUNCTION get_element_acronym(atomic_number) RESULT(element_acronym) ! generated by ChatGPT
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: atomic_number
  CHARACTER(LEN=2) :: element_acronym

  CHARACTER(LEN=2), DIMENSION(118) :: element_acronyms =  &
  [ 'H ', 'He', 'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', 'Ne', &
    'Na', 'Mg', 'Al', 'Si', 'P ', 'S ', 'Cl', 'Ar', 'K ', 'Ca', &
    'Sc', 'Ti', 'V ', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', &
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y ', 'Zr', &
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', &
    'Sb', 'Te', 'I ', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', &
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', &
    'Lu', 'Hf', 'Ta', 'W ', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', &
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', &
    'Pa', 'U ', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', &
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', &
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og' ]

  IF (atomic_number >= 1 .AND. atomic_number <= 118) THEN
    element_acronym = element_acronyms(atomic_number)
  ELSE
    element_acronym = '??'
  END IF
  
  ! Convert to lowercase
  CALL lowercase(element_acronym)

 END FUNCTION get_element_acronym


 !--------------------------------------------------------------------------------------------------------------------------
 SUBROUTINE lowercase(str) ! generated by ChatGPT
  IMPLICIT NONE
  CHARACTER(LEN=*), INTENT(INOUT) :: str
  INTEGER :: i, icharval

  DO i = 1, LEN_TRIM(str)
    icharval = ICHAR(str(i:i))
    IF (icharval >= ICHAR('A') .AND. icharval <= ICHAR('Z')) THEN
      str(i:i) = CHAR(icharval + 32)
    END IF
  END DO

 END SUBROUTINE lowercase

 !
 !
 !        ADDED BY LUIS BRISEÑO   
 !
 !

 !This creates the I_Z(x,y) image used by FDES. I use simple OpenMP atomic updates; if we have high contention implement per-thread tiles and then reduce.
 ! How to calculate Nx, Ny???
 !--------------------------------------------------------------------------------------------------------------------------
 subroutine sq_deposit

    use global    !, only : I, px, py, pz, ntx, nty, ntz !?!  What do I use from global?
    implicit none
    
    !integer, intent(in) :: isl
    double precision r(3)                         ! position of atoms
    double precision um(3)                         ! unit magnetic moment of atoms
    !logical, intent(in), optional :: periodic    ! I put this here because it was suppoused to be an input fo the function?
    logical :: per
    integer :: iat, ix, iy, iz, ixp, iyp, itype
    double precision :: u, v
    double precision :: wll, wlr, wul, wur

    !double precision, parameter :: eps = 1.0d-30

    !r_frac(1) = dble(ix-1)/dble(nx)+eps
    !r_Ang(1) = r_frac(1)*lat(1)

    !npx = ntx   ! Number of pixels in x
    !npy = nty   ! Number of pixels in y            LET'S USE ntx, nty, ntz, instead
    !npz = ntz   ! Number of slices

    !px = nscx*lat(1)/dble(npx)  ! assuming len_x = nscx*lat(1) 
    !py = nscy*lat(2)/dble(npy)  ! assuming len_y = nscy*lat(2)
    !py = nscz*lat(3)/dble(npz)  ! assuming len_z = nscz*lat(3)
    ! ??? What to do when nscx, nscy are not 1!!!??? 
    !px = nscx/dble(npx)  ! assuming len_x = nscx in fractional coordinates
    !py = nscy/dble(npy)  ! assuming len_y = nscy in fractional coordinates
    !pz = nscz/dble(npz)  ! assuming len_z = nscz in fractional coordinates
    ! But maybe we would deal with this later
    !px = 1.0d0/dble(ntx)  ! assuming len_x = 1 in fractional coordinates
    !py = 1.0d0/dble(nty)  ! assuming len_y = 1 in fractional coordinates
    !pz = 1.0d0/dble(ntz)  ! assuming len_z = 1 in fractional coordinates
    !print *, '(px,py,pz)=', px,py,pz

    !if(.not.allocated(I)) allocate(I(ntx,nty))    ! We want to build a grid [1..Nx] × [1..Ny].
    !if(.not.allocated(mxI)) allocate(mxI(ntx,nty))    ! We want to build a grid [1..Nx] × [1..Ny].
    !if(.not.allocated(myI)) allocate(myI(ntx,nty))    ! We want to build a grid [1..Nx] × [1..Ny].

    per = .true.            ! periodic boundary conditions applied... or not?

    !if (present(periodic)) per = periodic

    ! Zero the images, allocated in init_scatfact subroutine
    I = 0.0d0

    ! $omp parallel do private(iat,r,x,y,z,ix,iy,iz,u,v,ixp,iyp,wll,wlr,wul,wur) schedule(static)
    do iat = 1, nat
    
      itype=atpres(z(iat))

      r = pos(iat,:)  ! in fractional coordinates
      !print *, ' position of atom : ', r
      !x = r(1) !+ eps
      !y = r(2) !+ eps
      !z = r(3) !+ eps
      um = magv(iat,:)
      !print *, ' magnetic moment of atom : ', um

      ! base pixel indices (Fortran 1-based)
      ! ??? What to do when nscx, nscy are not 1!!!??? 
      ! r ONLY contains the coordinates of the unit cell, or not?
      !ix = int(floor(x / px)) + 1
      !iy = int(floor(y / py)) + 1
      !iz = int(floor(z / pz)) + 1
      ix = int(floor( r(1) * ntx )) + 1
      iy = int(floor( r(2) * nty )) + 1
      iz = int(floor( r(3) * ntz )) + 1
      !print *, ' pixel coordinates of atom : ', ix,iy,iz

      ! fractional offsets inside pixel [0,1)
      u = abs( r(1)*ntx - dble(ix - 1) )
      v = abs( r(2)*nty - dble(iy - 1) )
      ! fractional offsets inside pixel (-0.5,0.5) ??
      !u = abs( x*ntx - dble(ix - 1) - 0.5d0 )
      !v = abs( y*nty - dble(iy - 1) - 0.5d0 )

      !print *, ' offset values of atom : ', u,v

      wll = (1.0d0 - u) * (1.0d0 - v)
      wlr = u * (1.0d0 - v)
      wul = (1.0d0 - u) * v
      wur = u * v

      !print *, ' bilinear interpoletion factors : ', wll,wlr,wul,wur

      ixp = ix + 1
      iyp = iy + 1

      ! handle boundaries (wrap or clamp)
      ! We want to stay within the valid grid [1..Nx] × [1..Ny].
      if (per) then
        ! What happens with the atoms outside the unit cell?
        !if (ix < 1) ix = mod(ix-1, ntx) + 1
        !if (iy < 1) iy = mod(iy-1, nty) + 1
        !ixp = mod(ixp-1, ntx) + 1
        !iyp = mod(iyp-1, nty) + 1
        ix  = mod(ix-1, ntx) + 1
        ixp = mod(ix, ntx) + 1
        iy  = mod(iy-1, nty) + 1
        iyp = mod(iy, nty) + 1
      else
        ! Ignore this atoms if they are "out of limit"?
        if (ix < 1 .or. ix > ntx) cycle
        if (iy < 1 .or. iy > nty) cycle
        if (ixp > Nx) ixp = ntx
        if (iyp > Ny) iyp = nty
      end if

      ! We need to make sure that we are only calculating I for the current slice within z and z + dz
      !if (iz > isl - 1 .and. iz <= isl) then ! test if atoms are within the same slice
        !$omp atomic
        !I(ix,iy,iz,0,itype) = I(ix,iy,iz,0,itype) + wll
        !I(ix,iy,iz,1,itype) = I(ix,iy,iz,1,itype) + um(1)*wll
        !I(ix,iy,iz,2,itype) = I(ix,iy,iz,2,itype) + um(2)*wll
        !I(ix,iy,iz,3,itype) = I(ix,iy,iz,3,itype) + um(3)*wll
        !$omp atomic
        !I(ixp,iy,iz,0,itype) = I(ixp,iy,iz,0,itype) + wlr
        !I(ixp,iy,iz,1,itype) = I(ixp,iy,iz,1,itype) + um(1)*wlr
        !I(ixp,iy,iz,2,itype) = I(ixp,iy,iz,2,itype) + um(2)*wlr
        !I(ixp,iy,iz,3,itype) = I(ixp,iy,iz,3,itype) + um(3)*wlr
        !$omp atomic
        !I(ix,iyp,iz,0,itype) = I(ix,iyp,iz,0,itype) + wul
        !I(ix,iyp,iz,1,itype) = I(ix,iyp,iz,1,itype) + um(1)*wul
        !I(ix,iyp,iz,2,itype) = I(ix,iyp,iz,2,itype) + um(2)*wul
        !I(ix,iyp,iz,3,itype) = I(ix,iyp,iz,3,itype) + um(3)*wul
        !$omp atomic
        !I(ixp,iyp,iz,0,itype) = I(ixp,iyp,iz,0,itype) + wur
        !I(ixp,iyp,iz,1,itype) = I(ixp,iyp,iz,1,itype) + um(1)*wur
        !I(ixp,iyp,iz,2,itype) = I(ixp,iyp,iz,2,itype) + um(2)*wur
        !I(ixp,iyp,iz,3,itype) = I(ixp,iyp,iz,3,itype) + um(3)*wur
        
        !write(*,*) 'sum I =', sum(I(:,:))
        !write(*,*) 'max I =', maxval(I(:,:))
        !write(*,*) 'min I =', minval(I(:,:))

        !write(*,*) 'ix,iy,u,v =', ix, iy, u, v    ! print for first few atoms
        !write(*,*) 'Ielem( ix-1:ix+2, iy-1:iy+2, it ) ='
        !print  *, I(max(1,ix-1):min(ntx,ix+2), max(1,iy-1):min(nty,iy+2))
      !end if
    end do
    ! $omp end parallel do

    print *, 'sq_deposit: creation of image I(x,y) completed for every slice'

 end subroutine sq_deposit

 subroutine sq_deposit_2D(isl)

    use global    !, only : I, px, py, pz, ntx, nty, ntz !?!  What do I use from global?
    implicit none
    
    integer, intent(in) :: isl
    double precision r(3)                         ! position of atoms
    double precision um(3)                         ! unit magnetic moment of atoms
    !logical, intent(in), optional :: periodic    ! I put this here because it was suppoused to be an input fo the function?
    logical :: per
    integer :: iat, ix, iy, iz, ixp, iyp, itype
    double precision :: u, v
    double precision :: wll, wlr, wul, wur

    !double precision, parameter :: eps = 1.0d-30

    !r_frac(1) = dble(ix-1)/dble(nx)+eps
    !r_Ang(1) = r_frac(1)*lat(1)

    !npx = ntx   ! Number of pixels in x
    !npy = nty   ! Number of pixels in y            LET'S USE ntx, nty, ntz, instead
    !npz = ntz   ! Number of slices

    !px = nscx*lat(1)/dble(npx)  ! assuming len_x = nscx*lat(1) 
    !py = nscy*lat(2)/dble(npy)  ! assuming len_y = nscy*lat(2)
    !py = nscz*lat(3)/dble(npz)  ! assuming len_z = nscz*lat(3)
    ! ??? What to do when nscx, nscy are not 1!!!??? 
    !px = nscx/dble(npx)  ! assuming len_x = nscx in fractional coordinates
    !py = nscy/dble(npy)  ! assuming len_y = nscy in fractional coordinates
    !pz = nscz/dble(npz)  ! assuming len_z = nscz in fractional coordinates
    ! But maybe we would deal with this later
    !px = 1.0d0/dble(ntx)  ! assuming len_x = 1 in fractional coordinates
    !py = 1.0d0/dble(nty)  ! assuming len_y = 1 in fractional coordinates
    !pz = 1.0d0/dble(ntz)  ! assuming len_z = 1 in fractional coordinates
    !print *, '(px,py,pz)=', px,py,pz

    !if(.not.allocated(I)) allocate(I(ntx,nty))    ! We want to build a grid [1..Nx] × [1..Ny].
    !if(.not.allocated(mxI)) allocate(mxI(ntx,nty))    ! We want to build a grid [1..Nx] × [1..Ny].
    !if(.not.allocated(myI)) allocate(myI(ntx,nty))    ! We want to build a grid [1..Nx] × [1..Ny].

    per = .true.            ! periodic boundary conditions applied... or not?

    !if (present(periodic)) per = periodic

    ! Zero the images, allocated in init_scatfact subroutine
    I = 0.0d0

    ! $omp parallel do private(iat,r,x,y,z,ix,iy,iz,u,v,ixp,iyp,wll,wlr,wul,wur) schedule(static)
    do iat = 1, nat
    
      itype=atpres(z(iat))

      r = pos(iat,:)  ! in fractional coordinates
      !print *, ' position of atom : ', r
      !x = r(1) !+ eps
      !y = r(2) !+ eps
      !z = r(3) !+ eps
      um = magv(iat,:)
      !print *, ' magnetic moment of atom : ', um

      ! base pixel indices (Fortran 1-based)
      ! ??? What to do when nscx, nscy are not 1!!!??? 
      ! r ONLY contains the coordinates of the unit cell, or not?
      !ix = int(floor(x / px)) + 1
      !iy = int(floor(y / py)) + 1
      !iz = int(floor(z / pz)) + 1
      ix = int(floor( r(1) * ntx )) + 1
      iy = int(floor( r(2) * nty )) + 1
      iz = int(floor( r(3) * ntz )) + 1
      !print *, ' pixel coordinates of atom : ', ix,iy,iz

      ! fractional offsets inside pixel [0,1)
      u = abs( r(1)*ntx - dble(ix - 1) )
      v = abs( r(2)*nty - dble(iy - 1) )
      ! fractional offsets inside pixel (-0.5,0.5) ??
      !u = abs( x*ntx - dble(ix - 1) - 0.5d0 )
      !v = abs( y*nty - dble(iy - 1) - 0.5d0 )

      !print *, ' offset values of atom : ', u,v

      wll = (1.0d0 - u) * (1.0d0 - v)
      wlr = u * (1.0d0 - v)
      wul = (1.0d0 - u) * v
      wur = u * v

      !print *, ' bilinear interpoletion factors : ', wll,wlr,wul,wur

      ixp = ix + 1
      iyp = iy + 1

      ! handle boundaries (wrap or clamp)
      ! We want to stay within the valid grid [1..Nx] × [1..Ny].
      if (per) then
        ! What happens with the atoms outside the unit cell?
        !if (ix < 1) ix = mod(ix-1, ntx) + 1
        !if (iy < 1) iy = mod(iy-1, nty) + 1
        !ixp = mod(ixp-1, ntx) + 1
        !iyp = mod(iyp-1, nty) + 1
        ix  = mod(ix-1, ntx) + 1
        ixp = mod(ix, ntx) + 1
        iy  = mod(iy-1, nty) + 1
        iyp = mod(iy, nty) + 1
      else
        ! Ignore this atoms if they are "out of limit"?
        if (ix < 1 .or. ix > ntx) cycle
        if (iy < 1 .or. iy > nty) cycle
        if (ixp > Nx) ixp = ntx
        if (iyp > Ny) iyp = nty
      end if

      ! We need to make sure that we are only calculating I for the current slice within z and z + dz
      if (iz > isl - 1 .and. iz <= isl) then ! test if atoms are within the same slice
        !$omp atomic
        I(ix,iy,0,itype) = I(ix,iy,0,itype) + wll
        I(ix,iy,1,itype) = I(ix,iy,1,itype) + um(1)*wll
        I(ix,iy,2,itype) = I(ix,iy,2,itype) + um(2)*wll
        !I(ix,iy,3,itype) = I(ix,iy,3,itype) + um(3)*wll
        !$omp atomic
        I(ixp,iy,0,itype) = I(ixp,iy,0,itype) + wlr
        I(ixp,iy,1,itype) = I(ixp,iy,1,itype) + um(1)*wlr
        I(ixp,iy,2,itype) = I(ixp,iy,2,itype) + um(2)*wlr
        !I(ixp,iy,3,itype) = I(ixp,iy,3,itype) + um(3)*wlr
        !$omp atomic
        I(ix,iyp,0,itype) = I(ix,iyp,0,itype) + wul
        I(ix,iyp,1,itype) = I(ix,iyp,1,itype) + um(1)*wul
        I(ix,iyp,2,itype) = I(ix,iyp,2,itype) + um(2)*wul
        !I(ix,iyp,3,itype) = I(ix,iyp,3,itype) + um(3)*wul
        !$omp atomic
        I(ixp,iyp,0,itype) = I(ixp,iyp,0,itype) + wur
        I(ixp,iyp,1,itype) = I(ixp,iyp,1,itype) + um(1)*wur
        I(ixp,iyp,2,itype) = I(ixp,iyp,2,itype) + um(2)*wur
        !I(ixp,iyp,3,itype) = I(ixp,iyp,3,itype) + um(3)*wur
        
        !write(*,*) 'sum I =', sum(I(:,:))
        !write(*,*) 'max I =', maxval(I(:,:))
        !write(*,*) 'min I =', minval(I(:,:))

        !write(*,*) 'ix,iy,u,v =', ix, iy, u, v    ! print for first few atoms
        !write(*,*) 'Ielem( ix-1:ix+2, iy-1:iy+2, it ) ='
        !print  *, I(max(1,ix-1):min(ntx,ix+2), max(1,iy-1):min(nty,iy+2))
      end if
    end do
    ! $omp end parallel do

    !print *, 'sq_deposit: creation of image I(x,y) completed for every slice'

 end subroutine sq_deposit_2D

end module atoms
