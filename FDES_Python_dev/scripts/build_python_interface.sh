# Set your Fortran compiler and flags as they are crucial for consistency
F90="f2py"
#FFLAGS="-O3 -ffree-line-length-none"
FFLAGS="-O0 -g -Wall -fbounds-check -fcheck=all -ffpe-trap=invalid,zero -ffree-line-length-none"
INCLUDE="-I/usr/include"
LIB="-lfftw3"
MOD_DIR="./build"

# Re-generate a fresh .pyf file after renaming the module
f2py src/cms_interface.f90 -m cms_module -h cms_module.pyf

# All source files in dependency order
FORT_SRCS="src/glob_modules.f90 \
           src/atoms.f90 \
           src/fftw.f90 \
           src/init_wf.f90 \
           src/ak_mod.f90 \
           src/potential.f90 \
           src/cms.f90 \
           src/cms_interface.f90"

# --- Corrected Compilation Command ---
# Remove '-J$MOD_DIR' from --f90flags. We keep the -I$MOD_DIR for module search path.
f2py -c cms_module.pyf $FORT_SRCS \
    --fcompiler=gfortran \
    --f77flags="$FFLAGS" \
    --f90flags="$FFLAGS" \
    $INCLUDE \
    -I$MOD_DIR \
    $LIB
