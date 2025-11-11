# =========================================================
# Makefile for building the Python module cms_module via f2py
# =========================================================

# --- Compiler and flags ---
F90        = f2py
FFLAGS     = -O0 -g -Wall -fbounds-check -fcheck=all -ffpe-trap=invalid,zero -ffree-line-length-none
INCLUDE    = -I/usr/include
LIBS       = -lfftw3
MOD_DIR    = build

# --- Module name and sources ---
MODULE     = cms_module
PYF        = $(MODULE).pyf
FORT_SRCS  = src/glob_modules.f90 \
             src/atoms.f90 \
             src/fftw.f90 \
             src/init_wf.f90 \
             src/ak_mod.f90 \
             src/potential.f90 \
             src/cms.f90 \
             src/cms_interface.f90

# --- Default target ---
all: $(MODULE).so

# --- Generate .pyf interface file ---
$(PYF): src/cms_interface.f90
	@echo ">>> Generating interface file $(PYF)"
	$(F90) src/cms_interface.f90 -m $(MODULE) -h $(PYF)

# --- Build the Python module ---
$(MODULE).so: $(PYF) $(FORT_SRCS)
	@mkdir -p $(MOD_DIR)
	@echo ">>> Building Python module $(MODULE).so"
	$(F90) -c $(PYF) $(FORT_SRCS) \
		--fcompiler=gfortran \
		--f77flags="$(FFLAGS)" \
		--f90flags="$(FFLAGS)" \
		$(INCLUDE) \
		-I$(MOD_DIR) \
		$(LIBS)

# --- Clean build artifacts ---
clean:
	@echo ">>> Cleaning build artifacts"
	rm -rf *.so $(PYF) $(MOD_DIR) *.mod *.pyc build/ __pycache__

.PHONY: all clean
