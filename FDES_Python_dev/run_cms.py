import numpy as np
import cms_module
import os

# Define the input file path (assuming it's in the same directory)
INPUT_FILE = 'data/mult.in' 

def run_fdes_simulation():
    """Initializes and runs the Fortran FDES multislice simulation step-by-step."""
    print("--- FDES Simulation Control ---")

    # --- 1. Initialization ---
    try:
        # Calls the Fortran routine to read input, load structure, init FFTW, etc.
        cms_module.cms_interface.initialize_simulation(INPUT_FILE)
        print(f"✅ Initialization successful using input file: {INPUT_FILE}")
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return

    # --- 2. Get Dimensions ---
    # Retrieve the final dimensions calculated by the Fortran setup.
    try:
        ntx = cms_module.cms_interface.get_ntx() # <--- CRASH LIKELY HERE
        nty = cms_module.cms_interface.get_nty()
        ntz = cms_module.cms_interface.get_ntz()
        print(f"Dimensions (Ntx, Nty, Ntz): ({ntx}, {nty}, {ntz})")
    except Exception as e:
        print(f"❌ Error retrieving dimensions. Check Fortran 'get_n*' functions.")
        return

    # --- 3. Run Simulation Loop (Real-Time Control) ---
    # --- && PRE-ALLOCATE the 3D Python Array (Crucial for efficiency) ---
    # Assuming the data type is complex*16 (complex(8) in Fortran)
    trans_array = np.zeros((ntx, nty, ntz), dtype=np.complex128, order='F')
    # Note: Using 'order='F' (Fortran order) can sometimes be faster 
    # when assigning slices from Fortran routines.
    
    #results_wf_norms = []
    
    prop_array = cms_module.cms_interface.get_prop_python(ntx, nty)
    
    for iz in range(1, ntz + 1):
        # Call the Fortran routine for a single step (msfdes_step)
        cms_module.cms_interface.msfdes_step_python(iz)
        
        # --- Real-Time Array Access (Efficiently fetching data) ---
        
        # Get the Propagator array for the current/next step
        # Dimensions (ntx, nty) are explicitly passed.
        trans_array[:, :, iz - 1] = cms_module.cms_interface.get_prop_python(ntx, nty)
        
        # Get the Wavefront array (wf) after propagation/scattering
        #wf_array = cms_module.cms_interface.get_wf_python(ntx, nty)
        
        # Perform Python-side analysis
        #current_norm = np.sum(np.abs(wf_array)**2)
        #results_wf_norms.append(current_norm)

        #print(f"  > Slice {iz}/{ntz}: Wavefront Norm = {current_norm:.6f}")
        if iz % (ntz // 10 or 1) == 0:
            #print(f"  > Slice {iz}/{ntz}: Wavefront Norm = {current_norm:.6f}")
            print(f"  > Slice {iz}/{ntz}")

    print("--- Simulation Complete ---")
    cms_module.cms_interface.deallocate_all()
         
    # --- 5. Return Results ---
    return prop_array, trans_array

# Execute the simulation
if __name__ == "__main__":
    # Ensure the Fortran module file is present in the execution directory.
    if not os.path.exists(cms_module.__file__):
         print(f"Error: Module file not found at {cms_module.__file__}")
    
    final_prop, final_trans = run_fdes_simulation()
    
    if final_trans is not None:
        print("\n** Python Analysis Summary **")
        print(f"Final Wavefront shape: {final_trans.shape}, Dtype: {final_trans.dtype}")
        print(f"Final Propagator shape: {final_prop.shape}")
        # Plotting or further analysis would go here (e.g., using matplotlib)
        
    final_prop, final_trans = run_fdes_simulation()
