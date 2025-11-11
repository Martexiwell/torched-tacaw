subroutine init_wf

    use fdes_const
    use global
    use atoms, only : lat
    use fftw
    !use potential, only : kx_arr, ky_arr
  
    implicit none
    
    integer ix, iy         ! angular momentum (hbar), # of pixels in x,y direction
    double precision kx, ky, kcx, kcy, kmax, norm
    double precision ::  phi, th, th2, th3, th4, th5, th6, chi
    
    if(.not.allocated(wf))  allocate(wf(ntx,nty))
    if(.not.allocated(kwf)) allocate(kwf(ntx,nty))
    
    lambda = planck*lspeed / sqrt( echarge*accv*dble(1000) * ( dble(2)*emass*lspeed*lspeed + echarge*accv*dble(1000) ) ) ! now in m
    lambda = lambda*1.0d10 ! now in Angstrom
    
    k = dble(1)/lambda
    !kv(:) = k*dir(:)/sqrt(dot_product(dir,dir))
    
    sigma = dble(2)*pi/(lambda*accv) * ( emass*lspeed*lspeed + echarge*accv*dble(1000) ) / ( dble(2)*emass*lspeed*lspeed + echarge*accv*dble(1000) )
    print *, 'Interaction parameter sigma=', sigma
    
    write(*,'(a)')        '=================== Initial wavefunction generation =================='
    write(*,'(a,f10.2)')  '             Acceleration voltage (kV): ',accv
    write(*,'(a,f10.4)')  '                 Wavelength (Angstrom): ',lambda
    write(*,'(a,f10.4)')  '              Wave-vector (1/Angstrom): ',k
    !write(*,'(a,3f9.4)')  '                        ... components: ',kv(1:3)
    write(*,'(a,f10.4)')  ' Interaction parameter (1/kV Angstrom): ',sigma
    if(conv<0) then ! plane wave, for the moment just (0,0,1) works
      write(*,'(a)')      '                 *** Plane wave calculation ***'
      wf(:,:) = dble(1)/sqrt(dble(ntx*nty))
    else ! convergent electron beam, possibly with angular momentum
      write(*,'(a,f10.2)')  '              Convergence angle (mrad): ',conv
      write(*,'(a,i10)')    '       Orbital angular momentum (hbar): ',angm
      kmax = k*conv/dble(1000)
      write(*,'(a,f10.4)')  '                     qmax (1/Angstrom): ',kmax
      kwf(:,:) = dble(0)
      
      ! --- debug checks start ---
      !print *, '--- init_wf: sanity checks ---'
      !print *, 'ntx, nty =', ntx, nty
      !print *, 'nscx, nscy =', nscx, nscy
      !print *, 'tiltx, tilty =', tiltx, tilty
      !print *, 'conv =', conv, 'angm =', angm, 'lambda =', lambda, 'k =', k
      !flush(6)

      ! Check arrays exist and types/kinds/sizes
      !if (.not. allocated(kwf)) then
      !  print *, 'ERROR: kwf is NOT allocated!'
      !  stop 1
      !else
      !  print *, 'kwf shape =', size(kwf,1), size(kwf,2)
      !end if

      !if (.not. allocated(wf)) then
      !  print *, 'ERROR: wf is NOT allocated!'
      !  stop 1
      !else
      !  print *, 'wf shape =', size(wf,1), size(wf,2)
      !end if

      !! check shapes match expectations
      !if ( size(kwf,1) /= ntx .or. size(kwf,2) /= nty ) then
      !  print *, 'ERROR: kwf shape mismatch expected (',ntx,',',nty,') got (', &
      !           size(kwf,1),',',size(kwf,2),')'
      !  stop 1
      !end if

      !if ( size(wf,1) /= ntx .or. size(wf,2) /= nty ) then
      !  print *, 'ERROR: wf shape mismatch expected (',ntx,',',nty,') got (', &
      !           size(wf,1),',',size(wf,2),')'
      !  stop 1
      !end if

      ! Optional: check kinds (requires you have dp known here)
      !print *, 'kind of kwf element =', kind(kwf(1,1)), ' expected kind for complex double (use kind(1.0d0) or iso_fortran_env)'
      !flush(6)
      ! --- debug checks end ---

      do while(tiltx<0);   tiltx=tiltx+ntx; enddo
      do while(tilty<0);   tilty=tilty+nty; enddo
      do while(tiltx>ntx); tiltx=tiltx-ntx; enddo
      do while(tilty>nty); tilty=tilty-nty; enddo
      !kcx = getkx(tiltx+1)
      !kcy = getky(tilty+1)
      kcx = kx_arr(tiltx+1)
      kcy = ky_arr(tilty+1)
      !print *, 'kcx, kcy=', kcx, kcy
      do ix = 1, ntx
        !kx = getkx(ix)
        kx = kx_arr(ix)
        do iy = 1, nty
          !ky=1
          !ky = getky(iy)
          ky = ky_arr(iy)
          !print *, 'kx, ky=', kx, ky
          if((kx-kcx)*(kx-kcx)+(ky-kcy)*(ky-kcy)<kmax*kmax) then
          phi = atan2(ky-kcy,kx-kcx)
          kwf(ix,iy) = exp(dcmplx(0,angm)*phi) * exp(dcmplx(0,-2)*pi*(dble(shx*(ix-1))/dble(ntx)+dble(shy*(iy-1))/dble(nty)))

          !phi=ATAN2(ky-tilty,kx-tiltx)
          !!!ABBERRATIONS!!! 
          th = sqrt(kx*kx+ky*ky)*lambda
          th2 = th*th
          chi = -th2*df/2.d0 ! df = -c10
          chi = chi + th2*( c12a*cos(2.d0*phi) + c12b*sin(2.d0*phi) )/2.d0
          th3 = th*th2
          chi = chi + th3*( c21a*cos(     phi) + c21b*sin(     phi) )/3.d0
          chi = chi + th3*( c23a*cos(3.d0*phi) + c23b*sin(3.d0*phi) )/3.d0
          th4 = th*th3
          chi = chi + th4*c3/4.d0
          chi = chi + th4*( c32a*cos(2.d0*phi) + c32b*sin(2.d0*phi) )/4.d0
          chi = chi + th4*( c34a*cos(4.d0*phi) + c34b*sin(4.d0*phi) )/4.d0
          th5 = th*th4
          chi = chi + th5*( c41a*cos(     phi) + c41b*sin(     phi) )/5.d0
          chi = chi + th5*( c43a*cos(3.d0*phi) + c43b*sin(3.d0*phi) )/5.d0
          chi = chi + th5*( c45a*cos(5.d0*phi) + c45b*sin(5.d0*phi) )/5.d0
          th6 = th*th5
          chi = chi + th6*c5/6.d0
          chi = chi + th6*( c52a*cos(2.d0*phi) + c52b*sin(2.d0*phi) )/6.d0
          chi = chi + th6*( c54a*cos(4.d0*phi) + c54b*sin(4.d0*phi) )/6.d0
          chi = chi + th6*( c56a*cos(6.d0*phi) + c56b*sin(6.d0*phi) )/6.d0
          kwf(ix,iy) = kwf(ix,iy)*exp(-dcmplx(0,2)*pi*chi/(lambda))

          endif
!!!!!!!!!!!!!!!!!!!!!!!!!!
          if(angm.ne.0 .and. abs(kx-kcx)<1d-10 .and. abs(ky-kcy)<1d-10) kwf(ix,iy)=0.d0 ! remove the plane wave component with undefined phase
!!!!!!!!!!!!!!!!!!!!!!!!!!
        enddo
      enddo

      !print*,'call bfft2d'
      call bfft2d(kwf,wf)
      !print*,'bfft2d called'
      norm = sum(wf*dconjg(wf))
      wf = wf/sqrt(norm)
    endif
    write(*,'(a,i4,a,i4)')       '       Dimensions of the cell (pixels): ',ntx,' x ',nty
    write(*,'(a,i4,a,i4)')       '           Dimensions of the supercell: ',nscx,' x ',nscy
    if(conv>0) then
      write(*,'(a,f6.2,a,f6.2,a)') '                 Beam tilted by (mrad): (',dble(1000)*kcx/k,',',dble(1000)*kcy/k,')'
      write(*,'(a,f6.2,a,f6.2,a)') '            Beam shifted by (Angstrom): (',dble(shx)/dble(nx)*lat(1),',',dble(shy)/dble(ny)*lat(2),')'
    endif
    write(*,'(a)')        '======================================================================'
    
    if(print_psi_0) call output_array(wf,'(a,i0,a)','results/WF_',0,'.dat')

end subroutine init_wf
