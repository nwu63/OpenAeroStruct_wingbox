!     -*- f90 -*-
!     This file is autogenerated with f2py (version:2)
!     It contains Fortran 90 wrappers to fortran functions.

      
      subroutine f2pyinitoas_api(f2pysetupfunc)
      use oas_api, only : compute_normals_b
      use oas_api, only : manipulate_mesh
      use oas_api, only : manipulate_mesh_d
      use oas_api, only : manipulate_mesh_b
      use oas_api, only : forcecalc
      use oas_api, only : forcecalc_d
      use oas_api, only : forcecalc_b
      use oas_api, only : momentcalc
      use oas_api, only : momentcalc_d
      use oas_api, only : momentcalc_b
      use oas_api, only : assemblestructmtx
      use oas_api, only : assemblestructmtx_d
      use oas_api, only : assemblestructmtx_b
      use oas_api, only : assembleaeromtx
      use oas_api, only : assembleaeromtx_d
      use oas_api, only : assembleaeromtx_b
      use oas_api, only : calc_vonmises
      use oas_api, only : calc_vonmises_b
      use oas_api, only : calc_vonmises_d
      use oas_api, only : transferdisplacements
      use oas_api, only : transferdisplacements_d
      use oas_api, only : transferdisplacements_b
      external f2pysetupfunc
      call f2pysetupfunc(compute_normals_b,manipulate_mesh,manipulate_me&
     &sh_d,manipulate_mesh_b,forcecalc,forcecalc_d,forcecalc_b,momentcal&
     &c,momentcalc_d,momentcalc_b,assemblestructmtx,assemblestructmtx_d,&
     &assemblestructmtx_b,assembleaeromtx,assembleaeromtx_d,assembleaero&
     &mtx_b,calc_vonmises,calc_vonmises_b,calc_vonmises_d,transferdispla&
     &cements,transferdisplacements_d,transferdisplacements_b)
      end subroutine f2pyinitoas_api


