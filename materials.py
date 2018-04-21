from __future__ import division, print_function
import numpy as np

from openmdao.api import Component
def norm(vec):
    return np.sqrt(np.sum(vec**2))
def unit(vec):
    return vec / norm(vec)

def getQ(E1,E2,G12,nu12,ang):
    ang = ang/180*np.pi
    # T = lambda t:np.array([[np.cos(t)**2,np.sin(t)**2,2*np.sin(t)*np.cos(t)],
    #                        [np.sin(t)**2,np.cos(t)**2,-2*np.sin(t)*np.cos(t)],
    #                        [-np.sin(t)*np.cos(t),np.sin(t)*np.cos(t),np.cos(t)**2 - np.sin(t)**2]],dtype=complex)
    S = np.array([[1/E1,-nu12/E1,0],
                  [-nu12/E1,1/E2,0],
                  [0,0,1/G12]],dtype=complex)
    Q = np.linalg.inv(S)
    C = np.cos(ang)
    S = np.sin(ang)
    Qbar = np.zeros((3,3),dtype=complex)
    Qbar[0,0] = Q[0,0]*C**4 + Q[1,1]*S**4 + 2*(Q[0,1] + 2*Q[2,2])*S**2*C**2
    Qbar[0,1] = (Q[0,0] + Q[1,1] - 4*Q[2,2])*S**2*C**2 + Q[0,1]*(C**4 + S**4)
    Qbar[1,1] = Q[0,0]*S**4 + Q[1,1]*C**4 + 2*(Q[0,1] + 2*Q[2,2])*S**2*C**2
    Qbar[0,2] = (Q[0,0] - Q[0,1] - 2*Q[2,2])*C**3*S - (Q[1,1] - Q[0,1] - 2*Q[2,2])*C*S**3
    Qbar[1,2] = (Q[0,0] - Q[0,1] - 2*Q[2,2])*C*S**3 - (Q[1,1] - Q[0,1] - 2*Q[2,2])*C**3*S
    Qbar[2,2] = (Q[0,0] + Q[1,1] - 2*Q[0,1] - 2*Q[2,2])*S**2*C**2 + Q[2,2]*(S**4 + C**4)
    Qbar[2,1] = Qbar[1,2]
    Qbar[1,0] = Qbar[0,1]
    Qbar[2,0] = Qbar[0,2]
    # Qbar = np.linalg.inv(T(ang)).dot(Q)
    # Qbar = Qbar.dot(np.linalg.inv(T(ang)).T)
    return Qbar

def wingbox_props(chord, sparthickness, skinthickness, data_x_upper, data_x_lower, data_y_upper, data_y_lower, twist=0.):
    
    # Scale data points with chord 
    data_x_upper = chord * data_x_upper
    data_y_upper = chord * data_y_upper
    data_x_lower = chord * data_x_lower
    data_y_lower = chord * data_y_lower
    
    # Compute enclosed area for torsion constant
    # This currently does not change with twist
    A_enc = 0
    for i in range(data_x_upper.size-1):
        
        A_enc += (data_x_upper[i+1] - data_x_upper[i]) * (data_y_upper[i+1] + data_y_upper[i] - skinthickness ) / 2 # area above 0 line
        A_enc += (data_x_lower[i+1] - data_x_lower[i]) * (-data_y_lower[i+1] - data_y_lower[i] - skinthickness ) / 2 # area below 0 line

    A_enc -= (data_y_upper[0] - data_y_lower[0]) * sparthickness / 2 # area of spars
    A_enc -= (data_y_upper[-1] - data_y_lower[-1]) * sparthickness / 2 # area of spars

    # Compute perimeter to thickness ratio for torsion constant
    # This currently does not change with twist
    p_by_t = 0
    for i in range(data_x_upper.size-1):
        p_by_t += ((data_x_upper[i+1] - data_x_upper[i])**2 + (data_y_upper[i+1] - data_y_upper[i])**2)**0.5 / skinthickness # length / thickness of caps
        p_by_t += ((data_x_lower[i+1] - data_x_lower[i])**2 + (data_y_lower[i+1] - data_y_lower[i])**2)**0.5 / skinthickness # length / thickness of caps
        
    p_by_t += (data_y_upper[0] - data_y_lower[0] - skinthickness) / sparthickness # length / thickness of spars
    p_by_t += (data_y_upper[-1] - data_y_lower[-1] - skinthickness) / sparthickness # length / thickness of spars

    # Torsion constant
    J = 4 * A_enc**2 / p_by_t

    # Rotate the wingbox
    theta = twist

    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]],dtype=complex)

    data_x_upper_2 = data_x_upper.copy()
    data_y_upper_2 = data_y_upper.copy()
    data_x_lower_2 = data_x_lower.copy()
    data_y_lower_2 = data_y_lower.copy()
        
    for i in range(data_x_upper.size):
        
        data_x_upper_2[i] = rot_mat[0,0] * data_x_upper[i] + rot_mat[0,1] * data_y_upper[i]
        data_y_upper_2[i] = rot_mat[1,0] * data_x_upper[i] + rot_mat[1,1] * data_y_upper[i]

        data_x_lower_2[i] = rot_mat[0,0] * data_x_lower[i] + rot_mat[0,1] * data_y_lower[i]
        data_y_lower_2[i] = rot_mat[1,0] * data_x_lower[i] + rot_mat[1,1] * data_y_lower[i]
        
    data_x_upper = data_x_upper_2.copy()
    data_y_upper = data_y_upper_2.copy()
    data_x_lower = data_x_lower_2.copy()
    data_y_lower = data_y_lower_2.copy()
    
    # Compute area moment of inertia about x axis
    # First compute centroid and area
    first_moment_area_upper = 0
    upper_area = 0
    first_moment_area_lower = 0
    lower_area = 0
    for i in range(data_x_upper.size-1):
        first_moment_area_upper += ((data_y_upper[i+1] + data_y_upper[i]) / 2 - (skinthickness/2) ) * skinthickness * (data_x_upper[i+1] - data_x_upper[i])
        upper_area += skinthickness * (data_x_upper[i+1] - data_x_upper[i])
        
        first_moment_area_lower += ((data_y_lower[i+1] + data_y_lower[i]) / 2 + (skinthickness/2) ) * skinthickness * (data_x_lower[i+1] - data_x_lower[i])
        lower_area += skinthickness * (data_x_lower[i+1] - data_x_lower[i])

    area = upper_area + lower_area
    centroid = (first_moment_area_upper + first_moment_area_lower) / area
    
    # Then compute area moment of inertia for upward bending
    # This is calculated using derived analytical expression assuming linear interpolation between airfoil data points
    I_horiz = 0
    for i in range(data_x_upper.size-1): # upper surface
        a = (data_y_upper[i] - data_y_upper[i+1]) / (data_x_upper[i] - data_x_upper[i+1])
        b = (data_y_upper[i+1] - data_y_upper[i] + skinthickness) / 2
        x2 = data_x_upper[i+1] - data_x_upper[i]
        
        I_horiz += 2 * ((1./12. * a**3 * x2**4 + 1./3. * a**2 * x2**3 * b + 1./2. * a * x2**2 * b**2 + 1./3. * b**3 * x2))
        
        I_horiz += x2  * skinthickness * ((data_y_upper[i] + data_y_upper[i+1])/2 - skinthickness/2 - centroid)**2

    
    # Compute area moment of inertia about y axis
    for i in range(data_x_lower.size-1): # lower surface
        a = -(data_y_lower[i] - data_y_lower[i+1]) / (data_x_lower[i] - data_x_lower[i+1])
        b = (-data_y_lower[i+1] + data_y_lower[i] + skinthickness) / 2
        x2 = data_x_lower[i+1] - data_x_lower[i]
        
        I_horiz += 2 * ((1./12. * a**3 * x2**4 + 1./3. * a**2 * x2**3 * b + 1./2. * a * x2**2 * b**2 + 1./3. * b**3 * x2))
        
        I_horiz += x2 * skinthickness * ((-data_y_lower[i] - data_y_lower[i+1])/2 - skinthickness/2 + centroid)**2
    
    # Contribution from the forward spar
    I_horiz += 1./12. * sparthickness * (data_y_upper[0] - data_y_lower[0] - 2 * skinthickness)**3 + sparthickness * (data_y_upper[0] - data_y_lower[0] - 2 * skinthickness) * ((data_y_upper[0] + data_y_lower[0]) / 2 - centroid)**2
    # Contribution from the rear spar
    I_horiz += 1./12. * sparthickness * (data_y_upper[-1] - data_y_lower[-1] - 2 * skinthickness)**3 + sparthickness * (data_y_upper[-1] - data_y_lower[-1] - 2 * skinthickness) * ((data_y_upper[-1] + data_y_lower[-1]) / 2 - centroid)**2
    

    # Compute area moment of inertia for backward bending
    I_vert = 0
    first_moment_area_left = (data_y_upper[0] - data_y_lower[0]) * sparthickness * (data_x_upper[0] + sparthickness / 2)
    first_moment_area_right = (data_y_upper[-1] - data_y_lower[-1]) * sparthickness * (data_x_upper[-1] - sparthickness / 2)
    centroid_Ivert = (first_moment_area_left + first_moment_area_right) / \
                    ( ((data_y_upper[0] - data_y_lower[0]) + (data_y_upper[-1] - data_y_lower[-1])) * sparthickness)

    I_vert += 1./12. * (data_y_upper[0] - data_y_lower[0]) * sparthickness**3 + (data_y_upper[0] - data_y_lower[0]) * sparthickness * (centroid_Ivert - (data_x_upper[0] + sparthickness/2))**2
    I_vert += 1./12. * (data_y_upper[-1] - data_y_lower[-1]) * sparthickness**3 + (data_y_upper[-1] - data_y_lower[-1]) * sparthickness * (data_x_upper[-1] - sparthickness/2 - centroid_Ivert)**2
    
    # Add contribution of skins
    I_vert += 2 * ( 1./12. * skinthickness * (data_x_upper[-1] - data_x_upper[0] - 2 * sparthickness)**3 + skinthickness * (data_x_upper[-1] - data_x_upper[0] - 2 * sparthickness) * (centroid_Ivert - (data_x_upper[-1] + data_x_upper[0]) / 2)**2 )

    area_spar = ((data_y_upper[0] - data_y_lower[0] - 2 * skinthickness) + (data_y_upper[-1] - data_y_lower[-1] - 2 * skinthickness)) * sparthickness 
    area += area_spar
    
    # Distances for calculating max bending stresses (KS function used)
    ks_rho = 500. # Hard coded, see Martins and Poon 2005 for more
    fmax_upper = np.max(data_y_upper)
    htop = fmax_upper + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (data_y_upper - fmax_upper)))) - centroid
    
    fmax_lower = np.max(-data_y_lower)
    hbottom = fmax_lower + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (-data_y_lower - fmax_lower)))) + centroid
    
    hleft =  centroid_Ivert - data_x_upper[0]
    hright = data_x_upper[-1] - centroid_Ivert

    return I_horiz, I_vert, J, area, A_enc, htop, hbottom, hleft, hright, area_spar

def getModuli(chord, sparthickness, skinthickness, data_x_upper, data_x_lower, data_y_upper, data_y_lower, theta):
    # Scale data points with chord 
    data_x_upper = chord * data_x_upper
    data_y_upper = chord * data_y_upper
    data_x_lower = chord * data_x_lower
    data_y_lower = chord * data_y_lower

    # Compute average spacing in x and y, prior to rotation
    avg_x_dist = (data_x_upper[-1] - data_x_upper[0] + data_x_lower[-1] - data_x_lower[0])/2 - sparthickness
    avg_y_dist = (data_y_upper[0] - data_y_lower[0] + data_y_upper[-1] - data_y_lower[-1])/2 - skinthickness
    ########## Tim's Composites Data ###########
    E1_skin = 117.9E9
    E2_skin = 9.7E9
    G12_skin = 4.8E9
    nu12_skin = 0.34
    E1_spar = 62.1E9
    E2_spar = 62.1E9
    G12_spar = 5E9
    nu12_spar = 0.045
    fv_skin = np.array([0.625,0.125,0.125,0.125],dtype=complex)
    fv_spar = np.array([0.125,0.375,0.375,0.125],dtype=complex)
    ######### Aluminum ##################
    # E1_skin = 73.1e9
    # E2_skin = E1_skin
    # G12_skin = 30.e9
    # nu12_skin = E1_skin/(2*G12_skin) - 1
    # E1_spar = E1_skin
    # E2_spar = E2_skin
    # G12_spar = G12_skin
    # nu12_spar = nu12_skin
    # fv_skin = np.array([0.25,0.25,0.25,0.25],dtype=complex)
    # fv_spar = np.array([0.25,0.25,0.25,0.25],dtype=complex)
    
    ang = np.array([0,45,-45,90],dtype=complex)
    ang_skin = ang + theta # theta is desvar
    Qavg_skin = np.zeros((3,3),dtype=complex)
    Qavg_spar = np.zeros((3,3),dtype=complex)
    for ilayer in range(4):
        Q_spar = getQ(E1_spar,E2_spar,G12_spar,nu12_spar,ang[ilayer])
        Q_skin = getQ(E1_skin,E2_skin,G12_skin,nu12_skin,ang_skin[ilayer])
        Qavg_skin += Q_skin*fv_skin[ilayer]
        Qavg_spar += Q_spar*fv_spar[ilayer]
    Al = Qavg_skin*skinthickness
    Au = Al
    Aeff_skin = Al + Au
    Beff_skin = (avg_y_dist/2) * Au - (avg_y_dist/2) * Al
    Deff_skin = (avg_y_dist/2)**2 * Aeff_skin

    Af = Qavg_spar * sparthickness
    Ar = Af
    Aeff_spar = Af + Ar
    Beff_spar = (avg_x_dist/2) * Af - (avg_x_dist/2) * Ar
    Deff_spar = (avg_x_dist/2)**2 * Aeff_spar


    mat_skin = np.block([[Aeff_skin,Beff_skin],[Beff_skin,Deff_skin]])
    matinv_skin = np.linalg.inv(mat_skin)
    Ainv_skin = matinv_skin[0:3,0:3]
    E_skin = 1/Ainv_skin[0,0]/(skinthickness*2)
    G_skin = 1/Ainv_skin[2,2]/(skinthickness*2)

    mat_spar = np.block([[Aeff_spar,Beff_spar],[Beff_spar,Deff_spar]])
    matinv_spar = np.linalg.inv(mat_spar)
    Ainv_spar = matinv_spar[0:3,0:3]
    E_spar = 1/Ainv_spar[0,0]/(sparthickness*2)
    G_spar = 1/Ainv_spar[2,2]/(sparthickness*2)
    
    V_skin = avg_x_dist*skinthickness/(avg_x_dist*skinthickness + avg_y_dist*sparthickness)
    V_spar = avg_y_dist*sparthickness/(avg_x_dist*skinthickness + avg_y_dist*sparthickness)
    E = E_spar*V_spar + E_skin*V_skin
    G = G_spar*V_spar + G_skin*V_skin
    Kbt = 2 * avg_x_dist * Deff_skin[0,2]

    #Kbt = 2 * avg_x_dist * (Deff_skin[0,2] - Deff_skin[0,1]*Deff_skin[1,2]/Deff_skin[1,1])
    return E, G, Kbt

class ComputeModuli(Component):
    def __init__(self, surface):
        super(ComputeModuli, self).__init__()
        self.surface = surface
        self.ny = surface['num_y']
        self.data_x_upper = surface['data_x_upper']
        self.data_x_lower = surface['data_x_lower']
        self.data_y_upper = surface['data_y_upper']
        self.data_y_lower = surface['data_y_lower']

        self.add_param('nodes', val=np.ones((self.ny, 3), dtype=complex))
        self.add_param('theta', val=0.)
        self.add_param('chords_fem', val=np.ones((self.ny - 1), dtype = complex))
        self.add_param('sparthickness', val=np.ones((self.ny - 1), dtype = complex))
        self.add_param('skinthickness', val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('Kbt',     val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('E',     val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('G',     val=np.ones((self.ny - 1),  dtype = complex))
        self.deriv_options['type'] = 'cs'
        self.deriv_options['step_size'] = 1.0e-6
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['check_type'] = 'fd'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['check_step_size'] = 1.0e-6
        self.deriv_options['check_step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):
        self.x_gl = np.array([1, 0, 0], dtype=complex)
        for i in range(self.ny - 1):
            P0 = params['nodes'][i, :]
            P1 = params['nodes'][i+1, :]

            x_loc = unit(P1 - P0)
            spar_ang = 180*(np.arccos(x_loc.dot(self.x_gl)))/np.pi - 90
            theta = spar_ang - params['theta']
            # print('P0: ',P0)
            # print('P1: ',P1)
            # print('x_loc: ',x_loc)
            # print('param theta: ',params['theta'])
            # print('spar_ang: ',spar_ang)
            # print('rel theta: ',theta)
            #print(params['nodes'].dtype)
            # local fibre angle is then theta - spar_ang where theta is the global fibre angle
            # what about removing the z-component of x_loc (and normalizing it), then dotting with x_gl?
            # x_loc_spar = x_loc.copy()
            # x_loc_spar[2] = 0
            # x_loc_spar = unit(x_loc_spar)
            # spar_ang = np.rad2deg(np.arccos(x_loc_spar.dot(x_gl))) - 90


            unknowns['E'][i],unknowns['G'][i],unknowns['Kbt'][i] = getModuli(params['chords_fem'][i],\
            params['sparthickness'][i], params['skinthickness'][i],\
            self.data_x_upper, self.data_x_lower, self.data_y_upper, self.data_y_lower,theta)
            # if (x_loc.dtype == np.dtype('complex')):
            #     print(np.imag(unknowns['E']))
            # print('Kbt: ',unknowns['Kbt'])


class MaterialsTube(Component):
    """
    Compute geometric properties for a tube element.
    The thicknesses are added to the interior of the element, so the
    'radius' value is the outer radius of the tube.

    Parameters
    ----------
    radius : numpy array
        Outer radii for each FEM element.
    thickness : numpy array
        Tube thickness for each FEM element.

    Returns
    -------
    A : numpy array
        Cross-sectional area for each FEM element.
    Iy : numpy array
        Area moment of inertia around the y-axis for each FEM element.
    Iz : numpy array
        Area moment of inertia around the z-axis for each FEM element.
    J : numpy array
        Polar moment of inertia for each FEM element.
    """

    def __init__(self, surface):
        super(MaterialsTube, self).__init__()

        self.surface = surface

        self.ny = surface['num_y']
        self.mesh = surface['mesh']
        name = surface['name']
        
        self.data_x_upper = surface['data_x_upper']
        self.data_x_lower = surface['data_x_lower']
        self.data_y_upper = surface['data_y_upper']
        self.data_y_lower = surface['data_y_lower']

        # self.add_param('radius', val=np.ones((self.ny - 1)))
        self.add_param('chords_fem', val=np.ones((self.ny - 1), dtype = complex))
        self.add_param('twist_fem',  val=np.ones((self.ny - 1),  dtype = complex))
        self.add_param('thickness',  val=np.ones((self.ny - 1), dtype = complex))
        
        self.add_param('sparthickness', val=np.ones((self.ny - 1), dtype = complex))
        self.add_param('skinthickness', val=np.ones((self.ny - 1),  dtype = complex))
        
        self.add_output('A',       val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('A_enc',   val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('A_spar',  val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('Iy',      val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('Iz',      val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('J',       val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('htop',    val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('hbottom', val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('hleft',   val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('hright',  val=np.ones((self.ny - 1),  dtype = complex))

        self.arange = np.arange((self.ny - 1))
        
        self.deriv_options['type'] = 'cs'
        self.deriv_options['step_size'] = 1.0e-10
        self.deriv_options['step_calc'] = 'relative'

        self.deriv_options['check_type'] = 'fd'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['check_step_size'] = 1.0e-8
        self.deriv_options['check_step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):
        
        for i in range(self.ny - 1):
            
            unknowns['Iz'][i], unknowns['Iy'][i], unknowns['J'][i], unknowns['A'][i], unknowns['A_enc'][i],\
            unknowns['htop'][i], unknowns['hbottom'][i], unknowns['hleft'][i], unknowns['hright'][i],\
            unknowns['A_spar'][i]  = \
            wingbox_props(params['chords_fem'][i], params['sparthickness'][i], params['skinthickness'][i],\
            self.data_x_upper, self.data_x_lower, self.data_y_upper, self.data_y_lower, -params['twist_fem'][i])
