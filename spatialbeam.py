"""
Define the structural analysis component using spatial beam theory.

Each FEM element has 6-DOF; translation in the x, y, and z direction and
rotation about the x, y, and z-axes.

"""

from __future__ import division, print_function
import numpy as np
from openmdao.api import Component, Group
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg import solve as scipy_solve
from scipy.linalg import det
from numpy.linalg import cond

try:
    import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex

def norm(vec):
    return np.sqrt(np.sum(vec**2))

def unit(vec):
    return vec / norm(vec)

def chords_fem(mesh):
    """
    Obtain the chord of the FEM element.
    This computes the chords by taking the average of the chords from the mesh.
    
    """
    vectors = mesh[-1, :, :] - mesh[0, :, :]
    chords_fem = np.sqrt(np.sum(vectors**2, axis=1))
    chords_fem = 0.5 * chords_fem[:-1] + 0.5 * chords_fem[1:] # Average chord between nodes
    return chords_fem

def _assemble_system(nodes, A, J, Iy, Iz, Kbt,
                     K_a, K_t, K_y, K_z,
                     cons, E, G, x_gl, T,
                     K_elem, S_a, S_t, S_y, S_z, S_k, T_elem,
                     const2, const_y, const_z, const_k, n, size, K):

    """
    Assemble the structural stiffness matrix based on 6 degrees of freedom
    per element.

    Can be run in Fortran or Python code depending on the flags used.
    Need to give area, torsion constant, area moments of inertia, E, G
    """

    # print(nodes, A, J, Iy, Iz)
    # Fortran
    #if fortran_flag:
    if False:
        K = OAS_API.oas_api.assemblestructmtx(nodes, A, J, Iy, Iz,
                                     K_a, K_t, K_y, K_z,
                                     cons, E, G, x_gl, T,
                                     K_elem, S_a, S_t, S_y, S_z, S_k, T_elem,
                                     const2, const_y, const_z)

    # Python
    else:
        K[:] = 0.
    
        # Loop over each element
        for ielem in range(n-1):
            # Obtain the element nodes
            P0 = nodes[ielem, :]
            P1 = nodes[ielem+1, :]

            x_loc = unit(P1 - P0)
            y_loc = unit(np.cross(x_loc, x_gl))
            z_loc = unit(np.cross(x_loc, y_loc))
            spar_ang = np.rad2deg(np.arccos(x_loc.dot(x_gl))) - 90
            print('angle = ',spar_ang) # this is correct, basically spar sweep angle (assuming projection)
            # local fibre angle is then theta - spar_ang where theta is the global fibre angle
            # what about removing the z-component of x_loc (and normalizing it), then dotting with x_gl?
            # x_loc_spar = x_loc.copy()
            # x_loc_spar[2] = 0
            # x_loc_spar = unit(x_loc_spar)
            # spar_ang = np.rad2deg(np.arccos(x_loc_spar.dot(x_gl))) - 90

            T[0, :] = x_loc
            T[1, :] = y_loc
            T[2, :] = z_loc

            for ind in range(4):
                T_elem[3*ind:3*ind+3, 3*ind:3*ind+3] = T

            L = norm(P1 - P0)
            EA_L = E * A[ielem] / L
            GJ_L = G * J[ielem] / L
            EIy_L3 = E * Iy[ielem] / L**3
            EIz_L3 = E * Iz[ielem] / L**3
            K2_GJL3 = Kbt[ielem]**2 / (G*J[ielem]*L**3)
            K_L = Kbt[ielem]/L

            K_a[:, :] = EA_L * const2
            K_t[:, :] = GJ_L * const2

            K_y[:, :] = EIy_L3 * const_y
            K_y[1, :] *= L
            K_y[3, :] *= L
            K_y[:, 1] *= L
            K_y[:, 3] *= L

            K_z[:, :] = EIz_L3 * const_z + K2_GJL3 * const_k
            K_z[1, :] *= L
            K_z[3, :] *= L
            K_z[:, 1] *= L
            K_z[:, 3] *= L

            K_elem[:] = 0
            K_elem += S_a.T.dot(K_a).dot(S_a)
            K_elem += S_t.T.dot(K_t).dot(S_t)
            K_elem += S_y.T.dot(K_y).dot(S_y)
            K_elem += S_z.T.dot(K_z).dot(S_z)
            K_elem += S_k * K_L

            res = T_elem.T.dot(K_elem).dot(T_elem)

            in0, in1 = ielem, ielem+1
            # Populate the full matrix with stiffness
            # contributions from each node
            K[6*in0:6*in0+6, 6*in0:6*in0+6] += res[:6, :6]
            K[6*in1:6*in1+6, 6*in0:6*in0+6] += res[6:, :6]
            K[6*in0:6*in0+6, 6*in1:6*in1+6] += res[:6, 6:]
            K[6*in1:6*in1+6, 6*in1:6*in1+6] += res[6:, 6:]

        # Include a scaled identity matrix in the rows and columns
        # corresponding to the structural constraints.
        # Hardcoded 1 constraint for now.
        for ind in range(1):
            for k in range(6):
                K[-6+k, 6*cons+k] = 1.e9
                K[6*cons+k, -6+k] = 1.e9

    return K


class ComputeNodes(Component):
    """
    Compute FEM nodes based on aerodynamic mesh.

    The FEM nodes are placed at fem_origin * chord.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Array defining the nodal points of the lifting surface.

    Returns
    -------
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.

    """

    def __init__(self, surface):
        super(ComputeNodes, self).__init__()

        self.ny = surface['num_y']
        self.nx = surface['num_x']
        # self.fem_origin = (surface['data_x_upper'][0] + surface['data_x_upper'][-1]) / 2.
        self.fem_origin = (surface['data_x_upper'][0] *(surface['data_y_upper'][0]-surface['data_y_lower'][0]) + \
        surface['data_x_upper'][-1]*(surface['data_y_upper'][-1]-surface['data_y_lower'][-1])) / \
        ( (surface['data_y_upper'][0]-surface['data_y_lower'][0]) + (surface['data_y_upper'][-1]-surface['data_y_lower'][-1]))

        self.add_param('mesh', val=np.zeros((self.nx, self.ny, 3), dtype=data_type))
        self.add_output('nodes', val=np.zeros((self.ny, 3), dtype=data_type))

    def solve_nonlinear(self, params, unknowns, resids):
        w = self.fem_origin
        mesh = params['mesh']
        unknowns['nodes'] = (1-w) * mesh[0, :, :] + w * mesh[-1, :, :]

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        w = self.fem_origin

        n = self.ny * 3
        jac['nodes', 'mesh'][:n, :n] = np.eye(n) * (1-w)
        jac['nodes', 'mesh'][:n, -n:] = np.eye(n) * w

        return jac


class AssembleK(Component):
    """
    Compute the displacements and rotations by solving the linear system
    using the structural stiffness matrix.

    Parameters
    ----------
    A[ny-1] : numpy array
        Areas for each FEM element.
    Iy[ny-1] : numpy array
        Area moment of inertia around the y-axis for each FEM element.
    Iz[ny-1] : numpy array
        Area moment of inertia around the z-axis for each FEM element.
    J[ny-1] : numpy array
        Polar moment of inertia for each FEM element.
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.

    Returns
    -------
    K[6*(ny+1), 6*(ny+1)] : numpy array
        Stiffness matrix for the entire FEM system. Used to solve the linear
        system K * u = f to obtain the displacements, u.
    """

    def __init__(self, surface):
        super(AssembleK, self).__init__()

        self.ny = surface['num_y']

        self.size = size = 6 * self.ny + 6

        self.add_param('A', val=np.ones((self.ny - 1), dtype=data_type))
        self.add_param('Iy', val=np.ones((self.ny - 1), dtype=data_type))
        self.add_param('Iz', val=np.ones((self.ny - 1), dtype=data_type))
        self.add_param('J', val=np.ones((self.ny - 1), dtype=data_type))
        self.add_param('nodes', val=np.ones((self.ny, 3), dtype=data_type))
        self.add_param('Kbt', val=np.ones((self.ny - 1), dtype=data_type))
        self.add_output('K', val=np.zeros((size, size), dtype=data_type))

        # Get material properties from the surface dictionary
        self.E = surface['E']
        self.G = surface['G']

        # Set up arrays to easily set up the K matrix
        self.const2 = np.array([
            [1, -1],
            [-1, 1],
        ], dtype=data_type)
        self.const_y = np.array([
            [12, -6, -12, -6],
            [-6, 4, 6, 2],
            [-12, 6, 12, 6],
            [-6, 2, 6, 4],
        ], dtype=data_type)
        self.const_z = np.array([
            [12, 6, -12, 6],
            [6, 4, -6, 2],
            [-12, -6, 12, -6],
            [6, 2, -6, 4],
        ], dtype=data_type)
        self.const_k = np.array([
            [-12, 6, 12, 6],
            [6, -3, -6, -3],
            [12, -6, -12, -6],
            [6, -3, -6, -3],
        ], dtype=data_type)
        self.x_gl = np.array([1, 0, 0], dtype=data_type)

        self.K_elem = np.zeros((12, 12), dtype=data_type)
        self.T_elem = np.zeros((12, 12), dtype=data_type)
        self.T = np.zeros((3, 3), dtype=data_type)

        self.K = np.zeros((size, size), dtype=data_type)
        self.forces = np.zeros((size), dtype=data_type)

        self.K_a = np.zeros((2, 2), dtype=data_type)
        self.K_t = np.zeros((2, 2), dtype=data_type)
        self.K_y = np.zeros((4, 4), dtype=data_type)
        self.K_z = np.zeros((4, 4), dtype=data_type)

        self.S_a = np.zeros((2, 12), dtype=data_type)
        self.S_a[(0, 1), (0, 6)] = 1.

        self.S_t = np.zeros((2, 12), dtype=data_type)
        self.S_t[(0, 1), (3, 9)] = 1.

        self.S_y = np.zeros((4, 12), dtype=data_type)
        self.S_y[(0, 1, 2, 3), (2, 4, 8, 10)] = 1.

        self.S_z = np.zeros((4, 12), dtype=data_type)
        self.S_z[(0, 1, 2, 3), (1, 5, 7, 11)] = 1.

        self.S_k = np.zeros((12, 12), dtype=data_type)
        self.S_k[(3, 5, 9, 11), (5, 3, 11, 9)] = 1.
        self.S_k[(3, 5, 9, 11), (11, 9, 5, 3)] = -1.

        if not fortran_flag:
            self.deriv_options['type'] = 'cs'

        self.surface = surface

        self.surface = surface

    def solve_nonlinear(self, params, unknowns, resids):

        # Find constrained nodes based on closeness to specified cg point
        symmetry = self.surface['symmetry']
        if symmetry:
            idx = self.ny - 1
        else:
            idx = (self.ny - 1) // 2
        self.cons = idx

        self.K = \
            _assemble_system(params['nodes'],
                             params['A'], params['J'], params['Iy'],
                             params['Iz'], params['Kbt'], self.K_a, self.K_t,
                             self.K_y, self.K_z, self.cons,
                             self.E, self.G, self.x_gl, self.T, self.K_elem,
                             self.S_a, self.S_t, self.S_y, self.S_z, self.S_k,
                             self.T_elem, self.const2, self.const_y,
                             self.const_z, self.const_k, self.ny, self.size,
                             self.K)

        unknowns['K'] = self.K

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):

        # Find constrained nodes based on closeness to specified cg point
        symmetry = self.surface['symmetry']
        if symmetry:
            idx = self.ny - 1
        else:
            idx = (self.ny - 1) // 2
        self.cons = idx
        nodes = params['nodes']

        A = params['A']
        J = params['J']
        Iy = params['Iy']
        Iz = params['Iz']
        Kbt = params['Kbt']

        if mode == 'fwd':
            K, Kd = OAS_API.oas_api.assemblestructmtx_d(nodes, dparams['nodes'], A, dparams['A'],
                                         J, dparams['J'], Iy, dparams['Iy'],
                                         Iz, dparams['Iz'],
                                         self.K_a, self.K_t, self.K_y, self.K_z,
                                         self.cons, self.E, self.G, self.x_gl, self.T,
                                         self.K_elem, self.S_a, self.S_t, self.S_y, self.S_z, self.S_k, self.T_elem,
                                         self.const2, self.const_y, self.const_z, self.const_k)

            dresids['K'] += Kd

        if mode == 'rev':
            nodesb, Ab, Jb, Iyb, Izb = OAS_API.oas_api.assemblestructmtx_b(nodes, A, J, Iy, Iz,
                                self.K_a, self.K_t, self.K_y, self.K_z,
                                self.cons, self.E, self.G, self.x_gl, self.T,
                                self.K_elem, self.S_a, self.S_t, self.S_y, self.S_z, self.S_k, self.T_elem,
                                self.const2, self.const_y, self.const_z, self.const_k, self.K, dresids['K'])

            dparams['nodes'] += nodesb
            dparams['A'] += Ab
            dparams['J'] += Jb
            dparams['Iy'] += Iyb
            dparams['Iz'] += Izb

class CreateRHS(Component):
    """
    Compute the right-hand-side of the K * u = f linear system to solve for the displacements.
    The RHS is based on the loads. For the aerostructural case, these are
    recomputed at each design point based on the aerodynamic loads.

    Parameters
    ----------
    loads[ny, 6] : numpy array
        Flattened array containing the loads applied on the FEM component,
        computed from the sectional forces.

    Returns
    -------
    forces[6*(ny+1)] : numpy array
        Right-hand-side of the linear system. The loads from the aerodynamic
        analysis or the user-defined loads.
    """

    def __init__(self, surface):
        super(CreateRHS, self).__init__()

        self.ny = surface['num_y']

        self.add_param('loads', val=np.zeros((self.ny, 6), dtype=data_type))
        self.add_output('forces', val=np.zeros(((self.ny+1)*6), dtype=data_type))

    def solve_nonlinear(self, params, unknowns, resids):
        # Populate the right-hand side of the linear system using the
        # prescribed or computed loads
        unknowns['forces'][:6*self.ny] = params['loads'].reshape(self.ny*6)

        # Remove extremely small values from the RHS so the linear system
        # can more easily be solved
        unknowns['forces'][np.abs(unknowns['forces']) < 1e-6] = 0.

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        n = self.ny * 6
        jac['forces', 'loads'][:n, :n] = np.eye((n))
        return jac

class SpatialBeamFEM(Component):
    """
    Compute the displacements and rotations by solving the linear system
    using the structural stiffness matrix.
    This component is copied from OpenMDAO's LinearSystem component with the
    names of the parameters and outputs changed to match our problem formulation.

    Parameters
    ----------
    K[6*(ny+1), 6*(ny+1)] : numpy array
        Stiffness matrix for the entire FEM system. Used to solve the linear
        system K * u = f to obtain the displacements, u.
    forces[6*(ny+1)] : numpy array
        Right-hand-side of the linear system. The loads from the aerodynamic
        analysis or the user-defined loads.

    Returns
    -------
    disp_aug[6*(ny+1)] : numpy array
        Augmented displacement array. Obtained by solving the system
        K * u = f, where f is a flattened version of loads.

    """

    def __init__(self, size):
        super(SpatialBeamFEM, self).__init__()

        self.add_param('K', val=np.zeros((size, size), dtype=data_type))
        self.add_param('forces', val=np.zeros((size), dtype=data_type))
        self.add_state('disp_aug', val=np.zeros((size), dtype=data_type))

        self.size = size

        # cache
        self.lup = None
        self.forces_cache = None

    def solve_nonlinear(self, params, unknowns, resids):
        """ Use np to solve Ax=b for x.
        """
        # print(np.min(params['K']))
        # lu factorization for use with solve_linear
        self.lup = lu_factor(params['K'])

        unknowns['disp_aug'] = lu_solve(self.lup, params['forces'])
        resids['disp_aug'] = params['K'].dot(unknowns['disp_aug']) - params['forces']

    def apply_nonlinear(self, params, unknowns, resids):
        """Evaluating residual for given state."""

        resids['disp_aug'] = params['K'].dot(unknowns['disp_aug']) - params['forces']

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ Apply the derivative of state variable with respect to
        everything."""

        if mode == 'fwd':

            if 'disp_aug' in dunknowns:
                dresids['disp_aug'] += params['K'].dot(dunknowns['disp_aug'])
            if 'K' in dparams:
                dresids['disp_aug'] += dparams['K'].dot(unknowns['disp_aug'])
            if 'forces' in dparams:
                dresids['disp_aug'] -= dparams['forces']

        elif mode == 'rev':

            if 'disp_aug' in dunknowns:
                dunknowns['disp_aug'] += params['K'].T.dot(dresids['disp_aug'])
            if 'K' in dparams:
                dparams['K'] += np.outer(unknowns['disp_aug'], dresids['disp_aug']).T
            if 'forces' in dparams:
                dparams['forces'] -= dresids['disp_aug']

    def solve_linear(self, dumat, drmat, vois, mode=None):
        """ LU backsubstitution to solve the derivatives of the linear system."""

        if mode == 'fwd':
            sol_vec, forces_vec = self.dumat, self.drmat
            t=0
        else:
            sol_vec, forces_vec = self.drmat, self.dumat
            t=1

        if self.forces_cache is None:
            self.forces_cache = np.zeros((self.size, ))
        forces = self.forces_cache

        for voi in vois:
            forces[:] = forces_vec[voi]['disp_aug']

            sol = lu_solve(self.lup, forces, trans=t)

            sol_vec[voi]['disp_aug'] = sol[:]


class SpatialBeamDisp(Component):
    """
    Reshape the flattened displacements from the linear system solution into
    a 2D array so we can more easily use the results.

    The solution to the linear system has meaingless entires due to the
    constraints on the FEM model. The displacements from this portion of
    the linear system are not needed, so we select only the relevant
    portion of the displacements for further calculations.

    Parameters
    ----------
    disp_aug[6*(ny+1)] : numpy array
        Augmented displacement array. Obtained by solving the system
        K * disp_aug = forces, where forces is a flattened version of loads.

    Returns
    -------
    disp[6*ny] : numpy array
        Actual displacement array formed by truncating disp_aug.

    """

    def __init__(self, surface):
        super(SpatialBeamDisp, self).__init__()

        self.ny = surface['num_y']

        self.add_param('disp_aug', val=np.zeros(((self.ny+1)*6), dtype=data_type))
        self.add_output('disp', val=np.zeros((self.ny, 6), dtype=data_type))

    def solve_nonlinear(self, params, unknowns, resids):
        # Obtain the relevant portions of disp_aug and store the reshaped
        # displacements in disp
        unknowns['disp'] = params['disp_aug'][:-6].reshape((-1, 6))

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        n = self.ny * 6
        jac['disp', 'disp_aug'][:n, :n] = np.eye((n))
        return jac


class SpatialBeamEnergy(Component):
    """ Compute strain energy.

    Parameters
    ----------
    disp[ny, 6] : numpy array
        Actual displacement array formed by truncating disp_aug.
    loads[ny, 6] : numpy array
        Array containing the loads applied on the FEM component,
        computed from the sectional forces.

    Returns
    -------
    energy : float
        Total strain energy of the structural component.

    """

    def __init__(self, surface):
        super(SpatialBeamEnergy, self).__init__()

        ny = surface['num_y']

        self.add_param('disp', val=np.zeros((ny, 6), dtype=data_type))
        self.add_param('loads', val=np.zeros((ny, 6), dtype=data_type))
        self.add_output('energy', val=0.)

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['energy'] = np.sum(params['disp'] * params['loads'])

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        jac['energy', 'disp'][0, :] = params['loads'].real.flatten()
        jac['energy', 'loads'][0, :] = params['disp'].real.flatten()
        return jac

class SpatialBeamWeight(Component):
    """ Compute total weight.

    Parameters
    ----------
    A[ny-1] : numpy array
        Areas for each FEM element.
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.

    Returns
    -------
    structural_weight : float
        Weight of the structural spar.
    cg_location[3] : numpy array
        Location of the structural spar's cg.

    """

    def __init__(self, surface):
        super(SpatialBeamWeight, self).__init__()

        self.surface = surface

        self.ny = surface['num_y']

        self.add_param('A', val=np.zeros((self.ny - 1), dtype=data_type))
        self.add_param('nodes', val=np.zeros((self.ny, 3), dtype=data_type))
        self.add_output('structural_weight', val=0.)
        self.add_output('cg_location', val=np.zeros((3), dtype=data_type))

    def solve_nonlinear(self, params, unknowns, resids):
        A = params['A']
        nodes = params['nodes']

        # Calculate the volume and weight of the total structure
        element_volumes = np.linalg.norm(nodes[1:, :] - nodes[:-1, :], axis=1) * A
        volume = np.sum(element_volumes)

        center_of_elements = (nodes[1:, :] + nodes[:-1, :]) / 2.
        cg_loc = np.sum(center_of_elements.T * element_volumes, axis=1) / volume

        weight = volume * self.surface['mrho'] * 9.81

        if self.surface['symmetry']:
            weight *= 2.
            cg_loc[1] = 0.

        unknowns['structural_weight'] = weight
        unknowns['cg_location'] = cg_loc

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()

        fd_jac = self.fd_jacobian(params, unknowns, resids,
                                        fd_params=['A', 'nodes'],
                                        fd_unknowns=['cg_location'],
                                        fd_states=[])
        jac.update(fd_jac)

        A = params['A']
        nodes = params['nodes']

        # First we will solve for dweight_dA
        # Calculate the volume and weight of the total structure
        norms = np.linalg.norm(nodes[1:, :] - nodes[:-1, :], axis=1).reshape(1, -1)

        # Multiply by the material density and force of gravity
        dweight_dA = norms * self.surface['mrho'] * 9.81

        # Account for symmetry
        if self.surface['symmetry']:
            dweight_dA *= 2.

        # Save the result to the jacobian dictionary
        jac['structural_weight', 'A'] = dweight_dA

        # Next, we will compute the derivative of weight wrt nodes.
        # Here we're using results from AD to compute the derivative
        # Initialize the reverse seeds.
        nodesb = np.zeros(nodes.shape)
        tempb = (nodes[1:, :] - nodes[:-1, :]) * (A / norms).reshape(-1, 1)
        nodesb[1:, :] += tempb
        nodesb[:-1, :] -= tempb

        # Apply the multipliers for material properties and symmetry
        nodesb *= self.surface['mrho'] * 9.81

        if self.surface['symmetry']:
            nodesb *= 2.

        # Store the flattened array in the jacobian dictionary
        jac['structural_weight', 'nodes'] = nodesb.reshape(1, -1)

        return jac

class SpatialBeamVonMisesTube(Component):
    """ Compute the von Mises stress in each element.

    Parameters
    ----------
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.
    radius[ny-1] : numpy array
        Radii for each FEM element.
    disp[ny, 6] : numpy array
        Displacements of each FEM node.

    Returns
    -------
    vonmises[ny-1, 2] : numpy array
        von Mises stress magnitudes for each FEM element.

    """

    def __init__(self, surface):
        super(SpatialBeamVonMisesTube, self).__init__()

        self.ny = surface['num_y']

        self.add_param('nodes', val=np.zeros((self.ny, 3),
                       dtype=complex))
        # self.add_param('radius', val=np.zeros((self.ny - 1),
        #                dtype=complex))
        self.add_param('disp', val=np.zeros((self.ny, 6),
                       dtype=complex))

        self.add_output('vonmises', val=np.zeros((self.ny-1, 4),
                        dtype=complex))
                        
        self.add_param('A', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_param('Iy', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_param('Iz', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_param('J', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_param('Kbt', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_param('A_enc', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_param('thickness', val=np.zeros((self.ny - 1)), dtype=complex)
        self.add_param('A_spar', val=np.ones((self.ny - 1),  dtype = complex))
        
        self.add_param('sparthickness', val=np.zeros((self.ny - 1)), dtype=complex)
        self.add_param('skinthickness', val=np.zeros((self.ny - 1)), dtype=complex)
        
        self.add_param('htop', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_param('hbottom', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_param('hleft', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_param('hright', val=np.zeros((self.ny - 1)), dtype=complex)
        
        # if not fortran_flag:
        self.deriv_options['type'] = 'cs'
        # self.deriv_options['form'] = 'central'

        self.E = surface['E']
        self.G = surface['G']

        self.T = np.zeros((3, 3), dtype=complex)
        self.x_gl = np.array([1, 0, 0], dtype=complex)
        self.t = 0
        
        self.tssf = top_skin_strength_factor = surface['strength_factor_for_upper_skin']

    def solve_nonlinear(self, params, unknowns, resids):
        # radius = params['radius']
        disp = params['disp']
        nodes = params['nodes']
        vonmises = unknowns['vonmises']
        A = params['A']
        A_enc = params['A_enc']
        A_spar = params['A_spar']
        Iy = params['Iy']
        Iz = params['Iz']
        J = params['J']
        Kbt = params['Kbt']
        # print("A is", A, "A_enc is",A_enc,"Iy",Iy,"Iz",Iz, "J", J)
        htop = params['htop']
        hbottom = params['hbottom']
        hleft = params['hleft']
        hright = params['hright']
        thickness = params['thickness']
        sparthickness = params['sparthickness']
        skinthickness = params['skinthickness']
        
        # print(thickness); exit()
        # print("htop is", htop, "hbottom", hbottom)
        
        T = self.T
        E = self.E
        G = self.G
        x_gl = self.x_gl

        num_elems = self.ny - 1
        for ielem in range(self.ny-1):

            P0 = nodes[ielem, :]
            P1 = nodes[ielem+1, :]
            L = norm(P1 - P0)
            
            x_loc = unit(P1 - P0)
            y_loc = unit(np.cross(x_loc, x_gl))
            z_loc = unit(np.cross(x_loc, y_loc))

            T[0, :] = x_loc
            T[1, :] = y_loc
            T[2, :] = z_loc

            u0x, u0y, u0z = T.dot(disp[ielem, :3])
            r0x, r0y, r0z = T.dot(disp[ielem, 3:])
            u1x, u1y, u1z = T.dot(disp[ielem+1, :3])
            r1x, r1y, r1z = T.dot(disp[ielem+1, 3:])
            
            
            axial_stress = E * (u1x - u0x) / L      # this is stress = modulus * strain; positive is tensile
            torsion_stress = G * J[ielem] / L * (r1x - r0x) / 2 / sparthickness[ielem] / A_enc[ielem]   # this is Torque / (2 * thickness_min * Area_enclosed)
            top_bending_stress = E / (L**2) * (6 * u0y + 2 * r0z * L - 6 * u1y + 4 * r1z * L ) * htop[ielem] # this is moment * htop / I  
            bottom_bending_stress = - E / (L**2) * (6 * u0y + 2 * r0z * L - 6 * u1y + 4 * r1z * L ) * hbottom[ielem] # this is moment * htop / I  
            left_bending_stress = - E / (L**2) * (-6 * u0z + 2 * r0y * L + 6 * u1z + 4 * r1y * L ) * hleft[ielem] # this is moment * htop / I  
            right_bending_stress = E / (L**2) * (-6 * u0z + 2 * r0y * L + 6 * u1z + 4 * r1y * L ) * hright[ielem] # this is moment * htop / I  
            
            vertical_shear = 1.5 / A_spar[ielem] * E * Iz[ielem] / (L**3) *(-12 * u0y - 6 * r0z * L + 12 * u1y - 6 * r1z * L ) # conservative estimate for shear due to bending
            
            # print("==========",ielem,"================")
            # print("vertical_shear", vertical_shear)
            # print("top",top_bending_stress)
            # print("bottom",bottom_bending_stress)
            # print("left",left_bending_stress)
            # print("right",right_bending_stress)
            # print("axial", axial_stress)
            # print("torsion", torsion_stress)
        
            vonmises[ielem, 0] = np.sqrt((top_bending_stress + right_bending_stress + axial_stress)**2 + 3*torsion_stress**2) / self.tssf
            vonmises[ielem, 1] = np.sqrt((bottom_bending_stress + left_bending_stress + axial_stress)**2 + 3*torsion_stress**2)
            vonmises[ielem, 2] = np.sqrt((left_bending_stress + axial_stress)**2 + 3*(torsion_stress-vertical_shear)**2) 
            vonmises[ielem, 3] = np.sqrt((right_bending_stress + axial_stress)**2 + 3*(torsion_stress+vertical_shear)**2) / self.tssf


class SpatialBeamFailureKS(Component):
    """
    Aggregate failure constraints from the structure.

    To simplify the optimization problem, we aggregate the individual
    elemental failure constraints using a Kreisselmeier-Steinhauser (KS)
    function.

    The KS function produces a smoother constraint than using a max() function
    to find the maximum point of failure, which produces a better-posed
    optimization problem.

    The rho parameter controls how conservatively the KS function aggregates
    the failure constraints. A lower value is more conservative while a greater
    value is more aggressive (closer approximation to the max() function).

    Parameters
    ----------
    vonmises[ny-1, 2] : numpy array
        von Mises stress magnitudes for each FEM element.

    Returns
    -------
    failure : float
        KS aggregation quantity obtained by combining the failure criteria
        for each FEM node. Used to simplify the optimization problem by
        reducing the number of constraints.

    """

    def __init__(self, surface, rho=100):
        super(SpatialBeamFailureKS, self).__init__()

        self.ny = surface['num_y']

        self.add_param('vonmises', val=np.zeros((self.ny-1, 4), dtype=data_type))
        self.add_output('failure', val=0.)

        self.sigma = surface['yield']
        self.rho = rho

    def solve_nonlinear(self, params, unknowns, resids):
        sigma = self.sigma
        rho = self.rho
        vonmises = params['vonmises']

        fmax = np.max(vonmises/sigma - 1)

        nlog, nsum, nexp = np.log, np.sum, np.exp
        ks = 1 / rho * nlog(nsum(nexp(rho * (vonmises/sigma - 1 - fmax))))
        unknowns['failure'] = fmax + ks

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        vonmises = params['vonmises']
        sigma = self.sigma
        rho = self.rho

        # Find the location of the max stress constraint
        fmax = np.max(vonmises / sigma - 1)
        i, j = np.where((vonmises/sigma - 1)==fmax)
        i, j = i[0], j[0]

        # Set incoming seed as 1 so we simply get the jacobian entries
        ksb = 1.

        # Use results from the AD code to compute the jacobian entries
        tempb0 = ksb / (rho * np.sum(np.exp(rho * (vonmises/sigma - fmax - 1))))
        tempb = np.exp(rho*(vonmises/sigma-fmax-1))*rho*tempb0
        fmaxb = ksb - np.sum(tempb)

        # Populate the entries
        derivs = tempb / sigma
        derivs[i, j] += fmaxb / sigma

        # Reshape and save them to the jac dict
        jac['failure', 'vonmises'] = derivs.reshape(1, -1)

        return jac

class SpatialBeamFailureExact(Component):
    """
    Outputs individual failure constraints on each FEM element.

    Parameters
    ----------
    vonmises[ny-1, 2] : numpy array
        von Mises stress magnitudes for each FEM element.

    Returns
    -------
    failure[ny-1, 2] : numpy array
        Array of failure conditions. Positive if element has failed.

    """

    def __init__(self, surface):
        super(SpatialBeamFailureExact, self).__init__()

        self.ny = surface['num_y']

        self.add_param('vonmises', val=np.zeros((self.ny-1, 4), dtype=data_type))
        self.add_output('failure', val=np.zeros((self.ny-1, 4), dtype=data_type))

        self.sigma = surface['yield']

    def solve_nonlinear(self, params, unknowns, resids):
        sigma = self.sigma
        vonmises = params['vonmises']

        unknowns['failure'] = vonmises/sigma - 1

    def linearize(self, params, unknowns, resids):
        return {('failure', 'vonmises') : np.eye(((self.ny-1)*2)) / self.sigma}

class SparWithinWing(Component):
    """

    .. warning::
        This component has not been extensively tested.
        It may require additional coding to work as intended.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Array defining the nodal points of the lifting surface.
    radius[ny-1] : numpy array
        Radius of each element of the FEM spar.

    Returns
    -------
    spar_within_wing[ny-1] : numpy array
        If all the values are negative, each element is within the wing,
        based on the surface's t_over_c value and the current chord.
        Set a constraint with
        `OASProblem.add_constraint('spar_within_wing', upper=0.)`
    """

    def __init__(self, surface):
        super(SparWithinWing, self).__init__()

        self.surface = surface
        self.ny = surface['num_y']
        self.nx = surface['num_x']

        self.add_param('mesh', val=np.zeros((self.nx, self.ny, 3), dtype=data_type))
        self.add_param('radius', val=np.zeros((self.ny-1), dtype=data_type))
        self.add_output('spar_within_wing', val=np.zeros((self.ny-1), dtype=data_type))

        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        mesh = params['mesh']
        max_radius = radii(mesh, self.surface['t_over_c'])
        unknowns['spar_within_wing'] = params['radius'] - max_radius

    def linearize(self, params, unknowns, resids):
        jac = {}
        jac['spar_within_wing', 'radius'] = -np.eye(self.ny-1)
        fd_jac = self.fd_jacobian(params, unknowns, resids,
                                        fd_params=['mesh'],
                                        fd_unknowns=['spar_within_wing'],
                                        fd_states=[])
        jac.update(fd_jac)
        return jac

class NonIntersectingThickness(Component):
    """

    Parameters
    ----------
    thickness[ny-1] : numpy array
        Thickness of each element of the FEM spar.
    radius[ny-1] : numpy array
        Radius of each element of the FEM spar.

    Returns
    -------
    thickness_intersects[ny-1] : numpy array
        If all the values are negative, each element does not intersect itself.
        If a value is positive, then the thickness within the hollow spar
        intersects itself and presents an impossible design.
        Add a constraint as
        `OASProblem.add_constraint('thickness_intersects', upper=0.)`
    """

    def __init__(self, surface):
        super(NonIntersectingThickness, self).__init__()

        self.ny = surface['num_y']
        self.nx = surface['num_x']

        self.add_param('thickness', val=np.zeros((self.ny-1)))
        self.add_param('radius', val=np.zeros((self.ny-1)))
        self.add_output('thickness_intersects', val=np.zeros((self.ny-1)))

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['thickness_intersects'] = params['thickness'] - params['radius']

    def linearize(self, params, unknowns, resids):
        jac = {}
        jac['thickness_intersects', 'thickness'] = np.eye(self.ny-1)
        jac['thickness_intersects', 'radius'] = -np.eye(self.ny-1)
        return jac

class SpatialBeamSetup(Group):
    """ Group that sets up the spatial beam components and assembles the
        stiffness matrix."""

    def __init__(self, surface):
        super(SpatialBeamSetup, self).__init__()

        self.add('nodes',
                 ComputeNodes(surface),
                 promotes=['*'])
        self.add('assembly',
                 AssembleK(surface),
                 promotes=['*'])

class SpatialBeamStates(Group):
    """ Group that contains the spatial beam states. """

    def __init__(self, surface):
        super(SpatialBeamStates, self).__init__()

        size = 6 * surface['num_y'] + 6

        self.add('create_rhs',
                 CreateRHS(surface),
                 promotes=['*'])
        self.add('fem',
                 SpatialBeamFEM(size),
                 promotes=['*'])
        self.add('disp',
                 SpatialBeamDisp(surface),
                 promotes=['*'])


class SpatialBeamFunctionals(Group):
    """ Group that contains the spatial beam functionals used to evaluate
    performance. """

    def __init__(self, surface):
        super(SpatialBeamFunctionals, self).__init__()

        # Commented out energy for now since we haven't ever used its output
        # self.add('energy',
        #          SpatialBeamEnergy(surface),
        #          promotes=['*'])
        self.add('structural_weight',
                 SpatialBeamWeight(surface),
                 promotes=['*'])
        self.add('vonmises',
                 SpatialBeamVonMisesTube(surface),
                 promotes=['*'])
        # self.add('thicknessconstraint',
        #          NonIntersectingThickness(surface),
        #          promotes=['*'])
        # The following component has not been fully tested so we leave it
        # commented out for now. Use at own risk.
        # self.add('sparconstraint',
        #          SparWithinWing(surface),
        #          promotes=['*'])

        if surface['exact_failure_constraint']:
            self.add('failure',
                     SpatialBeamFailureExact(surface),
                     promotes=['*'])
        else:
            self.add('failure',
                    SpatialBeamFailureKS(surface),
                    promotes=['*'])
