"""
Trimmed design optimization of the ScanEagle UAV.

Some of the finer levels may not converge well for aerostructural optimization.
"""

from __future__ import division, print_function
import sys
from time import time
import numpy as np

# Append the parent directory to the system path so we can call those Python
# files. If you have OpenAeroStruct in your PYTHONPATH, this is not necessary.
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from OpenAeroStruct import OASProblem

# Can do 'aerostruct', 'aero', or 'struct'
prob_type = 'aerostruct'

# L1 is finest, L3 is coarsest. Can do L1, L1.5, L2, L2.5, L3
mesh_level = 'L1'
print(mesh_level)

solver_options = ['gs_wo_aitken', 'gs_w_aitken', 'newton_gmres', 'hybrid_GSN', 'newton_direct']
solver_combo = solver_options[2] 
solver_atol = 1e-7

# Set problem type
prob_dict = {'optimize' : True,
             'type' : prob_type,
             'compute_static_margin' : True,
             'optimizer' : 'SLSQP',
             'with_viscous' : True,
             'W0' : 14.,  # kg, empty weight from manufacturer
             'a' : 322.2,  # m/s, at 15,000 ft
             'rho' : 0.770816, # kg/m^3, at 15,000 ft
             'R' : 2500e3, # estimated range based on cruise speed and flight endurance
             'CT' : 9.80665 * 8.6e-6,  # piston-prop estimation from Raymer
             'Re' : 4e5,
             'M' : .093, # calc'd from 30 m/s cruise speed
             'cg' : np.array([.4, 0., 0.]),  # estimated based on aircraft pictures
             'solver_combo' : solver_combo,
             'solver_atol' : solver_atol,
             'print_level' : 2
             }

# Instantiate problem and add default surface
OAS_prob = OASProblem(prob_dict)

zshear_cp = np.zeros(10)
zshear_cp[0] = .3

xshear_cp = np.zeros(10)
xshear_cp[0] = .15

chord_cp = np.ones(10)
chord_cp[0] = .5

radius_cp = 0.02  * np.ones(10)
radius_cp[0] = 0.015

if mesh_level == 'L1':
    num_y = 101
    num_x = 5
    spacing = .2
elif mesh_level == 'L1.5':
    num_y = 41
    num_x = 3
    spacing = .5
elif mesh_level == 'L2':
    num_y = 21
    num_x = 3
    spacing = 1.
elif mesh_level == 'L2.5':
    num_y = 15
    num_x = 2
    spacing = 1.
else:
    num_y = 7
    num_x = 2
    spacing = 1.

# Create a dictionary to store options about the surface
surf_dict = {'num_y' : num_y,
             'num_x' : num_x,
             'wing_type' : 'rect',
             'symmetry' : True,
             'span_cos_spacing' : spacing,
             'span' : 6.11,
             'root_chord' : .3,  # estimate
             'sweep' : 30.,
             'taper' : .8,
             'zshear_cp' : zshear_cp,
             'xshear_cp' : xshear_cp,
             'chord_cp' : chord_cp,

             # Material properties taken from http://www.performance-composites.com/carbonfibre/mechanicalproperties_2.asp
             'E' : 45.e9,
             'G' : 15.e9,
             'yield' : 350.e6 / 1.25 / 2.5,
             'mrho' : 1.6e3,
             'CD0' : 0.015,

             }

# Add the specified wing surface to the problem
OAS_prob.add_surface(surf_dict)

if prob_type == 'aero':

    # Setup problem and add design variables, constraint, and objective
    # OAS_prob.add_desvar('alpha', lower=-10., upper=15.)
    OAS_prob.add_desvar('wing.twist_cp', lower=-10., upper=15.)

    # OAS_prob.add_desvar('wing.chord_cp', lower=0.5, upper=3.)
    # OAS_prob.add_desvar('wing.xshear_cp', lower=-10., upper=15.)
    # OAS_prob.add_desvar('wing.sweep', lower=-60., upper=60.)
    # OAS_prob.add_desvar('wing.taper', lower=.5, upper=2.)
    OAS_prob.add_constraint('wing_perf.CL', equals=0.6032)
    # OAS_prob.add_constraint('CM', equals=0.)
    OAS_prob.add_objective('wing_perf.CD', scaler=1e3)

else:

    # Add design variables, constraint, and objective on the problem
    # OAS_prob.add_desvar('alpha', lower=-10., upper=10.)
    OAS_prob.add_constraint('L_equals_W', equals=0.)
    OAS_prob.add_objective('fuelburn', scaler=0.1)

    # Setup problem and add design variables, constraint, and objective
    OAS_prob.add_desvar('wing.twist_cp', lower=-15., upper=15.)
    OAS_prob.add_desvar('wing.thickness_cp', lower=0.0001, upper=0.5, scaler=1e3)
    OAS_prob.add_desvar('wing.sweep', lower=-60., upper=60., scaler=1e-1)
    OAS_prob.add_constraint('wing_perf.failure', upper=0.)
    OAS_prob.add_constraint('wing_perf.thickness_intersects', upper=0.)
    OAS_prob.add_constraint('CM', equals=0.)

OAS_prob.setup()

st = time()

# Actually run the problem
OAS_prob.run()

print("\nWing CL:", OAS_prob.prob['wing_perf.CL'])
print("Wing CD:", OAS_prob.prob['wing_perf.CD'])
print("Time elapsed: {} secs".format(time() - st))
