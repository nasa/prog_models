from .. import deriv_prog_model

import math

class CentrefugalPump(deriv_prog_model.DerivProgModel):
    """
    Prognostics model for a centrefugal pump
    """
    events = [
        'ImpellerWearFailure',
        'PumpOilOverheat',
        'RadialBearingOverheat',
        'ImpellerWearFailure'
    ]
    
    inputs = [
        'Tamb',     # Ambient Temperature (K)
        'V',        # Voltage (V)
        'pdisch',   # Discharge Pressure (Pa)
        'psuc',     # Suction Pressure (Pa)
        'wsync'     # Syncronous Rotational Speed of Supply Voltage (rad/sec)
    ]

    states = [
        'A',
        'Q',
        'To',
        'Tr',
        'Tt',
        'rRadial',
        'rThrust',
        'w',
        'wRadial',
        'wThrust'
    ]

    outputs = [
        'Qoutm',# Discharge Flow (m^3/s)
        'Tom',  # Oil Temperature (K)
        'Trm',  # Radial Bearing Temperature (K)
        'Ttm',  # Thrust Bearing Temperature (K)
        'wm'    # Mechanical Rotation (rad/sec)
    ]

    parameters { # Set to defaults
        'cycleTime': 3600,  # length of a pump usage cycle

        # Environmental parameters
        'pAtm': 101325,     # Atmospheric Pressure (Pa)

        # Torque and pressure parameters
        'a0': 0.00149204,	# empirical coefficient for flow torque eqn
        'a1': 5.77703,		# empirical coefficient for flow torque eqn
        'a2': 9179.4,		# empirical coefficient for flow torque eqn
        'A': 12.7084,		# impeller blade area
        'b': 17984.6,

        # Pump/motor dynamics
        'I': 50,            # impeller/shaft/motor lumped inertia
        'r': 0.008,         # lumped friction parameter (minus bearing friction)
        'R1': 0.36,
        'R2': 0.076,
        'L1': 0.00063,

        # Flow coefficients
        'FluidI': 5,        # pump fluid inertia
        'c': 8.24123e-5,    # pump flow coefficient
        'cLeak': 1,         # internal leak flow coefficient
        'ALeak': 1e-10,     # internal leak area

        # Thrust bearing temperature
        'mcThrust': 7.3,
        'rThrust': 1.4e-6,
        'HThrust1': 0.0034,
        'HThrust2': 0.0026,

        # Radial bearing temperature
        'mcRadial': 2.4,
        'rRadial': 1.8e-6,
        'HRadial1': 0.0018,
        'HRadial2': 0.020,

        # Bearing oil temperature
        'mcOil': 8000,
        'HOil1': 1.0,
        'HOil2': 3.0,
        'HOil3': 1.5,

        # Parameter limits
        'lim': {
            'A': 9.5,
            'Tt': 370,
            'Tr': 370,
            'To': 350
        },

        # Initial state
        'x0': {
            'w': 376.9908, # 3600 rpm (rad/sec)
            'Q': 0,
            'Tt': 290,
            'Tr': 290,
            'To': 290,
            'A': 12.7084,
            'rThrust': 1.4e-6,
            'rRadial': 1.8e-6,
            'wA': 0,
            'wThrust': 0,
            'wRadial': 0
        }

        def initialize(self, u, z):
            return self.parameters['x0']

        def dx(self, t, x, u):
            pass

        def output(self, t, x):
            pass

        def event_state(self, t, x):
            pass

        def threshold_met(self, t, x):
            pass
    }