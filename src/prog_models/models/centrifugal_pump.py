# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from .. import prognostics_model

import math

class CentrifugalPump(prognostics_model.PrognosticsModel):
    """
    Prognostics model for a centrifugal pump

    This class implements a Centrifugal Pump model as described in the following paper:
    `M. Daigle and K. Goebel, "Model-based Prognostics with Concurrent Damage Progression Processes," IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 43, no. 4, pp. 535-546, May 2013. https://www.researchgate.net/publication/260652495_Model-Based_Prognostics_With_Concurrent_Damage_Progression_Processes`

    Events (4)
        | ImpellerWearFailure: Failure of the impeller due to wear
        | PumpOilOverheat: Overheat of the pump oil
        | RadialBearingOverheat: Overheat of the radial bearing
        | ThrustBearingOverheat: Overhead of the thrust bearing

    Inputs/Loading: (5)
        | Tamb: Ambient Temperature (K)
        | V: Voltage
        | pdisch: Discharge Pressure (Pa)
        | psuc: Suction Pressure (Pa)
        | wsync: Syncronous Rotational Speed of Supply Voltage (rad/sec)

    States: (12)
        A, Q, To, Tr, Tt, rRadial, rThrust, w, wA, wRadial, wThrust, QLeak

    Outputs/Measurements: (5)
        | Qout: Discharge Flow (m^3/s)
        | To: Oil Temperature (K)
        | Tr: Radial Bearing Temperature (K)
        | Tt: Thrust Bearing Temperature (K)
        | w: Mechanical Rotation (rad/sec)

    Model Configuration Parameters:
        | process_noise : Process noise (applied at dx/next_state). 
                    Can be number (e.g., .2) applied to every state, a dictionary of values for each 
                    state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x) -> x
        | process_noise_dist : Optional, distribution for process noise (e.g., normal, uniform, triangular)
        | measurement_noise : Measurement noise (applied in output eqn)
                    Can be number (e.g., .2) applied to every output, a dictionary of values for each 
                    output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z
        | measurement_noise_dist : Optional, distribution for measurement noise (e.g., normal, uniform, triangular)
        | pAtm : Atmospheric pressure
        | a0, a1, a2 : empirical coefficients for flow torque eqn
        | A : impeller blade area
        | b : 
        | I : impeller/shaft/motor lumped inertia
        | r : lumped friction parameter (minus bearing friction)
        | R1, R2 :
        | L1 : 
        | FluidI: Pump fluid inertia
        | c : Pump flow coefficient
        | cLeak : Internal leak flow coefficient
        | ALeak : Internal leak area
        | mcThrust : 
        | rThrust :
        | HThrust1, HThrust2 : 
        | mcRadial :
        | rRadial :
        | HRadial1, HRadial2 :
        | mcOil : 
        | HOil1, HOil2, HOil3 : 
        | lim : Parameter limits (dict)
        | x0 : Initial state
    """
    events = ['ImpellerWearFailure', 'PumpOilOverheat', 'RadialBearingOverheat', 'ThrustBearingOverheat']
    inputs = ['Tamb', 'V', 'pdisch', 'psuc', 'wsync']
    states = ['A', 'Q', 'To', 'Tr', 'Tt', 'rRadial', 'rThrust', 'w', 'wA', 'wRadial', 'wThrust', 'QLeak']
    outputs = ['Qout', 'To', 'Tr', 'Tt', 'w']

    default_parameters = { # Set to defaults
        # Environmental parameters
        'pAtm': 101325,

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
            'w': 376.991118431, # 3600 rpm (rad/sec)
            'Q': 0,
            'Tt': 290,
            'Tr': 290,
            'To': 290,
            'A': 12.7084,
            'rThrust': 1.4e-6,
            'rRadial': 1.8e-6,
            'wA': 0.0,
            'wThrust': 0,
            'wRadial': 0,
        }
    }

    def initialize(self, u, z = None):
        x0 = self.parameters['x0']
        x0['QLeak'] = math.copysign(self.parameters['cLeak']*self.parameters['ALeak']*math.sqrt(abs(u['psuc']-u['pdisch'])), u['psuc']-u['pdisch'])
        return x0

    def next_state(self, x, u, dt):
        Todot = 1/self.parameters['mcOil'] * (self.parameters['HOil1']*(x['Tt']-x['To']) + self.parameters['HOil2']*(x['Tr']-x['To']) + self.parameters['HOil3']*(u['Tamb']-x['To']))
        Ttdot = 1/self.parameters['mcThrust'] * (x['rThrust']*x['w']*x['w'] - self.parameters['HThrust1']*(x['Tt']-u['Tamb']) - self.parameters['HThrust2']*(x['Tt']-x['To']))
        Adot = -x['wA']*x['Q']*x['Q']
        rRadialdot = x['wRadial']*x['rRadial']*x['w']*x['w']
        rThrustdot = x['wThrust']*x['wThrust']*x['w']*x['w']
        friction = (self.parameters['r']+x['rThrust']+x['rRadial'])*x['w']
        QLeak = math.copysign(self.parameters['cLeak']*self.parameters['ALeak']*math.sqrt(abs(u['psuc']-u['pdisch'])), u['psuc']-u['pdisch'])
        Trdot = 1/self.parameters['mcRadial'] * (x['rRadial']*x['w']*x['w'] - self.parameters['HRadial1']*(x['Tr']-u['Tamb']) - self.parameters['HRadial2']*(x['Tr']-x['To']))
        slipn = (u['wsync']-x['w'])/(u['wsync'])
        ppump = x['A']*x['w']*x['w'] + self.parameters['b']*x['w']*x['Q']
        Qout = max(0,x['Q']-QLeak)
        slip = max(-1,(min(1,slipn)))
        deltaP = ppump+u['psuc']-u['pdisch']
        Te = 3*self.parameters['R2']/slip/(u['wsync']+0.00001)*u['V']**2/((self.parameters['R1']+self.parameters['R2']/slip)**2+(u['wsync']*self.parameters['L1'])**2)
        backTorque = -self.parameters['a2']*Qout**2 + self.parameters['a1']*x['w']*Qout + self.parameters['a0']*x['w']**2
        Qo = math.copysign(self.parameters['c']*math.sqrt(abs(deltaP)), deltaP)
        wdot = (Te-friction-backTorque)/self.parameters['I']
        Qdot = 1/self.parameters['FluidI']*(Qo-x['Q'])
        QLeak = math.copysign(self.parameters['cLeak']*self.parameters['ALeak']*math.sqrt(abs(u['psuc']-u['pdisch'])), u['psuc']-u['pdisch'])
        return self.apply_process_noise({
            'A': x['A'] + Adot*dt,
            'Q': x['Q'] + Qdot*dt, 
            'To': x['To'] + Todot*dt,
            'Tr': x['Tr'] + Trdot*dt,
            'Tt': x['Tt'] + Ttdot*dt,
            'rRadial': x['rRadial'] + rRadialdot*dt,
            'rThrust': x['rThrust'] + rThrustdot*dt,
            'w': x['w']+wdot*dt,
            'wA': x['wA'],
            'wRadial': x['wRadial'],
            'wThrust': x['wThrust'],
            'QLeak': QLeak
        }, dt)

    def output(self, x):
        Qout = max(0,x['Q']-x['QLeak'])

        return self.apply_measurement_noise({
            'Qout': Qout,
            'To': x['To'],
            'Tr': x['Tr'],
            'Tt': x['Tt'],
            'w': x['w']
        })

    def event_state(self, x):
        return {
            'ImpellerWearFailure': (x['A'] - self.parameters['lim']['A'])/(self.parameters['x0']['A'] - self.parameters['lim']['A']),
            'ThrustBearingOverheat': (self.parameters['lim']['Tt'] - x['Tt'])/(self.parameters['x0']['Tt'] - self.parameters['lim']['Tt']),
            'RadialBearingOverheat': (self.parameters['lim']['Tr'] - x['Tr'])/(self.parameters['x0']['Tr'] - self.parameters['lim']['Tr']),
            'PumpOilOverheat': (self.parameters['lim']['To'] - x['To'])/(self.parameters['x0']['To'] - self.parameters['lim']['To'])
        }

    def threshold_met(self, x):
        return {
            'ImpellerWearFailure': x['A'] <= self.parameters['lim']['A'],
            'ThrustBearingOverheat': x['Tt'] >= self.parameters['lim']['Tt'],
            'RadialBearingOverheat': x['Tr'] >= self.parameters['lim']['Tr'],
            'PumpOilOverheat': x['To'] >= self.parameters['lim']['To']
        }
