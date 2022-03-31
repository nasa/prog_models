# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from .. import prognostics_model

from copy import deepcopy
import numpy as np
import warnings


class CentrifugalPumpBase(prognostics_model.PrognosticsModel):
    """
    Prognostics model for a Centrifugal Pump as described in the following paper:
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
        | wsync: Synchronous Rotational Speed of Supply Voltage (rad/sec)

    States: (9)
        | A: Impeller Area (m^2)
        | Q: Flow Rate into Pump (m^3/s)
        | To: Oil Temperature (K)
        | Tr: Radial Bearing Temperature (K)
        | Tt: Thrust Bearing Temperature (K)
        | rRadial: Radial (thrust) Friction Coefficient
        | rThrust: Rolling Friction Coefficient
        | w: Rotational Velocity of Pump (rad/sec)
        | QLeak: Leak Flow Rate (m^3/s)

    Outputs/Measurements: (5)
        | Qout: Discharge Flow (m^3/s)
        | To: Oil Temperature (K)
        | Tr: Radial Bearing Temperature (K)
        | Tt: Thrust Bearing Temperature (K)
        | w: Rotational Velocity of Pump (rad/sec)

    keyword args
    ------------
        process_noise : Optional, float or Dict[Srt, float]
          Process noise (applied at dx/next_state). 
          Can be number (e.g., .2) applied to every state, a dictionary of values for each 
          state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x) -> x
        process_noise_dist : Optional, String
          distribution for process noise (e.g., normal, uniform, triangular)
        measurement_noise : Optional, float or Dict[Srt, float]
          Measurement noise (applied in output eqn).
          Can be number (e.g., .2) applied to every output, a dictionary of values for each
          output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z
        measurement_noise_dist : Optional, String
          distribution for measurement noise (e.g., normal, uniform, triangular)
        pAtm : float
            Atmospheric pressure
        a0, a1, a2 : float
            empirical coefficients for flow torque eqn
        A : float
            impeller blade area
        b : float
        n : float
            Pole Phases 
        p : float
            Pole Pairs
        I : float
            impeller/shaft/motor lumped inertia
        r : float 
            lumped friction parameter (minus bearing friction)
        R1, R2 : float
        L1 : float
        FluidI: float
            Pump fluid inertia
        c : float
            Pump flow coefficient
        cLeak : float
            Internal leak flow coefficient
        ALeak : float
            Internal leak area
        mcThrust : float
        HThrust1, HThrust2 : float
        mcRadial : float
        HRadial1, HRadial2 : float
        mcOil : float
        HOil1, HOil2, HOil3 : float
        wA, wRadial, wThrust : float
            Wear rates. See also CentrifugalPumpWithWear
        lim : dict
            Parameter limits
        x0 : dict
            Initial state
    
    See Also
    --------
    CentrifugalPumpWithWear
    """
    events = ['ImpellerWearFailure', 'PumpOilOverheat', 'RadialBearingOverheat', 'ThrustBearingOverheat']
    inputs = ['Tamb', 'V', 'pdisch', 'psuc', 'wsync']
    states = ['w', 'Q', 'Tt', 'Tr', 'To', 'A', 'rRadial', 'rThrust', 'QLeak']
    outputs = ['w', 'Qout', 'Tt', 'Tr', 'To']
    is_vectorized = True

    default_parameters = {  # Set to defaults
        # Environmental parameters
        'pAtm': 101325,

        # Torque and pressure parameters
        'a0': 0.00149204,
        'a1': 5.77703,
        'a2': 9179.4,
        'A': 12.7084,
        'b': 17984.6,

        'n': 3,
        'p': 1,

        # Pump/motor dynamics
        'I': 50,
        'r': 0.008,
        'R1': 0.36,
        'R2': 0.076,
        'L1': 0.00063,

        # Flow coefficients
        'FluidI': 5,
        'c': 8.24123e-5,
        'cLeak': 1,
        'ALeak': 1e-10,

        # Thrust bearing temperature
        'mcThrust': 7.3,
        'HThrust1': 0.0034,
        'HThrust2': 0.0026,

        # Radial bearing temperature
        'mcRadial': 2.4,
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

        # Wear Rates
        'wA': 0,
        'wRadial': 0,
        'wThrust': 0,

        # Initial state
        'x0': {
            'w': 376.991118431,  # 3600 rpm (rad/sec)
            'Q': 0,
            'Tt': 290,
            'Tr': 290,
            'To': 290,
            'A': 12.7084,
            'rThrust': 1.4e-6,
            'rRadial': 1.8e-6
        }
    }

    state_limits = {
        'To': (0, np.inf),  # Limited by absolute zero (0 K)
        'Tr': (0, np.inf),  # Limited by absolute zero (0 K)
        'Tt': (0, np.inf),  # Limited by absolute zero (0 K)
        'A': (0, np.inf),
        'rThrust': (0, np.inf),
        'rRadial': (0, np.inf)
    }

    def initialize(self, u, z = None):
        x0 = self.parameters['x0']
        x0['QLeak'] = \
            self.parameters['cLeak']*self.parameters['ALeak']*\
                np.sqrt(abs(u['psuc']-u['pdisch'])) * np.sign(u['psuc']-u['pdisch'])
        return self.StateContainer(x0)

    def next_state(self, x, u, dt):
        params = self.parameters
        Todot = 1/params['mcOil'] * (params['HOil1']*(x['Tt']-x['To']) + params['HOil2']*(x['Tr']-x['To'])\
            + params['HOil3']*(u['Tamb']-x['To']))
        Ttdot = 1/params['mcThrust'] * (x['rThrust']*x['w']*x['w'] - params['HThrust1']*(x['Tt']-u['Tamb'])\
            - params['HThrust2']*(x['Tt']-x['To']))
        Adot = -params['wA']*x['Q']*x['Q']
        rRadialdot = params['wRadial']*x['rRadial']*x['w']*x['w']
        rThrustdot = params['wThrust']*x['rThrust']*x['w']*x['w']
        friction = (params['r']+x['rThrust']+x['rRadial'])*x['w']
        if type(x['A']) == np.ndarray:
            QLeak = np.array([params['cLeak']*params['ALeak'] *
                           np.sqrt(abs(u['psuc']-u['pdisch'])) * np.sign(u['psuc']-u['pdisch'])]*len(x['A']))
        else:
            QLeak = params['cLeak']*params['ALeak'] * \
                np.sqrt(abs(u['psuc']-u['pdisch'])) * np.sign(u['psuc']-u['pdisch'])
        Trdot = 1/params['mcRadial'] * (x['rRadial']*x['w']*x['w'] - params['HRadial1']*(x['Tr']-u['Tamb']) - params['HRadial2']*(x['Tr']-x['To']))
        slipn = (u['wsync']-x['w'])/(u['wsync'])
        ppump = x['A']*x['w']*x['w'] + params['b']*x['w']*x['Q']
        Qout = np.maximum(0,x['Q']-x['QLeak'])
        slip = np.maximum(-1,(np.minimum(1,slipn)))
        deltaP = ppump+u['psuc']-u['pdisch']
        Te = params['n']*params['p']*params['R2']/(slip*(u['wsync']+0.00001)) * u['V']**2 \
            /((params['R1']+params['R2']/slip)**2+(u['wsync']*params['L1'])**2)
        backTorque = -params['a2']*Qout**2 + params['a1']*x['w']*Qout + params['a0']*x['w']**2
        Qo = params['c']*np.sqrt(abs(deltaP)) * np.sign(deltaP)
        wdot = (Te-friction-backTorque)/params['I']
        Qdot = 1/params['FluidI']*(Qo-x['Q'])

        return self.StateContainer(np.array([
            [x['w'] + wdot * dt],
            [x['Q'] + Qdot * dt],
            [x['Tt'] + Ttdot * dt],
            [x['Tr'] + Trdot * dt],
            [x['To'] + Todot * dt],
            [x['A'] + Adot * dt],
            [x['rRadial'] + rRadialdot * dt],
            [x['rThrust'] + rThrustdot * dt],
            [QLeak]
        ]))

    def output(self, x):
        Qout = np.maximum(0,x['Q']-x['QLeak'])

        return self.OutputContainer({
            'w':    x['w'],
            'Qout': Qout,
            'Tt':   x['Tt'],
            'Tr':   x['Tr'],
            'To':   x['To']
        })

    def event_state(self, x):
        return {
            'ImpellerWearFailure': (x['A'] - self.parameters['lim']['A'])/(self.parameters['x0']['A'] - self.parameters['lim']['A']),
            'ThrustBearingOverheat': (self.parameters['lim']['Tt'] - x['Tt'])/(self.parameters['lim']['Tt']- self.parameters['x0']['Tt']),
            'RadialBearingOverheat': (self.parameters['lim']['Tr'] - x['Tr'])/(self.parameters['lim']['Tr']- self.parameters['x0']['Tr']),
            'PumpOilOverheat': (self.parameters['lim']['To'] - x['To'])/(self.parameters['lim']['To'] - self.parameters['x0']['To'])
        }

    def threshold_met(self, x):
        return {
            'ImpellerWearFailure': x['A'] <= self.parameters['lim']['A'],
            'ThrustBearingOverheat': x['Tt'] >= self.parameters['lim']['Tt'],
            'RadialBearingOverheat': x['Tr'] >= self.parameters['lim']['Tr'],
            'PumpOilOverheat': x['To'] >= self.parameters['lim']['To']
        }

def OverwrittenWarning(params):
    """
    Function to warn if overwritten changes
    """
    warnings.warn("Warning: wA, wRadial, and wThrust will be overwritten within the model, since the wear rates are part of the state. Use CentrifugalPumpBase to remove this behavior.")
    return {}


class CentrifugalPumpWithWear(CentrifugalPumpBase):
    """
    Prognostics model for a centrifugal pump with wear parameters as part of the model state. This is identical to CentrifugalPumpBase, only CentrifugalPumpBase has the wear params as parameters instead of states

    This class implements a Centrifugal Pump model as described in the following paper:
    `M. Daigle and K. Goebel, "Model-based Prognostics with Concurrent Damage Progression Processes," IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 43, no. 4, pp. 535-546, May 2013. https://www.researchgate.net/publication/260652495_Model-Based_Prognostics_With_Concurrent_Damage_Progression_Processes`

    Events (4) 
        See CentrifugalPumpBase

    Inputs/Loading: (5)
        See CentrifugalPumpBase
    
    States: (12)
        | States from CentrifugalPumpBase +
        | wA: Wear Rate for Impeller Area
        | wRadial: Wear Rate for Radial (thrust) Friction Coefficient
        | wRadial: Wear Rate for Rolling Friction Coefficient

    Outputs/Measurements: (5)
        See CentrifugalPumpBase

    Model Configuration Parameters:
        See CentrifugalPumpBase

    See Also
    --------
    CentrifugalPumpBase
    """
    inputs = CentrifugalPumpBase.inputs
    outputs = CentrifugalPumpBase.outputs
    states = CentrifugalPumpBase.states + ['wA', 'wRadial', 'wThrust']
    events = CentrifugalPumpBase.events

    default_parameters = deepcopy(CentrifugalPumpBase.default_parameters)
    default_parameters['x0'].update(
        {'wA': 0.0,
        'wThrust': 0,
        'wRadial': 0})

    state_limits = deepcopy(CentrifugalPumpBase.state_limits)

    param_callbacks = {
        'wA': [OverwrittenWarning],
        'wRadial': [OverwrittenWarning],
        'wThrust': [OverwrittenWarning]
    }

    def next_state(self, x, u, dt):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.parameters['wA'] = x['wA']
            self.parameters['wRadial'] = x['wRadial']
            self.parameters['wThrust'] = x['wThrust']
        next_x = CentrifugalPumpBase.next_state(self, x, u, dt)

        next_x.matrix = np.vstack((next_x.matrix, np.array([
            [x['wA']],
            [x['wRadial']],
            [x['wThrust']]
        ])))
        return next_x

CentrifugalPump = CentrifugalPumpWithWear
