# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from .. import deriv_prog_model

from math import exp

class BatteryCircuit(deriv_prog_model.DerivProgModel):
    """
    Prognostics model for a battery, represented by an electric circuit
    
    This class implements an equivilant circuit model as described in the following paper:
    `M. Daigle and S. Sankararaman, "Advanced Methods for Determining Prediction Uncertainty in Model-Based Prognostics with Application to Planetary Rovers," Annual Conference of the Prognostics and Health Management Society 2013, pp. 262-274, New Orleans, LA, October 2013. http://www.phmsociety.org/node/1055/`
    
    Events: (1)
        EOD: End of Discharge
    
    Inputs/Loading: (1)
        i: Current draw on the battery

    States: (4)
        tb, qb, qcp, qcs

    Outputs: (2)
        | t: Temperature of battery (°C)
        | v: Voltage supplied by battery
    
    Note: This is much quicker but also less accurate as the electrochemistry model
    """
    events = [
        'EOD' # End of Discharge
    ]
    
    inputs = [
        'i' # Current
    ]

    states = [
        'tb',
        'qb',
        'qcp',
        'qcs'
    ]

    outputs = [
        't', # Battery temperature (°C)
        'v'  # Battery Voltage
    ]

    parameters = { # Set to defaults
        'V0': 4.183,        # Nominal Battery Voltage
        'Rp': 1e4,          # Battery Parasitic Resistance
        'qMax': 7856.3254,  # Max Charge
        'CMax': 7777,       # Max Capacity
        'VEOD': 3.0,        # End of Discharge Voltage Threshold
        # Capacitance 
        'Cb0': 1878.155726,            # Battery Capacitance
        'Cbp0': -230,
        'Cbp1': 1.2,
        'Cbp2': 2079.9,
        'Cbp3': 27.055726,
        # R-C Pairs
        'Rs': 0.0538926, 
        'Cs': 234.387,
        'Rcp0': 0.0697776,
        'Rcp1': 1.50528e-17,
        'Rcp2': 37.223,
        'Ccp': 14.8223,
        # Temperature Parameters
        'Ta': 18.95,        # Ambient temperature (deg C)
        'Jt': 800,
        'ha': 0.5,          # Heat transfer coefficient, ambient
        'hcp': 19,
        'hcs': 1,
        'process_noise': 0.1, # Process noise
        'x0': {             # Default initial state
            'tb': 18.95,   
            'qb': 7856.3254,
            'qcp': 0,
            'qcs': 0
        }
    }

    def initialize(self, u, z):
        return self.parameters['x0']

    def dx(self, t, x, u): 
        parameters = self.parameters # Keep this here- accessing member can be expensive in python- this optimization reduces runtime by almost half!
        Vcs = x['qcs']/parameters['Cs']
        Vcp = x['qcp']/parameters['Ccp']
        SOC = (parameters['CMax'] - parameters['qMax'] + x['qb'])/parameters['CMax']
        Cb = parameters['Cbp0']*SOC**3 + parameters['Cbp1']*SOC**2 + parameters['Cbp2']*SOC + parameters['Cbp3']
        Rcp = parameters['Rcp0'] + parameters['Rcp1']*exp(parameters['Rcp2']*(-SOC + 1))
        Vb = x['qb']/Cb
        Tbdot = (Rcp*parameters['Rs']*parameters['ha']*(parameters['Ta'] - x['tb']) + Rcp*Vcs**2*parameters['hcs'] + parameters['Rs']*Vcp**2*parameters['hcp']) \
                /(parameters['Jt']*Rcp*parameters['Rs'])
        Vp = Vb - Vcp - Vcs
        ip = Vp/parameters['Rp']
        ib = u['i'] + ip
        icp = ib - Vcp/Rcp
        ics = ib - Vcs/parameters['Rs']

        return self.apply_process_noise({
            'tb':  Tbdot,
            'qb':  -ib,
            'qcp': icp,
            'qcs': ics,
        })
    
    def event_state(self, t, x):
        parameters = self.parameters
        return {
            'EOD': (parameters['CMax'] - parameters['qMax'] + x['qb'])/parameters['CMax']
        }

    def output(self, t, x):
        parameters = self.parameters
        Vcs = x['qcs']/parameters['Cs']
        Vcp = x['qcp']/parameters['Ccp']
        SOC = (parameters['CMax'] - parameters['qMax'] + x['qb'])/parameters['CMax']
        Cb = parameters['Cbp0']*SOC**3 + parameters['Cbp1']*SOC**2 + parameters['Cbp2']*SOC + parameters['Cbp3']
        Vb = x['qb']/Cb

        return {
            't': x['tb'],
            'v': Vb - Vcp - Vcs
        }

    def threshold_met(self, t, x):
        parameters = self.parameters
        Vcs = x['qcs']/parameters['Cs']
        Vcp = x['qcp']/parameters['Ccp']
        SOC = (parameters['CMax'] - parameters['qMax'] + x['qb'])/parameters['CMax']
        Cb = parameters['Cbp0']*SOC**3 + parameters['Cbp1']*SOC**2 + parameters['Cbp2']*SOC + parameters['Cbp3']
        Vb = x['qb']/Cb
        V = Vb - Vcp - Vcs

        # Return true if voltage is less than the voltage threshold
        return {
             'EOD': V < parameters['VEOD']
        }
