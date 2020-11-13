# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from .. import deriv_prog_model

from math import exp

class BatteryCircuit(deriv_prog_model.DerivProgModel):
    """
    Prognostics model for a battery, represented by an electric circuit
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

    def __init__(self, options = {}):
        self.parameters.update(options)
        super().__init__()

    def initialize(self, u, z):
        return self.parameters['x0']

    def dx(self, t, x, u): 
        Vcs = x['qcs']/self.parameters['Cs']
        Vcp = x['qcp']/self.parameters['Ccp']
        SOC = self.__soc(x['qb'])
        Cb = self.parameters['Cbp0']*SOC**3 + self.parameters['Cbp1']*SOC**2 + self.parameters['Cbp2']*SOC + self.parameters['Cbp3']
        Rcp = self.parameters['Rcp0'] + self.parameters['Rcp1']*exp(self.parameters['Rcp2']*(-SOC + 1))
        Vb = x['qb']/Cb
        Tbdot = (Rcp*self.parameters['Rs']*self.parameters['ha']*(self.parameters['Ta'] - x['tb']) + Rcp*Vcs**2*self.parameters['hcs'] + self.parameters['Rs']*Vcp**2*self.parameters['hcp']) \
                /(self.parameters['Jt']*Rcp*self.parameters['Rs'])
        Vp = Vb - Vcp - Vcs
        ip = Vp/self.parameters['Rp']
        ib = u['i'] + ip
        icp = ib - Vcp/Rcp
        ics = ib - Vcs/self.parameters['Rs']

        return self.apply_process_noise({
            'tb':  Tbdot,
            'qb':  -ib,
            'qcp': icp,
            'qcs': ics,
        })
    
    def __soc(self, qb):
        """
        Calculate SOC

        Created to avoid constructing dict using event_state when not necessary
        """
        return (self.parameters['CMax'] - self.parameters['qMax'] + qb)/self.parameters['CMax']
        
    def event_state(self, t, x):
        return {
            'EOD': self.__soc(x['qb'])
        }

    def output(self, t, x):
        Vcs = x['qcs']/self.parameters['Cs']
        Vcp = x['qcp']/self.parameters['Ccp']
        SOC = self.__soc(x['qb'])
        Cb = self.parameters['Cbp0']*SOC**3 + self.parameters['Cbp1']*SOC**2 + self.parameters['Cbp2']*SOC + self.parameters['Cbp3']
        Vb = x['qb']/Cb

        return {
            't': x['tb'],
            'v': Vb - Vcp - Vcs
        }

    def threshold_met(self, t, x):
        Vcs = x['qcs']/self.parameters['Cs']
        Vcp = x['qcp']/self.parameters['Ccp']
        SOC = self.__soc(x['qb'])
        Cb = self.parameters['Cbp0']*SOC**3 + self.parameters['Cbp1']*SOC**2 + self.parameters['Cbp2']*SOC + self.parameters['Cbp3']
        Vb = x['qb']/Cb
        V = Vb - Vcp - Vcs

        # Return true if voltage is less than the voltage threshold
        return {
             'EOD': V < self.parameters['VEOD']
        }
