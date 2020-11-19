from .. import prognostics_model

import math

class BatteryCircuit(prognostics_model.PrognosticsModel):
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
        't', # Battery temperature
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
        'x0': {             # Default initial state
            'tb': 18.95,   
            'qb': 7856.3254,
            'qcp': 0,
            'qcs': 0
        }
    }

    def initialize(self, u, z):
        return self.parameters['x0']

    # TODO(CT): Differential Model parent class 
    #   Which then defines state = delta() + state

    def next_state(self, t, x, u, dt): 
        Vcs = x['qcs']/self.parameters['Cs']
        Vcp = x['qcp']/self.parameters['Ccp']
        SOC = self.event_state(t, x)['EOD']
        Cb = self.parameters['Cbp0']*SOC**3 + self.parameters['Cbp1']*SOC**2 + self.parameters['Cbp2']*SOC + self.parameters['Cbp3']
        Rcp = self.parameters['Rcp0'] + self.parameters['Rcp1']*math.exp(self.parameters['Rcp2']*(-SOC + 1))
        Vb = x['qb']/Cb
        Tbdot = (Rcp*self.parameters['Rs']*self.parameters['ha']*(self.parameters['Ta'] - x['tb']) + Rcp*Vcs**2*self.parameters['hcs'] + self.parameters['Rs']*Vcp**2*self.parameters['hcp']) \
                /(self.parameters['Jt']*Rcp*self.parameters['Rs'])
        Vp = Vb - Vcp - Vcs
        ip = Vp/self.parameters['Rp']
        ib = u['i'] + ip
        icp = ib - Vcp/Rcp
        qcpdot = icp
        qbdot = -ib
        ics = ib - Vcs/self.parameters['Rs']
        qcsdot = ics

        return {
            'tb':  x['tb'] + Tbdot*dt,
            'qb':  x['qb'] + qbdot*dt,
            'qcp': x['qcp'] + qcpdot*dt,
            'qcs': x['qcs'] + qcsdot*dt,
        }
        
    def event_state(self, t, x):
        return {
            'EOD': (self.parameters['CMax'] - self.parameters['qMax'] + x['qb'])/self.parameters['CMax']
        }

    def output(self, t, x):
        Vcs = x['qcs']/self.parameters['Cs']
        Vcp = x['qcp']/self.parameters['Ccp']
        SOC = self.event_state(t, x)['EOD']
        Cb = self.parameters['Cbp0']*SOC**3 + self.parameters['Cbp1']*SOC**2 + self.parameters['Cbp2']*SOC + self.parameters['Cbp3']
        Vb = x['qb']/Cb

        return {
            't': x['tb'],
            'v': Vb - Vcp - Vcs
        }

    def threshold_met(self, t, x):
        Vcs = x['qcs']/self.parameters['Cs']
        Vcp = x['qcp']/self.parameters['Ccp']
        SOC = self.event_state(t, x)['EOD']
        Cb = self.parameters['Cbp0']*SOC**3 + self.parameters['Cbp1']*SOC**2 + self.parameters['Cbp2']*SOC + self.parameters['Cbp3']
        Vb = x['qb']/Cb
        V = Vb - Vcp - Vcs

        # Return true if voltage is less than the voltage threshold
        return {
             'EOD': V < self.parameters['VEOD']
        }
