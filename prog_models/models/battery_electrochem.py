from .. import deriv_prog_model

import math

# Constants of nature
R = 8.3144621; # universal gas constant, J/K/mol
F = 96487;     # Faraday's constant, C/mol

class BatteryElectroChemParamDict(dict):
    __setting = False
    def __update_derived_params(self):
        self['qMax'] = self['qMobile']/(self['xnMax']-self['xnMin']) # note qMax = qn+qp
        # Volumes (total volume is 2*P.Vol), assume volume at each electrode is the
        # same and the surface/bulk split is the same for both electrodes
        self['VolS'] = self['VolSFraction']*self['Vol'] # surface volume
        self['VolB'] = self['Vol'] - self['VolS'] # bulk volume

        # set up charges (Li ions)
        self['qpMin'] = self['qMax']*self['xpMin'] # min charge at pos electrode
        self['qpMax'] = self['qMax']*self['xpMax'] # max charge at pos electrode
        self['qpSMin'] = self['qMax']*self['xpMin']*self['VolSFraction'] # min charge at surface, pos electrode
        self['qpBMin'] = self['qMax']*self['xpMin']*(self['Vol'] - self['VolS'])/self['Vol'] # min charge at bulk, pos electrode
        self['qpSMax'] = self['qMax']*self['xpMax']*self['VolS']/self['Vol'] # max charge at surface, pos electrode
        self['qpBMax'] = self['qMax']*self['xpMax']*self['VolB']/self['Vol'] # max charge at bulk, pos electrode
        self['qnMin'] = self['qMax']*self['xnMin'] # max charge at neg electrode
        self['qnMax'] = self['qMax']*self['xnMax'] # max charge at neg electrode
        self['qnSMax'] = self['qMax']*self['xnMax']*self['VolSFraction'] # max charge at surface, neg electrode
        self['qnBMax'] = self['qMax']*self['xnMax']*(1-self['VolSFraction']) # max charge at bulk, neg electrode
        self['qnSMin'] = self['qMax']*self['xnMin']*self['VolSFraction'] # min charge at surface, neg electrode
        self['qnBMin'] = self['qMax']*self['xnMin']*(1-self['VolSFraction']) # min charge at bulk, neg electrode
        self['qSMax'] = self['qMax']*self['VolSFraction'] # max charge at surface (pos and neg)
        self['qBMax'] = self['qMax']*(1-self['VolSFraction']) # max charge at bulk (pos and neg)
        self['x0']['qpS'] = self['qpSMin']
        self['x0']['qpB'] = self['qpBMin']
        self['x0']['qnS'] = self['qnSMax']
        self['x0']['qnB'] = self['qnBMax']

    def __init__(self, *args, **kwarg):
        super(BatteryElectroChemParamDict, self).__init__(*args, **kwarg)
        if not self.__setting:
            self.__setting = True
            self.__update_derived_params()
            # Todo(CT): Handle Error
            self.__setting = False
    
    def __setitem__(self, item, value):
        super(BatteryElectroChemParamDict, self).__setitem__(item, value)
        if not self.__setting:
            self.__setting = True
            self.__update_derived_params()
            # Todo(CT): Handle Error
            self.__setting = False

class BatteryElectroChem(deriv_prog_model.DerivProgModel):
    """
    Prognostics model for a battery, represented by an electrochemical equations
    """
    events = [
        'EOD' # End of Discharge
    ]
    
    inputs = [
        'i' # Current
    ]

    states = [
        'tb',
        'Vo',
        'Vsn',
        'Vsp',
        'qnB',
        'qnS',
        'qpB',
        'qpS'
    ]

    outputs = [
        't', # Battery temperature (°C)
        'v'  # Battery Voltage
    ]

    parameters = BatteryElectroChemParamDict({ # Set to defaults
        'qMobile': 7600,
        'xnMax': 0.6,   # maximum mole fraction (neg electrode)
        'xnMin': 0,     # minimum mole fraction (neg electrode)
        'xpMax': 1.0,   # maximum mole fraction (pos electrode)
        'xpMin': 0.4,   # minimum mole fraction (pos electrode) -> note xn+xp=1
        'Ro': 0.117215, # for Ohmic drop (current collector resistances plus electrolyte resistance plus solid phase resistances at anode and cathode)
        
        # Li-ion parameters
        'alpha': 0.5,   # anodic/cathodic electrochemical transfer coefficient
        'Sn': 0.000437545, # surface area (- electrode)
        'Sp': 0.00030962, # surface area (+ electrode)
        'kn': 2120.96,  # lumped constant for BV (- electrode)
        'kp': 248898,   # lumped constant for BV (+ electrode)
        'Vol': 2e-5,    # total interior battery volume/2 (for computing concentrations)
        'VolSFraction': 0.1, # fraction of total volume occupied by surface volume

        # time constants
        'tDiffusion': 7e6, # diffusion time constant (increasing this causes decrease in diffusion rate)
        'to': 6.08671,  # for Ohmic voltage
        'tsn': 1001.38, # for surface overpotential (neg)
        'tsp': 46.4311, # for surface overpotential (pos)

        # Redlich-Kister parameters (positive electrode)
        'U0p': 4.03,
        'Ap': [-31593.7, 0.106747, 24606.4, -78561.9, 13317.9, 307387, 84916.1, -1.07469e+06, 2285.04, 990894, 283920, -161513, -469218],

        # Redlich-Kister parameters (negative electrode)
        'U0n': 0.01,
        'An': [86.19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

        'x0': {
            'Vo': 0,
            'Vsn': 0,
            'Vsp': 0,
            'tb': 292.1 # in K, about 18.95 C
        },

        # End of discharge voltage threshold
        'VEOD': 3.0
    })

    def initialize(self, u, z):
        return self.parameters['x0']

    def dx(self, t, x, u): 
        # Negative Surface
        CnBulk = x['qnB']/self.parameters['VolB']
        CnSurface = x['qnS']/self.parameters['VolS']
        xnS = x['qnS']/self.parameters['qSMax']

        qdotDiffusionBSn = (CnBulk-CnSurface)/self.parameters['tDiffusion']
        qnBdot = -qdotDiffusionBSn
        qnSdot = qdotDiffusionBSn - u['i']

        Jn = u['i']/self.parameters['Sn']
        Jn0 = self.parameters['kn']*((1-xnS)*xnS)**self.parameters['alpha']

        VsnNominal = R*x['tb']/F/self.parameters['alpha']*math.asinh(Jn/(2*Jn0))
        Vsndot = (VsnNominal-x['Vsn'])/self.parameters['tsn']

        # Positive Surface
        CpBulk = x['qpB']/self.parameters['VolB']
        CpSurface = x['qpS']/self.parameters['VolS']
        xpS = x['qpS']/self.parameters['qSMax']
        
        qdotDiffusionBSp = (CpBulk-CpSurface)/self.parameters['tDiffusion']
        qpBdot = -qdotDiffusionBSp
        qpSdot = u['i'] + qdotDiffusionBSp

        Jp = u['i']/self.parameters['Sp']
        Jp0 = self.parameters['kp']*((1-xpS)*xpS)**self.parameters['alpha']

        VspNominal = R*x['tb']/F/self.parameters['alpha']*math.asinh(Jp/(2*Jp0))
        Vspdot = (VspNominal-x['Vsp'])/self.parameters['tsp']

        # Combined
        VoNominal = u['i']*self.parameters['Ro']
        Vodot = (VoNominal-x['Vo'])/self.parameters['to']

        return {
            'tb': 0,
            'Vo': Vodot,
            'Vsn': Vsndot,
            'Vsp': Vspdot,
            'qnB': qnBdot,
            'qnS': qnSdot,
            'qpB': qpBdot,
            'qpS': qpSdot
        }
        
    def event_state(self, t, x):
        return {
            'EOD': (x['qnS'] + x['qnB'])/self.parameters['qnMax']
        }

    def output(self, t, x):
        # Negative Surface
        xnS = x['qnS']/self.parameters['qSMax']
        VenParts = [
            self.parameters['An'][0] *(2*xnS-1)/F, # Ven0
            self.parameters['An'][1] *((2*xnS-1)**2  - (2 *xnS*(1-xnS)))/F, # Ven1
            self.parameters['An'][2] *((2*xnS-1)**3  - (4 *xnS*(1-xnS))/(2*xnS-1)**(-1)) /F, #Ven2
            self.parameters['An'][3] *((2*xnS-1)**4  - (6 *xnS*(1-xnS))/(2*xnS-1)**(-2)) /F, #Ven3
            self.parameters['An'][4] *((2*xnS-1)**5  - (8 *xnS*(1-xnS))/(2*xnS-1)**(-3)) /F, #Ven4
            self.parameters['An'][5] *((2*xnS-1)**6  - (10*xnS*(1-xnS))/(2*xnS-1)**(-4)) /F, #Ven5
            self.parameters['An'][6] *((2*xnS-1)**7  - (12*xnS*(1-xnS))/(2*xnS-1)**(-5)) /F, #Ven6
            self.parameters['An'][7] *((2*xnS-1)**8  - (14*xnS*(1-xnS))/(2*xnS-1)**(-6)) /F, #Ven7
            self.parameters['An'][8] *((2*xnS-1)**9  - (16*xnS*(1-xnS))/(2*xnS-1)**(-7)) /F, #Ven8
            self.parameters['An'][9] *((2*xnS-1)**10 - (18*xnS*(1-xnS))/(2*xnS-1)**(-8)) /F, #Ven9
            self.parameters['An'][10]*((2*xnS-1)**11 - (20*xnS*(1-xnS))/(2*xnS-1)**(-9)) /F, #Ven10
            self.parameters['An'][11]*((2*xnS-1)**12 - (22*xnS*(1-xnS))/(2*xnS-1)**(-10))/F, #Ven11
            self.parameters['An'][12]*((2*xnS-1)**13 - (24*xnS*(1-xnS))/(2*xnS-1)**(-11))/F  #Ven12
        ]
        Ven = self.parameters['U0n'] + R*x['tb']/F*math.log((1-xnS)/xnS) + sum(VenParts)

        # Positive Surface
        xpS = x['qpS']/self.parameters['qSMax']
        VepParts = [
            self.parameters['Ap'][0] *(2*xpS-1)/F, #Vep0
            self.parameters['Ap'][1] *((2*xpS-1)**2  - (2 *xpS*(1-xpS)))/F, #Vep1
            self.parameters['Ap'][2] *((2*xpS-1)**3  - (4 *xpS*(1-xpS))/(2*xpS-1)**(-1)) /F, #Vep2
            self.parameters['Ap'][3] *((2*xpS-1)**4  - (6 *xpS*(1-xpS))/(2*xpS-1)**(-2)) /F, #Vep3
            self.parameters['Ap'][4] *((2*xpS-1)**5  - (8 *xpS*(1-xpS))/(2*xpS-1)**(-3)) /F, #Vep4
            self.parameters['Ap'][5] *((2*xpS-1)**6  - (10*xpS*(1-xpS))/(2*xpS-1)**(-4)) /F, #Vep5
            self.parameters['Ap'][6] *((2*xpS-1)**7  - (12*xpS*(1-xpS))/(2*xpS-1)**(-5)) /F, #Vep6
            self.parameters['Ap'][7] *((2*xpS-1)**8  - (14*xpS*(1-xpS))/(2*xpS-1)**(-6)) /F, #Vep7
            self.parameters['Ap'][8] *((2*xpS-1)**9  - (16*xpS*(1-xpS))/(2*xpS-1)**(-7)) /F, #Vep8
            self.parameters['Ap'][9] *((2*xpS-1)**10 - (18*xpS*(1-xpS))/(2*xpS-1)**(-8)) /F, #Vep9
            self.parameters['Ap'][10]*((2*xpS-1)**11 - (20*xpS*(1-xpS))/(2*xpS-1)**(-9)) /F, #Vep10
            self.parameters['Ap'][11]*((2*xpS-1)**12 - (22*xpS*(1-xpS))/(2*xpS-1)**(-10))/F, #Vep11
            self.parameters['Ap'][12]*((2*xpS-1)**13 - (24*xpS*(1-xpS))/(2*xpS-1)**(-11))/F  #Vep12
        ]
        Vep = self.parameters['U0p'] + R*x['tb']/F*math.log((1-xpS)/xpS) + sum(VepParts)

        return {
            't': x['tb'] - 273.15,
            'v': Vep - Ven - x['Vo'] - x['Vsn'] - x['Vsp']
        }

    def threshold_met(self, t, x):
        z = self.output(t, x)

        # Return true if voltage is less than the voltage threshold
        return {
             'EOD': z['v'] < self.parameters['VEOD']
        }
