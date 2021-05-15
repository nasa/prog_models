# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from .. import prognostics_model

from math import asinh, log

# Constants of nature
R = 8.3144621;  # universal gas constant, J/K/mol
F = 96487;      # Faraday's constant, C/mol

def update_qmax(params):
    # note qMax = qn+qp
    return {
        'qMax': params['qMobile']/(params['xnMax']-params['xnMin'])
    }

def update_vols(params):
    # Volumes (total volume is 2*P.Vol), assume volume at each electrode is the
    # same and the surface/bulk split is the same for both electrodes
    return {
        'VolS': params['VolSFraction']*params['Vol'],
        'VolB': params['Vol']*(1.0-params['VolSFraction'])
    }

# set up charges (Li ions)
def update_qpmin(params):
    # min charge at pos electrode
    return {
        'qpMin': params['qMax']*params['xpMin'] 
    }

def update_qpmax(params):
    # max charge at pos electrode
    return {
        'qpMax': params['qMax']*params['xpMax'] 
    }

def update_qnmin(params):
    # min charge at negative electrode
    return {
        'qnMin': params['qMax']*params['xnMin'] 
    }

def update_qnmax(params):
    # max charge at negative electrode
    return {
        'qnMax': params['qMax']*params['xnMax'] 
    }

def update_qpSBmin(params):
    # min charge at surface and bulk pos electrode
    return {
        'qpSMin': params['qMax']*params['xpMin']*params['VolSFraction'],
        'qpBMin': params['qMax']*params['xpMin']*(1.0-params['VolSFraction']),
        'x0': {
            **params['x0'],
            'qpS': params['qMax']*params['xpMin']*params['VolSFraction'],
            'qpB': params['qMax']*params['xpMin']*(1.0-params['VolSFraction'])
        }
    }

def update_qpSBmax(params):
    # max charge at surface and pos electrode
    return {
        'qpSMax': params['qMax']*params['xpMax']*params['VolSFraction'],
        'qpBMax': params['qMax']*params['xpMax']*(1.0-params['VolSFraction'])
    }

def update_qnSBmin(params):
    # min charge at surface and bulk pos electrode
    return {
        'qnSMin': params['qMax']*params['xnMin']*params['VolSFraction'],
        'qnBMin': params['qMax']*params['xnMin']*(1.0-params['VolSFraction'])

    }

def update_qnSBmax(params):
    # max charge at surface and pos electrode
    return {
        'qnSMax': params['qMax']*params['xnMax']*params['VolSFraction'],
        'qnBMax': params['qMax']*params['xnMax']*(1.0-params['VolSFraction']),
        'x0': {
            **params['x0'],
            'qnS': params['qMax']*params['xnMax']*params['VolSFraction'],
            'qnB': params['qMax']*params['xnMax']*(1.0-params['VolSFraction'])
        }
    }

def update_qSBmax(params):
    # max charge at surface, bulk (pos and neg)
    return {
        'qSMax': params['qMax']*params['VolSFraction'],
        'qBMax': params['qMax']*(1.0-params['VolSFraction']),
    }

derived_callbacks = {
    'qMobile': [update_qmax],
    'VolSFraction': [update_vols, update_qpSBmin, update_qpSBmax, update_qSBmax],
    'Vol': [update_vols],
    'qMax': [update_qpmin, update_qpmax, update_qpSBmin, update_qpSBmax, update_qnmin, update_qnmax, update_qpSBmin, update_qpSBmax, update_qSBmax],
    'xpMin': [update_qpmin, update_qpSBmin],
    'xpMax': [update_qpmax, update_qpSBmax],
    'xnMin': [update_qmax, update_qnmin, update_qnSBmin],
    'xnMax': [update_qmax, update_qnmax, update_qnSBmax]
}


class BatteryElectroChem(prognostics_model.PrognosticsModel):
    """
    Prognostics model for a battery, represented by an electrochemical equations.

    This class implements an Electro chemistry model as described in the following paper:
    `M. Daigle and C. Kulkarni, "Electrochemistry-based Battery Modeling for Prognostics," Annual Conference of the Prognostics and Health Management Society 2013, pp. 249-261, New Orleans, LA, October 2013. http://www.phmsociety.org/node/1054/`

    The default model parameters included are for Li-ion batteries, specifically 18650-type cells. Experimental discharge curves for these cells can be downloaded from the `Prognostics Center of Excellence Data Repository https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/`.

    Events: (1)
        EOD: End of Discharge

    Inputs/Loading: (1)
        i: Current draw on the battery

    States: (8)
        tb, Vo, Vsn, Vsp, qnB, qnS, qpB, qpS

    Outputs/Measurements: (2)
        | t: Temperature of battery (°C) 
        | v: Voltage supplied by battery`

    Model Configuration Parameters:
        | process_noise : Process noise (applied at dx/next_state). 
                    Can be number (e.g., .2) applied to every state, a dictionary of values for each 
                    state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x) -> x
        | process_noise_dist : Optional, distribution for process noise (e.g., normal, uniform, triangular)
        | measurement_noise : Measurement noise (applied in output eqn)
                    Can be number (e.g., .2) applied to every output, a dictionary of values for each 
                    output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z
        | measurement_noise_dist : Optional, distribution for measurement noise (e.g., normal, uniform, triangular)
        | qMobile :
        | xnMax : Maximum mole fraction (neg electrode)
        | xnMin : Minimum mole fraction (neg electrode)
        | xpMax : Maximum mole fraction (pos electrode)
        | xpMin : Minimum mole fraction (pos electrode) - note xn + xp = 1
        | Ro : for Ohmic drop (current collector resistances plus electrolyte resistance plus solid phase resistances at anode and cathode)
        | alpha : anodic/cathodic electrochemical transfer coefficient
        | Sn : Surface area (- electrode) 
        | Sp : Surface area (+ electrode)
        | kn : lumped constant for BV (- electrode)
        | kp : lumped constant for BV (+ electrode)
        | Vol : total interior battery volume/2 (for computing concentrations)
        | VolSFraction : fraction of total volume occupied by surface volume
        | tDiffusion : diffusion time constant (increasing this causes decrease in diffusion rate)
        | to : for Ohmic voltage
        | tsn : for surface overpotential (neg)
        | tsp : for surface overpotential (pos)
        | U0p : Redlich-Kister parameter (+ electrode)
        | Ap : Redlich-Kister parameters (+ electrode)
        | U0n : Redlich-Kister parameter (- electrode)
        | An : Redlich-Kister parameters (- electrode)
        | VEOD : End of Discharge Voltage Threshold
        | x0 : Initial state
    """
    events = ['EOD']
    inputs = ['i']
    states = ['tb', 'Vo', 'Vsn', 'Vsp', 'qnB', 'qnS', 'qpB', 'qpS']
    outputs = ['t', 'v']

    default_parameters = {  # Set to defaults
        'qMobile': 7600,
        'xnMax': 0.6,
        'xnMin': 0,
        'xpMax': 1.0,
        'xpMin': 0.4,
        'Ro': 0.117215,
        
        # Li-ion parameters
        'alpha': 0.5,
        'Sn': 0.000437545,
        'Sp': 0.00030962,
        'kn': 2120.96,
        'kp': 248898,
        'Vol': 2e-5,
        'VolSFraction': 0.1,

        # time constants
        'tDiffusion': 7e6,
        'to': 6.08671,
        'tsn': 1001.38,
        'tsp': 46.4311,

        # Redlich-Kister parameters (+ electrode)
        'U0p': 4.03,
        'Ap': [-31593.7, 0.106747, 24606.4, -78561.9, 13317.9, 307387, 84916.1, -1.07469e+06, 2285.04, 990894, 283920, -161513, -469218],

        # Redlich-Kister parameters (- electrode)
        'U0n': 0.01,
        'An': [86.19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

        'x0': {
            'Vo': 0,
            'Vsn': 0,
            'Vsp': 0,
            'tb': 292.1  # in K, about 18.95 C
        },

        'process_noise': 1e-3,

        # End of discharge voltage threshold
        'VEOD': 3.0
    }

    def get_derived_callbacks(self):
        return derived_callbacks

    def initialize(self, u, z):
        return self.parameters['x0']

    def dx(self, x, u):
        params = self.parameters
        # Negative Surface
        CnBulk = x['qnB']/params['VolB']
        CnSurface = x['qnS']/params['VolS']
        xnS = x['qnS']/params['qSMax']

        qdotDiffusionBSn = (CnBulk-CnSurface)/params['tDiffusion']

        Jn = u['i']/params['Sn']
        Jn0 = params['kn']*((1-xnS)*xnS)**params['alpha']

        v_part = R*x['tb']/F/params['alpha']

        VsnNominal = v_part*asinh(Jn/(Jn0 + Jn0))
        Vsndot = (VsnNominal-x['Vsn'])/params['tsn']

        # Positive Surface
        CpBulk = x['qpB']/params['VolB']
        CpSurface = x['qpS']/params['VolS']
        xpS = x['qpS']/params['qSMax']
        
        qdotDiffusionBSp = (CpBulk-CpSurface)/params['tDiffusion']
        qpBdot = -qdotDiffusionBSp
        qpSdot = u['i'] + qdotDiffusionBSp

        Jp = u['i']/params['Sp']
        Jp0 = params['kp']*((1-xpS)*xpS)**params['alpha']

        VspNominal = v_part*asinh(Jp/(Jp0+Jp0))
        Vspdot = (VspNominal-x['Vsp'])/params['tsp']

        # Combined
        VoNominal = u['i']*params['Ro']
        Vodot = (VoNominal-x['Vo'])/params['to']

        return self.apply_process_noise({
            'tb': 0,
            'Vo': Vodot,
            'Vsn': Vsndot,
            'Vsp': Vspdot,
            'qnB': -qdotDiffusionBSn,
            'qnS': qdotDiffusionBSn - u['i'],
            'qpB': qpBdot,
            'qpS': qpSdot
        })
        
    def event_state(self, x):
        return {
            'EOD': (x['qnS'] + x['qnB'])/self.parameters['qnMax']
        }

    def output(self, x):
        params = self.parameters
        An = params['An']
        # Negative Surface
        xnS = x['qnS']/params['qSMax']
        xnS2 = xnS+xnS  # Note: in python x+x is more efficient than 2*x
        VenParts = [
            An[0] *(xnS2-1)/F,  # Ven0
            An[1] *((xnS2-1)**2  - ((xnS + xnS)*(1-xnS)))/F,  # Ven1
            An[2] *((xnS2-1)**3  - (4 *xnS*(1-xnS))*(xnS2-1))/F,  #Ven2
            An[3] *((xnS2-1)**4  - (6 *xnS*(1-xnS))*(xnS2-1)**2) /F,  #Ven3
            An[4] *((xnS2-1)**5  - (8 *xnS*(1-xnS))*(xnS2-1)**3) /F,  #Ven4
            An[5] *((xnS2-1)**6  - (10*xnS*(1-xnS))*(xnS2-1)**4) /F,  #Ven5
            An[6] *((xnS2-1)**7  - (12*xnS*(1-xnS))*(xnS2-1)**5) /F,  #Ven6
            An[7] *((xnS2-1)**8  - (14*xnS*(1-xnS))*(xnS2-1)**6) /F,  #Ven7
            An[8] *((xnS2-1)**9  - (16*xnS*(1-xnS))*(xnS2-1)**7) /F,  #Ven8
            An[9] *((xnS2-1)**10 - (18*xnS*(1-xnS))*(xnS2-1)**8) /F,  #Ven9
            An[10]*((xnS2-1)**11 - (20*xnS*(1-xnS))*(xnS2-1)**9) /F,  #Ven10
            An[11]*((xnS2-1)**12 - (22*xnS*(1-xnS))*(xnS2-1)**10)/F,  #Ven11
            An[12]*((xnS2-1)**13 - (24*xnS*(1-xnS))*(xnS2-1)**11)/F   #Ven12
        ]
        Ven = params['U0n'] + R*x['tb']/F*log((1-xnS)/xnS) + sum(VenParts)

        # Positive Surface
        Ap = params['Ap']
        xpS = x['qpS']/params['qSMax']
        xpS2 = xpS + xpS
        VepParts = [
            Ap[0] *(xpS2-1)/F,  #Vep0
            Ap[1] *((xpS2-1)**2  - (xpS2*(1-xpS)))/F,  #Vep1 
            Ap[2] *((xpS2-1)**3  - (4 *xpS*(1-xpS))/(xpS2-1)**(-1)) /F,  #Vep2
            Ap[3] *((xpS2-1)**4  - (6 *xpS*(1-xpS))/(xpS2-1)**(-2)) /F,  #Vep3
            Ap[4] *((xpS2-1)**5  - (8 *xpS*(1-xpS))/(xpS2-1)**(-3)) /F,  #Vep4
            Ap[5] *((xpS2-1)**6  - (10*xpS*(1-xpS))/(xpS2-1)**(-4)) /F,  #Vep5
            Ap[6] *((xpS2-1)**7  - (12*xpS*(1-xpS))/(xpS2-1)**(-5)) /F,  #Vep6
            Ap[7] *((xpS2-1)**8  - (14*xpS*(1-xpS))/(xpS2-1)**(-6)) /F,  #Vep7
            Ap[8] *((xpS2-1)**9  - (16*xpS*(1-xpS))/(xpS2-1)**(-7)) /F,  #Vep8
            Ap[9] *((xpS2-1)**10 - (18*xpS*(1-xpS))/(xpS2-1)**(-8)) /F,  #Vep9
            Ap[10]*((xpS2-1)**11 - (20*xpS*(1-xpS))/(xpS2-1)**(-9)) /F,  #Vep10
            Ap[11]*((xpS2-1)**12 - (22*xpS*(1-xpS))/(xpS2-1)**(-10))/F,  #Vep11
            Ap[12]*((xpS2-1)**13 - (24*xpS*(1-xpS))/(xpS2-1)**(-11))/F   #Vep12
        ]
        Vep = params['U0p'] + R*x['tb']/F*log((1-xpS)/xpS) + sum(VepParts)

        return self.apply_measurement_noise({
            't': x['tb'] - 273.15,
            'v': Vep - Ven - x['Vo'] - x['Vsn'] - x['Vsp']
        })

    def threshold_met(self, x):
        z = self.output(x)

        # Return true if voltage is less than the voltage threshold
        return {
             'EOD': z['v'] < self.parameters['VEOD']
        }
