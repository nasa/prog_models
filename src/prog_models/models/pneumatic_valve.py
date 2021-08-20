# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from .. import prognostics_model
from math import sqrt, copysign, inf
from copy import deepcopy


class PneumaticValveBase(prognostics_model.PrognosticsModel):
    """
    Prognostics model for a pneumatic valve.

    This class implements a Pneumatic Valve model as described in the following paper:
    `M. Daigle and K. Goebel, "A Model-based Prognostics Approach Applied to Pneumatic Valves," International Journal of Prognostics and Health Management, vol. 2, no. 2, August 2011. https://papers.phmsociety.org/index.php/ijphm/article/view/1359`
    
    Events: (5)
        | Bottom Leak: Failure due to a leak at the bottom pneumatic port
        | Top Leak: Failure due to a leak at the top pneumatic port
        | Internal Leak: Failure due to an internal leak at the seal surrounding the piston
        | Spring Failure: Failure due to spring weakening with use
        | Friction Failure: Failure due to increase in friction along the piston with wear

    Inputs/Loading:
        | pL: Fluid pressure at the left side of the plug (Pa)
        | pR: Fluid pressure at the right side of the plug (Pa) 
        | uBot: input pressure at the bottom pneumatic port (Pa) 
        | uTop: input pressure at the botton pneumatic port (Pa) 

    States:
        | Aeb: Area of the leak at the bottom pneumatic port
        | Aet: Area of the leak at the top pneumatic port
        | Ai: Area of the internal leak
        | k: Spring Coefficient
        | mBot: Mass on bottom of piston (kg)
        | mTop: Mass on top of the piston (kg)
        | r: Friction Coefficient
        | v: Velocity of the piston (m/s)
        | x: Poisition of the piston (m)
        | pDiff: Difference in pressure between the left and the right

    Outputs/Measurements:
        | Q: Flowrate 
        | iB: Is the piston at the bottom (bool)
        | iT: Is the piston at the top (bool)
        | pB: Pressure at the bottom (Pa)
        | pT: Pressure at the top (Pa)
        | x: Position of piston (m)

    Model Configuration Parameters:
        | process_noise : Process noise (applied at dx/next_state). 
                    Can be number (e.g., .2) applied to every state, a dictionary of values for each 
                    state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x) -> x
        | process_noise_dist : Optional, distribution for process noise (e.g., normal, uniform, triangular)
        | measurement_noise : Measurement noise (applied in output eqn)
                    Can be number (e.g., .2) applied to every output, a dictionary of values for each 
                    output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z
        | measurement_noise_dist : Optional, distribution for measurement noise (e.g., normal, uniform, triangular)
        | g : Acceleration due to gravity (m/s^2)
        | pAtm : Atmospheric pressure (Pa)
        | m : Plug mass (kg)
        | offsetX : Spring offset distance (m)
        | Ls : Stroke Length (m)
        | Ap : Surface area of piston for gas contact (m^2)
        | Vbot0 : Below piston "default" volume (m^3)
        | Vtop0 : Above piston "default" volume (m^3)
        | indicatorTol : tolerance bound for open/close indicators
        | pSupply : Supply Pressure (Pa)
        | Av : Surface area of plug end (m^2)
        | Cv : flow coefficient assuming Cv of 1300 GPM
        | rhoL : density of LH2 in kg/m^3
        | gas_mass : Molar mass of used gas (kg/mol)
        | gas_temp : temperature of used gas (K)
        | gas_gamma :
        | gas_z :
        | gas_R :
        | At :
        | Ct :
        | Ab :
        | Cb :
        | AbMax : Max limit for state Aeb
        | AtMax : Max limit for state Aet
        | AiMax : Max limit for state Ai
        | kMin : Min limit for state k
        | rMax : Max limit for state r
        | x0 : Initial state
        | wb: Wear parameter for bottom leak
        | wi: Wear parameter for internal leak
        | wt: Wear parameter for top leak
        | wk: Wear parameter for spring
        | wr: Wear parameter for friction
    """
    events = ["Bottom Leak", "Top Leak", "Internal Leak", "Spring Failure", "Friction Failure"]
    inputs = ["pL", "pR", "uBot", "uTop"]
    states = [
        "Aeb",
        "Aet",
        "Ai",
        "k",
        "mBot",
        "mTop",
        "r",
        "v",
        "x",
        "pDiff"  # pL-pR
    ]
    outputs = ["Q", "iB", "iT", "pB", "pT", "x"]
    default_parameters = {  # Set to defaults
        # Environmental Parameters
        'R': 8.314,  # Universal Gas Constant
        'g': 9.81, 
        'pAtm': 101325,

        # Valve Parameters
        'm': 50,
        'offsetX': 0.254,
        'Ls': 0.0381,
        'Ap': 8.1073196655599634694e-3,
        'Vbot0': 8.107319665559963e-4,
        'Vtop0': 8.107319665559963e-4,
        'indicatorTol': 1e-3,

        # Flow Parameters
        'pSupply': 5.272420892278394995e6,
        'Av': 0.050670747909749769389,
        'Cv': 0.4358892767469993814,
        'rhoL': 70.99,

        # Supply gas params (Note: Default is nitrogen)
        'gas_mass': 28.01e-3,
        'gas_temp': 293,
        'gas_gamma': 1.4, 
        'gas_z': 1,
        'gas_R': 296.8225633702249454,

        # Orifice params
        'At': 1e-5,
        'Ct': 0.62,
        'Ab': 1e-5,
        'Cb': 0.62,

        # Limits
        "AbMax": 4e-5,
        "AtMax": 4e-5,
        "AiMax": 1.7e-6,
        "kMin": 3.95e4,
        "rMax": 4e6,

        # Initial state
        'x0': {
            'x': 0,
            'v': 0,
            'mTop': 0.067876043046174843,
            'mBot': 9.4455962535380932526e-4,
            'Aeb': 1e-5,
            'Ai': 0,
            'Aet': 1e-5,
            'k': 48000,
            'r': 6000
        },

        # Wear Rates
        'wb': 0,
        'wi': 0,
        'wk': 0,
        'wr': 0,
        'wt': 0
    }

    state_limits = {
        'Aeb': (0, inf),
        'Aet': (0, inf),
        'Ai': (0, inf),
        'k': (0, inf),
        'mBot': (0, inf),
        'mTop': (0, inf),
        'r': (0, inf)
    }

    def initialize(self, u, z = None):
        x0 = self.parameters['x0']
        x0['pDiff'] = u['pL'] - u['pR']
        return x0

    def gas_flow(self, pIn, pOut, C, A):
        k = self.parameters['gas_gamma']
        T = self.parameters['gas_temp']
        Z = self.parameters['gas_z']
        R = self.parameters['gas_R']
        threshold = ((k+1)/2)**(k/(k-1))

        if pIn/pOut>=threshold:
            return C*A*pIn*sqrt(k/Z/R/T*(2/(k+1))**((k+1)/(k-1)))
        if pIn>=pOut:
            return C*A*pIn*sqrt(2/Z/R/T*k/(k-1)*abs((pOut/pIn)**(2/k)-(pOut/pIn)**((k+1)/k)))
        if pOut/pIn>=threshold:
            return -C*A*pOut*sqrt(k/Z/R/T*(2/(k+1))**((k+1)/(k-1)))
        # pOut>pIn but pOut/pIn < threshold - only remaining possibility 
        return -C*A*pOut*sqrt(2/Z/R/T*k/(k-1)*abs((pIn/pOut)**(2/k)-(pIn/pOut)**((k+1)/k)))
    
    def next_state(self, x, u, dt):
        params = self.parameters # optimization
        pInTop = params['pSupply'] if u['uTop'] else params['pAtm'] 
        springForce = x['k']*(params['offsetX']+x['x'])
        friction = x['v']*x['r']
        fluidForce = (u['pL']-u['pR'])*params['Av']
        pInBot = params['pSupply'] if u['uBot'] else params['pAtm'] 
        volumeBot = params['Vbot0'] + params['Ap']*x['x']
        volumeTop = params['Vtop0'] + params['Ap']*(params['Ls']-x['x'])
        plugWeight = params['m']*params['g']
        kdot = -params['wk']*abs(x['v']*springForce)
        rdot = params['wr']*abs(x['v']*friction)
        Aidot = params['wi']*abs(x['v']*friction)
        pressureBot = x['mBot']*params['R']*params['gas_temp']/params['gas_mass']/volumeBot
        mBotDotn = self.gas_flow(pInBot,pressureBot,params['Cb'],params['Ab'])
        pressureTop = x['mTop']*params['R']*params['gas_temp']/params['gas_mass']/volumeTop
        leakBotToAtm = self.gas_flow(pressureBot,params['pAtm'],1,x['Aeb'])
        gasForceTop = pressureTop*params['Ap']
        gasForceBot = pressureBot*params['Ap']
        leakTopToAtm = self.gas_flow(pressureTop,params['pAtm'],1,x['Aet'])
        leakTopToBot = self.gas_flow(pressureTop,pressureBot,1,x['Ai'])
        mBotdot = mBotDotn + leakTopToBot - leakBotToAtm
        mTopDotn = self.gas_flow(pInTop,pressureTop,params['Ct'],params['At'])
        pistonForces = -fluidForce - plugWeight - friction - springForce + gasForceBot - gasForceTop
        mTopdot = mTopDotn - leakTopToBot - leakTopToAtm
        vdot = pistonForces/params['m']

        new_x = x['x']+x['v']*dt
        if (x['x']==0 and pistonForces<0) or (new_x<0):
            vel = 0
            pos = 0
        elif (x['x']==params['Ls'] and pistonForces>0) or (new_x>params['Ls']):
            vel = 0
            pos = params['Ls']
        else:
            # moving
            vel = x['v'] + vdot*dt
            pos = new_x

        return {
            'x': pos,
            'v': vel,
            'mTop': x['mTop'] + mTopdot * dt,
            'mBot': x['mBot'] + mBotdot * dt,
            'Aeb': x['Aeb'] + params['wb'] * dt,
            'Aet': x['Aet'] + params['wt'] * dt,
            'Ai': x['Ai'] + Aidot * dt,
            'k': x['k'] + kdot * dt,
            'r': x['r'] + rdot * dt,
            'pDiff': u['pL'] - u['pR']
        }
    
    def output(self, x):
        params = self.parameters  # Optimization
        indicatorTopm = (x['x'] >= params['Ls']-params['indicatorTol'])
        indicatorBotm = (x['x'] <= params['indicatorTol'])
        maxFlow = params['Cv']*params['Av']*copysign(sqrt(2/params['rhoL']*abs(x['pDiff'])),x['pDiff'])
        volumeBot = params['Vbot0'] + params['Ap']*x['x']
        volumeTop = params['Vtop0'] + params['Ap']*(params['Ls']-x['x'])
        trueFlow = maxFlow * max(0,x['x'])/params['Ls']
        pressureTop = x['mTop']*params['R']*params['gas_temp']/params['gas_mass']/volumeTop
        pressureBot = x['mBot']*params['R']*params['gas_temp']/params['gas_mass']/volumeBot

        return {
            'Q': trueFlow,
            'iB': indicatorBotm,
            'iT': indicatorTopm,
            'pB': 1e-6 * pressureBot,
            'pT': 1e-6 * pressureTop,
            'x': x['x']
        }

    def event_state(self, x):
        params = self.parameters
        return {
            "Bottom Leak": (params['AbMax'] - x['Aeb'])/(params['AbMax'] - params['x0']['Aeb']), 
            "Top Leak": (params['AtMax'] - x['Aet'])/(params['AtMax'] - params['x0']['Aet']), 
            "Internal Leak": (params['AiMax'] - x['Ai'])/(params['AiMax'] - params['x0']['Ai']),
            "Spring Failure": (x['k'] - params['kMin'])/(params['x0']['k'] - params['kMin']),
            "Friction Failure": (params['rMax'] - x['r'])/(params['rMax'] - params['x0']['r'])
        }

    def threshold_met(self, x):
        params = self.parameters
        return {
            "Bottom Leak": x['Aeb'] > params['AbMax'], 
            "Top Leak": x['Aet'] > params['AtMax'], 
            "Internal Leak": x['Ai'] > params['AiMax'],
            "Spring Failure": x['k'] < params['kMin'],
            "Friction Failure": x['r'] > params['rMax']
        }


class PneumaticValveWithWear(PneumaticValveBase):
    """
    Prognostics model for a pneumatic valve with wear parameters as part of the model state. This is identical to PneumaticValveBase, only PneumaticValveBase has the wear params as parameters instead of states

    This class implements a Pneumatic Valve model as described in the following paper:
    `M. Daigle and K. Goebel, "A Model-based Prognostics Approach Applied to Pneumatic Valves," International Journal of Prognostics and Health Management, vol. 2, no. 2, August 2011. https://www.phmsociety.org/node/602`

    Events (4) 
        See PneumaticValveBase

    Inputs/Loading: (5)
        See PneumaticValveBase
    
    States: (12)
        States from PneumaticValveBase +  wb, wi, wk, wr, wt

    Outputs/Measurements: (5)
        See PneumaticValveBase

    Model Configuration Parameters:
        See PneumaticValveBase
    """
    inputs = PneumaticValveBase.inputs
    outputs = PneumaticValveBase.outputs
    states = PneumaticValveBase.states + ['wb', 'wi', 'wk', 'wr', 'wt']
    events = PneumaticValveBase.events

    default_parameters = deepcopy(PneumaticValveBase.default_parameters)
    default_parameters['x0'].update(
        {'wb': 0,
        'wi': 0,
        'wk': 0,
        'wr': 0,
        'wt': 0})

    state_limits = deepcopy(PneumaticValveBase.state_limits)

    def next_state(self, x, u, dt):
        self.parameters['wb'] = x['wb']
        self.parameters['wi'] = x['wi']
        self.parameters['wk'] = x['wk']
        self.parameters['wr'] = x['wr']
        self.parameters['wt'] = x['wt']
        next_x = PneumaticValveBase.next_state(self, x, u, dt)
        next_x.update({
            'wb': x['wb'],
            'wi': x['wi'],
            'wk': x['wk'],
            'wr': x['wr'],
            'wt': x['wt']
        })
        return next_x

PneumaticValve = PneumaticValveWithWear
