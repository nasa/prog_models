from .. import prognostics_model

class PneumaticValve(prognostics_model.PrognosticsModel):
    """
    Prognostics model for a pneumatic valve.

    This class implements a Pneumatic Valve model as described in the following paper:
    `M. Daigle and K. Goebel, "A Model-based Prognostics Approach Applied to Pneumatic Valves," International Journal of Prognostics and Health Management, vol. 2, no. 2, August 2011. https://www.phmsociety.org/node/602`
    
    Events: (5)
        | Bottom Leak: Failure due to a leak at the bottom pneumatic port
        | Top Leak: Failure due to a leak at the top pneumatic port
        | Internal Leak: Failure due to an internal leak at the seal surrounding the piston
        | Spring Failure: Failure due to spring weakening with use
        | Friction Failure: Failure due to increase in friction along the piston with wear

    Inputs/Loading:
        | pL: Fluid pressure at the left side of the plug (Pa) TODO(CT): Confirm unit
        | pR: Fluid pressure at the right side of the plug (Pa) 
        | uBot: input pressure at the bottom pneumatic port (Pa) 
        | uTop: input pressure at the botton pneumatic port (Pa) 

    States:

    Outputs/Measurements:
    """
    events = [
        "Bottom Leak", # TODO(CT): CONFIRM
        "Top Leak", 
        "Internal Leak",
        "Spring Failure",
        "Friction Failure"
    ]
    inputs = [
        "pL",
        "pR",
        "uBot",
        "uTop"
    ]
    states = [
        "Aeb",
        "Aet",
        "Ai",
        "condition", # Consider removing
        "k",
        "mBot",
        "mTop",
        "r",
        "v",
        "wb",
        "wi",
        "wk",
        "wr",
        "wt",
        "x"
    ]
    outputs = [
        "Q",
        "iB", 
        "iT",
        "pB",
        "pT"
        "x"
    ]
    paramters = { # Set to defaults
        'process_noise': 0.1,

        # Environmental Parameters
        'R': 8.314, # Universal Gas Constant
        'g': 9.81, # Acceleration due to gravity (m^2/s)
        'pAtm': 101325, # Atmospheric pressure (Pa)

        # Valve Parameters
        'm': 50, # Plug mass (kg)
        'offsetX': 0.254, # Spring offset distance (m)
        'Ls': 0.0381, # Stroke Length (m)
        'Ap': 8.107319666e-4, # Surface area of piston for gas contact (m^2)
        'Vbot0': 8.107319666e-4, # Below piston "default" volume (m^3)
        'Vtop0': 8.107319666e-4, # Above piston "default" volume (m^3)

        # Flow Parameters
        'pSupply': 5272420.89227839, # Supply Pressure (Pa)
        'Av': 0.05067074791, # Surface area of plug end (m^2)

        # Gaseous nitrogen params
        'gn2_mass': 28.01e-3, # Molar mass of GN2 (kg/mol)
        'gn2_temp': 293, # temperature of GN2 (K)

        # Initial state
        'x0': {
            'x': 0,
            'v': 0,
        }
    }

    def initialize(self, u, z = None):
        return self.parameters['x0']
    
    def next_state(self, t, x, u, dt):
        xdot = x['v']
        pInTop = eq(u['uTop'],0)*self.parameters['pAtm'] + eq(u['uTop'],1)*self.parameters['pSupply']
        springForce = x['k']*(self.parameters['offsetX']+x['x'])
        friction = x['v']*x['r']
        fluidForce = (u['pL']-u['pR'])*self.parameters['Av']
        pInBot = eq(u['uBot'],0)*self.parameters['pAtm'] + eq(u['uBot'],1)*self.parameters['pSupply']
        volumeBot = self.parameters['Vbot0'] + self.parameters['Ap']*x['x']
        volumeTop = self.parameters['Vtop0'] + self.parameters['Ap']*(self.parameters['Ls']-x['x'])
        plugWeight = self.parameters['m']*self.parameters['g']
        kdot = -x['wk']*abs(x['v']*springForce)
        rdot = x['wr']*abs(x['v']*friction)
        Aidot = x['wi']*abs(x['v']*friction)
        pressureBot = x['mBot']*self.parameters['R']*self.parameters['gn2_temp']/self.parameters['gn2_mass']/volumeBot
        mBotDotn = PneumaticValve.gasFlow(pInBot,pressureBot,parameters.GN2,parameters.Cb,parameters.Ab)
        pressureTop = x['mTop']*self.parameters['R']*self.parameters['gn2_temp']/self.parameters['gn2_mass']/volumeTop
        leakBotToAtm = PneumaticValve.gasFlow(pressureBot,parameters.pAtm,parameters.GN2,1,Aeb)
        gasForceTop = pressureTop*self.parameters['Ap']
        gasForceBot = pressureBot*self.parameters['Ap']
        leakTopToAtm = PneumaticValve.gasFlow(pressureTop,parameters.pAtm,parameters.GN2,1,Aet)
        leakTopToBot = PneumaticValve.gasFlow(pressureTop,pressureBot,parameters.GN2,1,Ai)
        mBotdot = mBotDotn + leakTopToBot - leakBotToAtm
        mTopDotn = PneumaticValve.gasFlow(pInTop,pressureTop,self.parameters['GN2'],self.parameters['Ct'],self.parameters['At'])
        pistonForces = -fluidForce - plugWeight - friction - springForce + gasForceBot - gasForceTop
        mTopdot = mTopDotn - leakTopToBot - leakTopToAtm
        vdot = pistonForces/self.parameters['m']

        # Update discrete state (1 == pushed bottom/closed, 2 == moving, 3 == pushed top/open)
        condition = 1*((x==0 & pistonForces<0) | (x+xdot*dt<0)) \
            + 2*((x==0 & pistonForces>0) | (x+xdot*dt>=0 & x>0 & x<self.parameters['Ls'] & x+xdot*dt<=self.parameters['Ls']) | (x==self.parameters['Ls'] & pistonForces<0))\
            + 3*((x==self.parameters['Ls'] & pistonForces>0) | (x+xdot*dt>self.parameters['Ls']))

        # Compute new x, v based on condition
        pos = (condition==1)*0 + (condition==2)*(x['x']+xdot*dt) + (condition==3)*self.parameters['Ls']
        vel = (condition==1)*0 + (condition==2)*(x['v']+vdot*dt) + (condition==3)*0

        return {
            'Aeb': x['Aeb'] + x['wb'] * dt,
            'Aet': x['Aet'] + x['wt'] * dt,
            'Ai': x['Ai'] + Aidot * dt,
            'condition': x['condition'],
            'k': x['k'] + kdot * dt,
            'mBot': x['mBot'] + mBotdot * dt,
            'mTop': x['mTop'] + mTopdot * dt,
            'r': x['r'] + rdot * dt,
            'v': vel,
            'wb': x['wb'],
            'wi': x['wi'],
            'wk': x['wk'],
            'wr': x['wr'],
            'wt': x['wt'],
            'x': pos
        }
    
    def output(self, t, x):
        return {}

    def event_state(self, t, x):
        return {}

    def threshold_met(self, t, x):
        return {}