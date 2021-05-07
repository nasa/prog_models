from . import BatteryElectroChem as Battery

class BatteryElectroChemEOL(Battery):
    states = Battery.states + ['qMax', 'Ro', 'D']
    events = Battery.events + ['InsufficientCapacity']

    default_parameters = Battery.default_parameters
    default_parameters['x0']['qMax'] = 7600
    default_parameters['x0']['Ro'] = 0.117215
    default_parameters['x0']['D'] = 7e6
    default_parameters['wq'] = -1
    default_parameters['wr'] = 1e-2
    default_parameters['wd'] = 1e-2 
    default_parameters['qMaxThreshold'] = 3800

    def dx(self, x, u):
        params = self.parameters

        # Set parameters from EOD model
        self.parameters['qMobile'] = x['qMax']* (params['xnMax'] - params['xnMin'])
        # Make sure it's actually resetting qMax
        # Consider ability to register callback for derived params
        self.parameters['Ro'] = x['Ro']
        self.parameters['tDiffusion'] = x['D']
        
        # Calculate EOD Model
        x_dot = Battery.dx(self, x, u)

        # update EOL states
        x_dot['qMax'] = params['wq'] * abs(u['i'])
        x_dot['Ro'] = params['wr'] * abs(u['i'])
        x_dot['D'] = params['wd'] * abs(u['i'])

        return x_dot

    def event_state(self, x):
        e_state = Battery.event_state(self, x)
        e_state['InsufficientCapacity'] = max(min((x['qMax']-self.parameters['qMaxThreshold'])/(self.parameters['x0']['qMax']-self.parameters['qMaxThreshold']), 1.0), 0.0)
        return e_state

    def threshold_met(self, x):
        t_met = Battery.threshold_met(self, x)
        t_met['InsufficientCapacity'] = x['qMax'] < self.parameters['qMaxThreshold']
        return t_met