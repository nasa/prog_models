from .. import PrognosticsModel

from statistics import mean
from functools import reduce

class BatterySeries(PrognosticsModel):
    inputs = ['i']
    outputs = ['t', 'v']

    def __init__(self, batteries = []):
        self.batteries = batteries
        self.states = []
        for i, batt in zip(range(len(batteries)), batteries):
            self.states.extend([
                state + str(i) for state in batt.states
            ])
        
        if batteries:
            self.events = reduce(set.intersection, (set(batt.events) for batt in batteries)) # Events present in all batteries
        else: 
            self.events = set()

    def __str__(self):
        desc = "{} Prognostics Model (Events: {})".format(type(self).__name__, self.events)
        desc += "\n\t{} Batteries in Series".format(len(self.batteries))
        for i, batt in zip(range(len(self.batteries)), self.batteries):
            desc += "\n\tBatt {}. {}".format(i, type(batt).__name__)
        return desc
        
    def initialize(self, u, z):
        x0 = {}
        for i, batt in zip(range(len(self.batteries)), self.batteries):
            x0_i = batt.initialize(u, z)
            x0.update({
                key + str(i) : x0_i[key] for key in batt.states
            })
        return x0

    def add_battery(self, battery):
        n_batts = len(self.batteries)
        self.batteries.append(battery)
        self.states.append([state + str(n_batts) for state in battery.states])

    def next_state(self, x, u, dt):
        new_x = {}
        for i, batt in zip(range(len(self.batteries)), self.batteries):
            x_i = {
                key : x[key + str(i)] for key in batt.states
            }
            x_i = batt.next_state(x_i, u, dt)
            new_x.update({
                key + str(i) : x_i[key] for key in batt.states
            })

        return new_x

    def output(self, x):
        outputs = []
        for i, batt in zip(range(len(self.batteries)), self.batteries):
            x_i = {
                key : x[key + str(i)] for key in batt.states
            }
            outputs.append(batt.output(x_i))
        return {
            'v': sum([output['v'] for output in outputs]),
            't': mean([output['t'] for output in outputs])
        }

    def event_state(self, x):
        event_states = []
        for i, batt in zip(range(len(self.batteries)), self.batteries):
            x_i = {
                key : x[key + str(i)] for key in batt.states
            }
            event_states.append(batt.event_state(x_i))
        
        result_event_state = {}
        for key in self.events:
            result_event_state[key] = min([item[key] for item in event_states])
        return result_event_state
        # TODO(CT): Consider- should this be min or something else?
        
    def threshold_met(self, x):
        t_met = []
        for i, batt in zip(range(len(self.batteries)), self.batteries):
            x_i = {
                key : x[key + str(i)] for key in batt.states
            }
            t_met.append(batt.threshold_met(x_i))

        result_t_met = {}
        for key in self.events:
            result_t_met[key] = any([item[key] for item in t_met])

        return result_t_met


# class BatteryPack(PrognosticsModel):
#     """
#     Prognostics model for a pack of batteries 
#     """

#     events = ['EOD']
#     inputs = ['i']
#     states = []
#     outputs = ['t', 'v']

#     def __init__(self, serieses = []):
#         self.battery_serieses = serieses

#     def addSeries(self, series : BatterySeries):
#         self.battery_serieses.append(series)

#     def output(self, x):
#         outputs = [series.output(x) for series in self.battery_serieses]
#
# This one is a little more complicated - a model would have to include flow between batteries. I = (v1-V2)/(R1+R2)
