# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np

from . import PrognosticsModel


class EnsembleModel(PrognosticsModel):
    default_parameters = {
        'aggregation_method': np.mean,
    }
    def __init__(self, models, **kwargs):
        inputs = set()
        states = set()
        outputs = set()
        events = set()
        for m in models:
            inputs |= set(m.inputs)
            states |= set(m.states)
            outputs |= set(m.outputs)
            events |= set(m.events)
        self.inputs = list(inputs)
        self.states = list(states)
        self.outputs = list(outputs)
        self.events = list(events)

        super().__init__(**kwargs)
        self.parameters['models'] = models

    def initialize(self, u, z=None):
        xs = [m.initialize(m.InputContainer(u), m.OutputContainer(z) if z is not None else None) for m in self.parameters['models']]
        x0 = {}
        for x in xs:
            for key in x.keys():
                if key in x0:
                    x0[key].append(x[key])
                else:
                    x0[key] = [x[key]]
        for key in x0.keys():
            x0[key] = self.parameters['aggregation_method'](x0[key]) 
        return self.StateContainer(x0)


    def next_state(self, x, u, dt):
        xs = [m.next_state(m.StateContainer(x), m.InputContainer(u), dt) for m in self.parameters['models']]
        xs_final = {}
        for x in xs:
            for key in x.keys():
                if key in xs_final:
                    xs_final[key].append(x[key])
                else:
                    xs_final[key] = [x[key]]
        for key in xs_final.keys():
            xs_final[key] = self.parameters['aggregation_method'](xs_final[key])

        return self.StateContainer(xs_final)
        
    def output(self, x):
        zs = [m.output(m.StateContainer(x)) for m in self.parameters['models']]
        zs_final = {}
        for z in zs:
            for key in z.keys():
                if key in zs_final:
                    zs_final[key].append(z[key])
                else:
                    zs_final[key] = [z[key]]
        for key in zs_final.keys():
            zs_final[key] = self.parameters['aggregation_method'](zs_final[key])

        return self.OutputContainer(zs_final)
        
    def event_state(self, x):
        es = [m.event_state(m.StateContainer(x)) for m in self.parameters['models']]
        es_final = {}
        for es_i in es:
            for key in es_i.keys():
                if key in es_final:
                    es_final[key].append(es_i[key])
                else:
                    es_final[key] = [es_i[key]]
        for key in es_final.keys():
            es_final[key] = self.parameters['aggregation_method'](es_final[key])

        return es_final
