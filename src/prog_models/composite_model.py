# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections.abc import Iterable

from . import PrognosticsModel

DIVIDER = '.'


class CompositeModel(PrognosticsModel):

    default_parameters = {
        'connections': [],
    }

    def __init__(self, models, connections = [], **kwargs):

        # General Input Validation
        if not isinstance(models, Iterable):
            raise ValueError('The models argument must be a list')
        if not isinstance(connections, Iterable):
            raise ValueError('The connections argument must be a list')

        # Initialize
        self.inputs = set()
        self.states = set()
        self.outputs = set()
        self.events = set()
        self.model_names = set()
        duplicate_names = {}
        kwargs['models'] = []

        # Handle models
        for m in models:
            if isinstance(m, Iterable):
                if len(m) != 2: 
                    raise ValueError('Each model tuple must be of the form (name, model)')
                if not isinstance(m[0], str):
                    raise ValueError('The first element of each model tuple must be a string')
                if not isinstance(m[1], PrognosticsModel):
                    raise ValueError('The second element of each model tuple must be a PrognosticsModel')
                if m[0] in self.model_names:
                    duplicate_names[m[0]] = duplicate_names.get(m[0], 1) + 1
                    m = (m[0] + '_' + str(duplicate_names[m[0]]), m[1])
                self.model_names.add(m[0])
                kwargs['models'].append(m)
            else:
                if not isinstance(m, PrognosticsModel):
                    raise ValueError('Each model must be a PrognosticsModel')
                m = (m.__class__.__name__, m)
                if m[0] in self.model_names:
                    duplicate_names[m[0]] = duplicate_names.get(m[0], 1) + 1
                    m = (m[0] + '_' + str(duplicate_names[m[0]]), m[1])
                self.model_names.add(m[0])
                kwargs['models'].append(m)

        for (name, m) in kwargs['models']:
            self.inputs |= set([name + DIVIDER + u for u in m.inputs])
            self.states |= set([name + DIVIDER + x for x in m.states])
            self.outputs |= set([name + DIVIDER + z for z in m.outputs])
            self.events |= set([name + DIVIDER + e for e in m.events])
        
        # Handle outputs
        if 'outputs' in kwargs:
            if not set(kwargs['outputs']).issubset(self.outputs):
                raise ValueError('The outputs of the composite model must be a subset of the outputs of the models')
            self.outputs = kwargs['outputs']
        
        # Handle Connections
        kwargs['connections'] = []
        self.__to_input_connections = {m_name : [] for m_name in self.model_names}
        self.__to_state_connections = {m_name : [] for m_name in self.model_names}

        for in_key, out_key in connections:
            # Validation
            if out_key not in self.inputs:
                raise ValueError('The output key must be an input to one of the composite models')

            # Remove the out_key from inputs
            # These no longer are an input to the composite model
            # as they are now satisfied internally
            self.inputs.remove(out_key)
                
            # Split the keys into parts (model, key_part)
            (in_model, in_key_part) = in_key.split('.')
            (out_model, out_key_part) = out_key.split('.')

            # Validate parts
            if in_model == out_model:
                raise ValueError('The input and output models must be different')
            if in_model not in self.model_names:
                raise ValueError('The input model must be one of the models in the composite model')
            if out_model not in self.model_names:
                raise ValueError('The output model must be one of the models in the composite model')
            
            # Add to connections
            if in_key in self.states:
                self.__to_input_connections[out_model].append((in_key, out_key_part))
            elif in_key in self.outputs:
                # In output
                self.__to_input_connections[out_model].append((in_key, out_key_part))

                # Add to state (to preserve last between runs)
                self.__to_state_connections[in_model].append((in_key_part, in_key))
                self.states.add(in_key)
            else:
                raise ValueError('The input key must be an output or state of one of the composite models')
        
        # Finish initialization
        super().__init__(**kwargs)

    def initialize(self, u = {}, z = {}):
        x_0 = {}
        # Initialize the models
        for (name, m) in self.parameters['models']:
            u_i = {key: u.get(name + '.' + key, None) for key in m.inputs}
            z_i = {key: z.get(name + '.' + key, None) for key in m.outputs}
            x_i = m.initialize(u_i, z_i)
            for key, value in x_i.items():
                x_0[name + '.' + key] = value
        
            # Process connections
            # This initializes the states that are connected to outputs
            for (in_key_part, in_key) in self.__to_state_connections[name]:
                x_0[in_key] = z.get(in_key, None)
                
        return self.StateContainer(x_0)

    def next_state(self, x, u, dt):
        x_next = x.copy()
        for (name, m) in self.parameters['models']:
            # Prepare inputs
            u_i = {key: u.get(name + '.' + key, None) for key in m.inputs}
            # Process connections
            # This updates the inputs that are connected to states/outputs
            for (in_key, out_key_part) in self.__to_input_connections[name]:
                u_i[out_key_part] = x[in_key]
            u_i = m.InputContainer(u_i)
            
            # Prepare state
            x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})

            # Propogate state
            x_next_i = m.next_state(x_i, u_i, dt)

            # Save to super state
            for key, value in x_next_i.items():
                x_next[name + '.' + key] = value
            
            # Process connections
            # This updates the states that are connected to outputs
            if len(self.__to_state_connections[name]) > 0:
                z_i = m.outputs(x_next_i)
                for (in_key_part, in_key) in self.__to_state_connections[name]:
                    x_next[in_key] = z_i.get(in_key_part, None)

        return self.StateContainer(x_next)

    def output(self, x):
        z = {}
        for (name, m) in self.parameters['models']:
            # Prepare state
            x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})

            # Get outputs
            z_i = m.output(x_i)

            # Save to super outputs
            for key, value in z_i.items():
                z[name + '.' + key] = value
        return self.OutputContainer(z)

    def event_state(self, x):
        e = {}
        for (name, m) in self.parameters['models']:
            # Prepare state
            x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})

            # Get outputs
            e_i = m.event_state(x_i)

            # Save to super outputs
            for key, value in e_i.items():
                e[name + '.' + key] = value
        return e

    def threshold_met(self, x):
        tm = {}
        for (name, m) in self.parameters['models']:
            # Prepare state
            x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})

            # Get outputs
            tm_i = m.threshold_met(x_i)

            # Save to super outputs
            for key, value in tm_i.items():
                tm[name + '.' + key] = value
        return tm
