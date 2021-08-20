import prog_models
from math import inf

class CombinedLoads(prog_models.PrognosticsModel):
    outputs = ["current"]
    systems = {}

    def __init__(self, systems, **kwargs):
        self.inputs = []
        self.states = []
        kwargs['process_noise'] = False
        
        for (name, system) in systems.items():
            self.systems[name] = {
                'model': system, 
                'input_dict': {"{}.{}".format(name, key): key for key in system.inputs},
                'state_dict': {key: "{}.{}".format(name, key) for key in system.states}
            }
            self.inputs += list(self.systems[name]['input_dict'].keys())
            self.states += list(self.systems[name]['state_dict'].values())

        super().__init__(**kwargs)

    def __str__(self):
        response = "Combined Loads:"
        for (key, system) in self.systems.items():
            response += "\n\t* {}: {}".format(key, str(system['model']))
        return response

    def initialize(self, u={}, z={}):
        x0 = {}
        for system in self.systems.values():
            print(system)
            u_system = {}
            for (key, system_key) in system['input_dict'].items():
                if key in u:
                    u_system[system_key] = u[key]
            x_system = system['model'].initialize(u=u_system, z=z)
            for (key, value) in x_system.items():
                x0[system['state_dict'][key]] = value

        return x0
    
    def next_state(self, x, u, dt):
        x_prime = {}
        for system in self.systems.values():
            x_system = {
                model_key: x[key] for (model_key, key) in system['state_dict'].items()
            }
            u_system = {
                model_key: u[key] for (key, model_key) in system['input_dict'].items()
            }
            x_prime_system = system['model'].next_state(x_system, u_system, dt)
            x_prime.update(
                {key: x_prime_system[model_key] for (model_key, key) in system['state_dict'].items()}
                )
        return x_prime

    def output(self, x):
        z = []
        for system in self.systems.values():
            x_system = {
                model_key: x[key] for (model_key, key) in system['state_dict'].items()
            }
            z.append(system['model'].output(x_system)['current'])
        return {'current': sum(z)}
