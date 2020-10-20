from . import model 

class PrognosticsModel(model.Model):
    """
    """
    events = []
    
    def threshold_met(self, t, x):
        pass
 
    def simulate_to_threshold(self, future_loading_eqn, first_output, options = {}):
        config = { # Defaults
            'step_size': 1,
            'save_freq': 10
        }
        config.update(options)
        # TODO(CT): Add checks (e.g., stepsize, save_freq > 0)

        t = 0
        u = future_loading_eqn(t)
        x = self.initialize(u, first_output)
        times = [t]
        inputs = [u]
        states = [x]
        outputs = [first_output]
        event_states = [self.event_state(t, x)]
        next_save = config['save_freq']
        threshold_met = False
        while not threshold_met:
            t += config['step_size']
            u = future_loading_eqn(t)
            x = self.state(t, x, u, config['step_size'])
            thresholds_met = self.threshold_met(t, x)
            threshold_met = any(thresholds_met.values())
            if (t >= next_save):
                next_save += config['save_freq']
                times.append(t)
                inputs.append(u)
                states.append(x)
                outputs.append(self.output(t, x))
                event_states.append(self.event_state(t, x))
        if times[-1] != t:
            times.append(t)
            inputs.append(u)
            states.append(x)
            outputs.append(self.output(t, x))
            event_states.append(self.event_state(t, x))
        return {
            't': times,
            'u': inputs,
            'x': states,
            'z': outputs, 
            'event_state': event_states
        }