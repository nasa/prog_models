from prog_models.models import DCMotor
from .esc_loading_fcn import esc_loading

motor = DCMotor(process_noise=0)
first_output = {
    'v_rot': 0,
    'theta': 0
}

(times, inputs, states, outputs, event_states) = motor.simulate_to(100, esc_loading, first_output, print=True)
