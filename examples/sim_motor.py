from prog_models.models import DCMotor
from esc_loading_fcn import esc_loading

def run_example():
    motor = DCMotor(process_noise=0)
    first_output = {
        'v_rot': 0,
        'theta': 0
    }

    (times, inputs, states, outputs, event_states) = motor.simulate_to(5, esc_loading, first_output, dt=2e-5, save_freq=0.1, print=True)

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()
