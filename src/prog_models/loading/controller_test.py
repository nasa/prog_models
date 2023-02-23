import numpy as np

class ExampleController():

    def __init__(self, x_ref, **kwargs):

        # Inputs to controller are vehicle parameters and x_ref
        self.parameters = kwargs
        self.ref_traj = x_ref
    
    def __call__(self, t, x=None):
        # Extract current state at t
        if x is None:
            x_k = np.zeros((12,1))
        else:
            x_k = np.array([x.matrix[ii][0] for ii in range(len(x.matrix)-1)])
        
        # Identify reference (desired state) at t
        t_k      = np.round(t + self.parameters['dt']/2,1) 
        time_ind = np.argmin(np.abs(t_k - self.ref_traj['t'].tolist()))
        # x_ref_k  = np.concatenate ...  axis=0 
        x_ref_k  = np.array((self.ref_traj['x'][time_ind], self.ref_traj['y'][time_ind], self.ref_traj['z'][time_ind], 
                                   self.ref_traj['phi'][time_ind], self.ref_traj['theta'][time_ind], self.ref_traj['psi'][time_ind], 
                                   self.ref_traj['vx'][time_ind], self.ref_traj['vy'][time_ind], self.ref_traj['vz'][time_ind], 
                                   self.ref_traj['p'][time_ind], self.ref_traj['q'][time_ind], self.ref_traj['r'][time_ind]))
        
        # Compute system input using the error between current and reference state as input to the controller
        # u     = self.vehicle_model.control_fn(x_k - x_ref_k)                      # compute differential input values from error
        # u[0] += self.vehicle_model.steadystate_input                              # add steady-state input (defined as hover condition)
        # u[0]  = min(max([0., u[0]]), self.vehicle_model.dynamics['max_thrust'])   # limit thrust between 0 and vehicle's max thrust
        u = abs(x_k - x_ref_k)

        debug = 1

        # return self.InputContainer({'T': u[0], 'mx': u[1], 'my': u[2], 'mz': u[3]})
        # return ({'T': u[0], 'mx': u[1], 'my': u[2], 'mz': u[3]})
        return ({'T': 0, 'mx': 0, 'my': 0, 'mz': 0})

