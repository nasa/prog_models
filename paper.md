---
title: 'ProgPy: Python Packages for Prognostics and Health Management of Engineering Systems'
tags:
  - Python
  - Prognostics
  - Health Management
  - Degradation Simulation
  - Diagnostics
  - State Estimation
  - Prediction
  - CBM+
  - IVHM
  - PHM
authors:
  - name: Christopher Teubert
    orcid: 0000-0001-6788-4507
    equal-contrib: true
    affiliation: 1
  - name: Katelyn Jarvis
    equal-contrib: true
    affiliation: 1
  - name: Matteo Corbetta
    orcid: 0000-0002-7169-1051
    equal-contrib: true
    affiliation: 2
affiliations:
 - name: NASA Ames Research Center, United States
   index: 1
 - name: KBR, Inc. 
   index: 2
date: 7 December 2022
bibliography: paper.bib
---

# Summary
Prognostics of engineering systems or system of systems is the prediction of future performance and/or the time at which one or more events of interest occur. Prognostics can be applied in a variety of applications, from spacecraft and aircraft to wind turbines, oil and gas infrastructure, and assembly lines. The results of prognostics are used to inform action to extend life or prevent failures through changes in use or predictive maintenance.

The NASA Prognostics Python Packages (ProgPy) are a set of open-sourced python packages supporting research and development of prognostics and health management for engineering systems. ProgPy implements architectures and common functionalities of prognostics, supporting both researchers and practitioners.

# Statement of need
Prognostics and Health Management (PHM) is a fast-growing field. Successful PHM application can reduce operational costs and prevent failure, making systems safer. There have been limited application of prognostics in-practice. This is partially because prognostics is a multi-faceted and complex problem, including data availability, sensor design and placement, and, of interest to us, software. Often, software is written for ad-hoc single prognostic application and cannot be transferred to others. There is a need for a foundational set of efficient tools to enable new PHM technologies. 

ProgPy provides a set of support packages for individuals researching and developing prognostic technologies. ProgPy consists of three packages: prog_models, prog_algs, and prog_server. prog_models provides tools aiding the development, evaluation, simulation, and tuning of prognostic models, whether physics-based or data-driven. prog_algs supports uncertainty representation, state estimation, prognostics, the evaluation and visualization of prognostic results, and the creation of new prognostic algorithms. prog_server is a Service Oriented Architecture for prognostics and state estimation. prog_server is also distributed with a python client, prog_client. 

The following sections describe some ways ProgPy could be used. There are many features and use cases beyond those illustrated here. See the [ProgPy documentation](https://nasa.github.io/progpy/index.html) for more information.

# Selected use case: building and simulating models
One of the primary use-cases of ProgPy is building new models. Prognostic models are created by subclassing the PrognosticsModel class. Users can copy the model template as a starting point, replacing the representative member functions with model logic.

Prognostic models have inputs (load/control applied to a system), internal states, outputs (measurable quantities), and events of interest (what we’re predicting). For example, a battery might have inputs of current and outputs of voltage and temperature.

Logic of a prognostic model is defined using the state transition (dx or next_state), output, event_state, and threshold_met functions.

In the below example, a user creates a physics-based model of a Lithium-ion battery. In this model (see [here]( https://github.com/nasa/prog_models/blob/ac7cf016996ac707a6588f41ac54ff747816552a/src/prog_models/models/battery_electrochem.py#L161)), differential equations (i.e. internal states) relates the voltage discharge from the battery (i.e. the outputs) given an applied current (i.e. input). 
```python
class Battery(PrognosticsModel):
    inputs = [
 	‘i’ # current applied to battery 
    ] 
    states = [ # internal battery model states, e.g. temperature, surface 			  potentials, etc. (see ref)
        ‘x_1’, # State 1
        ‘x_2’,  # State 2 
     …  
        ]
    outputs = [
        ‘t’, # Battery temperature
                   ‘v’  # Voltage supplied by battery
    ]
    events = [
       'EOD’, # battery end-of-discharge 
    ]

    # Default parameters. Overwritten by passing parameters into constructor
    default_parameters = {
       ’x0': {  # Initial State
        },
       'param1': p_1, 
        …. # Include parameters to define battery model (see ref)
    }
    
    def dx(self, x, u):
 # calculate derivative of the battery state (ref)
        return self.StateContainer({ })  # Return state container with derivative 

    def output(self, x):
# From the state, calculate temperature and voltage 
        return self.OutputContainer('t': x[‘t'], ‘v’: x[‘v’]})

    def event_state(self, x): 
        # From current state, calculate progress towards EOD
        return {
         ‘EOD': v_now – v_threshold # EOD occurs when voltage drops below						    threshold 
        }
```

The resulting model can then be used in simulation:
```python
m = Battery()
def future_load(t, x=None):  # system loadeding
    return m.InputContainer({‘i’: 1})  # Constant 1 amp applied

simulated_results = m.simulate_to_threshold(future_load, threshold_key‘=['EOD’], dt=0.005, save_freq=1, print = True)

print('EOD was reached in {}seconds'.format(round(simulated_results.times[-1],2)))
```

ProgPy also includes data-driven models such as the LSTM State Transition and Dynamic Mode Decomposition models. These are trained using data and then used for simulation or prognostics, as above.

# Selected use case: prognostics of battery discharge cycle
Models can be used for prognostics with prog_algs. Prognostics is often split into two steps: state estimation and prediction. In state estimation, the system state is estimated, with uncertainty, using the prior state estimate and sensor data. In prediction, the state estimate is predicted forward.

This example illustrates predicting the battery discharge. Here data is retrieved from some unspecified source (data_source). This can be a data stream, playback file, or any other source.

```python
batt = Battery()
x0 = batt.initialize()
state_estimator = state_estimators.ParticleFilter(batt, x0)
predictor = predictors.MonteCarlo(batt)
load = batt.InputContainer({'i': 2.35})
def future_loading(t, x=None):
    return load  # Constant load

while RUNNING:
    u, z = data_source.get_data()
    # Estimate system state using loading (u) and output measurements (z)
    state_estimator.estimate(t, u, z)  
    eod = batt.event_state(filt.x.mean)['EOD']
    print("  - State of charge (mean): ", eod)
    # Only predict every PREDICTION_UPDATE_FREQ steps
    if (step%PREDICTION_UPDATE_FREQ == 0):
        mc_results = mc.predict(filt.x, future_loading, t0 = t, dt=TIME_STEP)
        metrics = mc_results.time_of_event.metrics()
        print('  - Predicted end of discharge: {} (sigma: {})'.format(metrics['EOD']['mean'], metrics['EOD']['std']))
```

# NASA use cases
ProgPy has been used in various NASA projects. Two are described below.
## Data and Reasoning Fabric
ProgPy functionality predicting battery degradation was implemented to assess the Li-ion batteries state of charge during unmanned aerial vehicle (UAV) flight. Based on planned trajectories, ProgPy provided UAV operators with statistics on expected battery health during flight and helped to ensure safety in the national airspace. 

## Autonomous Spacecraft Operations
ProgPy was used to create physics-based and data-driven models predicting the degradation of life support systems on ISS informing maintenance. Researchers evaluated the performance of multiple potential models with data from the system and ProgPy metrics and visualization. Researchers updated models based on performance results. The selected model will be integrated with ProgPy state estimation and prediction into a prognostic application for crew or ground support.
# Acknowledgements
ProgPy is supported by NASA's Autonomous Spacecraft Operations, Data and Reasoning Fabric, System-Wide Safety, and Transformative Tools and Technologies projects. Additionally, development is supported by Northrop Grumman Corporation, Vanderbilt University, the German Aerospace Center (DLR), and others. 

# References
