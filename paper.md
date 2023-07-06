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
    equal-contrib: false
    affiliation: 1
  - name: Katelyn Jarvis
    equal-contrib: false
    affiliation: 1
  - name: Matteo Corbetta
    orcid: 0000-0002-7169-1051
    equal-contrib: false
    affiliation: 2
  - name: Chetan Kulkarni
    equal-contrib: false
    affiliation: 2
  - name: Matthew Daigle
    equal-contrib: false
    affiliation: 2
affiliations:
 - name: NASA Ames Research Center, United States
   index: 1
 - name: KBR, Inc. 
   index: 2
date: 12 December 2022
bibliography: paper.bib
---

# Summary
Prognostics of engineering systems or systems of systems is the prediction of future performance and/or the time at which one or more events of interest occur. Prognostics can be applied in a variety of applications, from spacecraft and aircraft to wind turbines, oil and gas infrastructure, and assembly lines. Prognostic results are used to inform action to extend life or prevent failures through changes in use or predictive maintenance.

The [NASA Prognostics Python Packages (ProgPy)](https://nasa.github.io/progpy/) [@2022_nasa_progpy] are a set of open-sourced Python packages supporting research and development of prognostics and health management for engineering systems, as described in [@goebel2017prognostics]. ProgPy builds upon the architecture of the Matlab Prognostics Libraries [@2016_nasa_prog_model_library; @2016_nasa_prog_algs_library; @2016_nasa_prog_metrics_library], Generic Software Architecture for Prognostics [@teubert2017generic], and Prognostics As-A-Service [@watkins2019prognostics]. ProgPy implements architectures and common functionalities of prognostics, supporting both researchers and practitioners.

# Statement of need
Prognostics and Health Management (PHM) is a fast-growing field. Successful PHM application can reduce operational costs and prevent failure, making systems safer. There has been limited application of prognostics in practice. This is partially because prognostics is a multi-faceted and complex problem, including data availability, sensor design and placement, and, of interest to us, software. 

Often, software is written for an ad-hoc single prognostic application and cannot be transferred to others, or is limited in scope. A few related packages are described here. Simantha is a discrete manufacturing system simulation package that simulates degradation, but it is limited to a Discrete-Time Markov Chain and doesn't include prognostic capabilities [@Simantha]. Lifelines is a survival analysis tool that can be used for reliability analysis to establish fixed-interval maintenance schedules, a different problem than that solved by ProgPy [@davidson_pilon_cameron_2022_7329096]. Pomegranate is a Python probabilistic modeling package effective for data science applications [@schreiber2017pomegranate]. However, Pomegranate does not include explicit state estimation capabilities, prognostics tools, or physics-based degradation modeling features. Finally, there are a number of general machine-learning packages such as TensorFlow, scikit learn, and PyTorch. These are general tools that can be used for diagnostics and prognostics, but are not designed specifically for that application.

There is a need for a foundational set of efficient tools to enable new PHM technologies. 

ProgPy provides a set of support packages for individuals researching and developing prognostic technologies. ProgPy consists of three packages: `prog_models`, `prog_algs`, and `prog_server`. `prog_models` provides tools aiding the development, evaluation, simulation, and tuning of prognostic models, whether physics-based or data-driven. `prog_models` also supports downloading select relevant datasets [@Dataset_RWBattery; @Dataset_CMAPPS]. `prog_algs` supports uncertainty representation, state estimation, prognostics, the evaluation and visualization of prognostic results, and the creation of new prognostic algorithms. `prog_server` is a Service-Oriented Architecture for prognostics and state estimation. prog_server is also distributed with a Python client, `prog_client`. 

The following sections describe some ways ProgPy could be used. There are many features and use cases beyond those illustrated here. See the [ProgPy documentation](https://nasa.github.io/progpy/) for more information.

# Selected use case: building and simulating models
One of the primary use-cases of ProgPy is building new models. Prognostic models are created by subclassing the `PrognosticsModel` class. Users can copy the model template as a starting point, replacing the representative member functions with model logic.

Prognostic models have inputs (load/control applied to a system), internal states, outputs (measurable quantities), and events of interest (what we’re predicting).
Logic of a prognostic model is defined using the state transition (`dx` or `next_state`), `output`, `event_state`, and `threshold_met` functions.

In the below example, a user creates a physics-based model of a Lithium-ion battery. In this model (see [here]( https://github.com/nasa/prog_models/blob/ac7cf016996ac707a6588f41ac54ff747816552a/src/prog_models/models/battery_electrochem.py#L161)), state transition equations (i.e., internal states) relate the voltage discharge from the battery (i.e., the output) given an applied current (i.e., the input). 

```python
class Battery(PrognosticsModel):
    inputs = [
 	‘i’ # current applied to battery 
    ] 
    states = [ 
        # internal battery model states, e.g., temperature, surface potentials
        # nasa.github.io/progpy/api_ref/prog_models/IncludedModels.html
        ‘x_1’, # State 1
        ‘x_2’,  # State 2 
    	      …  
        ]
    outputs = [
        ‘t’, # Battery temperature
        ‘v’  # Voltage supplied by battery
    ]
    events = [
       'EOD' # battery end-of-discharge 
    ]

    # Default parameters. Overwritten by passing parameters into constructor
    default_parameters = {
       'x0':{  # Initial State
        },
       'param1':p_1, 
        …. 
        # Include parameters to define battery model
        # nasa.github.io/progpy/api_ref/prog_models/IncludedModels.html
    }
    
    def dx(self, x, u):
        # calculate derivative of the battery state
        return self.StateContainer({})  # Return state container with derivative 

    def output(self, x):
        # From the state, calculate temperature and voltage 
        return self.OutputContainer({'t': x['t'], 'v': x['v']})

    def event_state(self, x): 
        # From current state, calculate progress towards EOD
        return {
         'EOD': v_now – v_threshold 
         # EOD occurs when voltage drops below threshold 
        }
```

The resulting model can then be used in simulation:
```python
m = Battery()
def future_load(t, x=None):  # system loading
    return m.InputContainer({‘i’:1})  # Constant 1 amp applied

simulated_results = m.simulate_to_threshold(future_load, dt=0.005)

print(f'EOD was reached in {round(simulated_results.times[-1],2)}seconds') 
```

ProgPy also includes data-driven models such as the LSTM State Transition and Dynamic Mode Decomposition models. These are trained using data and then used for simulation or prognostics, as above.

# Selected use case: prognostics of battery discharge cycle
Models can be used for prognostics with `prog_algs`. Prognostics is often split into two steps: state estimation and prediction. In state estimation, the system state is estimated, with uncertainty, using the prior state estimate and sensor data. In prediction, the state estimate is predicted forward.

This example illustrates predicting the battery discharge. Here data is retrieved from some unspecified source (`data_source`). This can be a data stream, playback file, or any other source. This is similar to the `sim_battery_eol` example (see [here](https://github.com/nasa/prog_models/blob/master/examples/)).

```python
batt = Battery()
x0 = batt.initialize()
# Create Particle Filter State Estimator
state_estimator = state_estimators.ParticleFilter(batt, x0)
# Create Monte Carlo Predictor
predictor = predictors.MonteCarlo(batt)

# Future loading as function of time (t) and state (x)
# In this case- constant load
def future_loading(t, x=None):
    return batt.InputContainer({'i':2.35})

while RUNNING:
    u, z = data_source.get_data()
    # Estimate state using loading (u) and output measurements (z)
    state_estimator.estimate(t, u, z)  
    eod = batt.event_state(filt.x.mean)['EOD']
    print(f"  - State of charge (mean): {eod}")
    # Only predict every PREDICTION_UPDATE_FREQ steps
    if (step%PREDICTION_UPDATE_FREQ==0):
        mc_results = mc.predict(filt.x, future_loading, t0 = t, dt=TIME_STEP)
        metrics = mc_results.time_of_event.metrics()
        eod_mean = metrics['EOD']['mean']
        eod_std = metrics['EOD']['std']))
        print(f'  - Predicted end of discharge: {eod_mean} (sigma: {eod_std})')
```
# NASA use cases
ProgPy has been used in various NASA projects. Two are described below.

## Data and Reasoning Fabric 
ProgPy functionality predicting battery degradation was implemented to assess the Li-ion batteries state of charge during unmanned aerial vehicle (UAV) flight. Based on planned trajectories, ProgPy provided UAV operators with statistics on expected battery health during flight and helped to ensure safety in the national airspace [@jarvis2022enabling].

## Autonomous Spacecraft Operations
ProgPy was used to create models predicting the ISS life support system degradation informing maintenance. Researchers evaluated the performance of multiple potential models with data from the system and ProgPy metrics and visualization. Researchers updated models based on performance results. The selected model will be integrated with ProgPy state estimation and prediction into a prognostic application for crew or ground support.

# Acknowledgements
ProgPy is supported by NASA's Autonomous Spacecraft Operations, Data and Reasoning Fabric, System-Wide Safety, and Transformative Tools and Technologies projects. Additionally, development is supported by Northrop Grumman Corporation, Vanderbilt University, the German Aerospace Center (DLR), Research Institutes of Sweden and others. 

# References
