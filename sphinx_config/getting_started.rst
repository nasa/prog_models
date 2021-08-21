Getting Started
===============

The NASA Prognostics Models Package is a python framework for defining, building, using, and testing models for prognostics (computation of remaining useful life) of engineering systems, and provides a set of prognostics models for select components developed within this framework, suitable for use in prognostics applications for these components. It can be used in conjunction for the Prognostics Algorithms Library to perform research in prognostics methods. 

The foundation of this package is the class :class:`prog_models.PrognosticsModel`. This class defines the model interface and provides tools for analysis and simulation. New models must either be a subclass of this model or use the model generator method (:py:meth:`prog_models.PrognosticsModel.generate_model`)

A few definitions:

* **events**: some state that can be predicted (e.g., system failure). An event has either occured or not. 

* **event state**: progress towards event occuring. Defined as a number where an event state of 0 indicates the event has occured and 1 indicates no progress towards the event (i.e., fully healthy operation for a failure event). For gradually occuring events (e.g., discharge) the number will progress from 1 to 0 as the event nears. In prognostics, event state is frequently called "State of Health"

* **inputs**: control applied to the system being modeled (e.g., current drawn from a battery)

* **outputs**: measured sensor values from a system (e.g., voltage and temperature of a battery)

* **states**: Internal parameters (typically hidden states) used to represent the state of the system- can be same as inputs/outputs but do not have to be. 

* **process noise**: stochastic process representing uncertainty in the model transition. 

* **measurement noise**: stochastic process representing uncertainty in the measurement process; e.g., sensor sensitivity, sensor misalignements, environmental effects 

Installing
----------
You can install the package using pip:
    `pip install prog_models`

Use 
----
See the below examples for examples of use. Run these examples using the command `python -m examples.[Example name]` (e.g., `python -m examples.sim_example`). Note: some windows users would use `py -m examples.[Example name]` instead of python. The examples are summarized below:

* :download:`examples.sim <../examples/sim.py>`
    .. automodule:: examples.sim
    |
* :download:`examples.sim_valve <../examples/sim_valve.py>`
    .. automodule:: examples.sim_valve
    |
* :download:`examples.model_gen <../examples/model_gen.py>`
    .. automodule:: examples.model_gen
    |
* :download:`examples.benchmarking <../examples/benchmarking.py>`
    .. automodule:: examples.benchmarking
    |
* :download:`examples.new_model <../examples/new_model.py>`
    .. automodule:: examples.new_model
    |
* :download:`examples.sensitivity <../examples/sensitivity.py>`
    .. automodule:: examples.sensitivity
    |
* :download:`examples.noise <../examples/noise.py>`
    .. automodule:: examples.noise
    |
* :download:`examples.visualize <../examples/visualize.py>`
    .. automodule:: examples.visualize
    |
* :download:`examples.future_loading <../examples/future_loading.py>`
    .. automodule:: examples.future_loading
    |
* :download:`examples.param_est <../examples/param_est.py>`
    .. automodule:: examples.param_est
    |
* :download:`examples.derived_params <../examples/derived_params.py>`
    .. automodule:: examples.derived_params
    |
* :download:`examples.state_limits <../examples/state_limits.py>`
    .. automodule:: examples.state_limits
    |

There is also an included tutorial (:download:`tutorial <../tutorial.ipynb>`).

Model Specific examples
------------------------
* :download:`examples.sim_battery_eol <../examples/sim_battery_eol.py>`
    .. automodule:: examples.sim_battery_eol
    |
* :download:`examples.sim_pump <../examples/sim_pump.py>`
    .. automodule:: examples.sim_pump
    |
* :download:`examples.sim_valve <../examples/sim_valve.py>`
    .. automodule:: examples.sim_valve
    |

Extending
----------
There are two methods for creating new models: 1. Implementing a new subclass of the base model class (:class:`prog_models.PrognosticsModel`) or 2. use the model generator method (:py:meth:`prog_models.PrognosticsModel.generate_model`). These methods are described more below.

1. Subclass Method
********************
The first method for creating a new prog_model is by creating a new subclass of :class:`prog_models.PrognosticsModel`. :class:`prog_models.PrognosticsModel` is described below:

.. autoclass:: prog_models.PrognosticsModel

To generate a new model create a new class for your model that inherits from this class. Alternatively, you can copy the template :class:`prog_model_template.ProgModelTemplate`, replacing the methods with logic defining your specific model.

The analysis and simulation tools defined in :class:`prog_models.PrognosticsModel` will then work with your new model. 

See :download:`examples.new_model <../examples/new_model.py>` for an example of this approach.

2. Model Generator Method
*************************
The second way to generate a new model is using the model generator method, :py:meth:`prog_models.PrognosticsModel.generate_model`. Pass a dictionary of the keys for input, state, output, events (optional), and the required transition into the method, and it will return a constructed model. See :py:meth:`prog_models.PrognosticsModel.generate_model` for more detail.

See :download:`examples.model_gen <../examples/model_gen.py>` for an example of this approach.

Updates in V1.1
---------------
* New Feature: Derived Parameters
    * Users can specify callbacks for parameters that are defined from others. These callbacks will be called when the dependency parameter is updated.
    * See `examples.derived_params` for more information.
* New Feature: Parameter Estimation
    * Users can use the estimate_parameters method to estimate all or select parameters. 
    * see `examples.param_est`
* New Feature: Automatic Noise Generation
    * Now noise is automatically generated when next_state/dx (process_noise) and output (measurement_noise). This removed the need to explicitly call apply_*_noise functions in these methods. 
    * See `examples.noise` for more details in setting noise
    * For any classes users created using V1.0.*, you should remove any call to apply_*_noise functions to prevent double noise application. 
* New Feature: Configurable State Bounds
    * Users can specify the range of valid values for each state (e.g., a temperature in celcius would have to be greater than -273.15 - absolute zero)
* New Feature: Simulation Result Class
    * Simulations now return a simulation result object for each value (e.g., output, input, state, etc) 
    * These simulation result objects can be used just like the previous lists. 
    * Output and Event State are now "Lazily Evaluated". This speeds up simulation when intermediate states are not printed and these properties are not used
    * A plot method has been added directly to the class (e.g., `event_states.plot()`)
* New Feature: Intermediate Result Printing
    * Use the print parameter to enable printing intermediate results during a simulation 
    * e.g., `model.simulate_to_threshold(..., print=True)`
    * Note: This slows down simulation performance
* Added support for python 3.9
* Various bug fixes

ElectroChemistry Model Updates
************************************
* New Feature: Added thermal effects. Now the model include how the temperature is effected by use. Previous implementation only included effects of temperature on performance.
* New Feature: Added `degraded_capacity` (i.e., EOL) event to model. There are now three different models: BatteryElectroChemEOL (degraded_capacity only), BatteryElectroChemEOD (discharge only), and BatteryElectroChemEODEOL (combined). BatteryElectroChem is an alias for BatteryElectroChemEODEOL. 
* New Feature: Updated SOC (EOD Event State) calculation to include voltage when near V_EOD. This prevents a situation where the voltage is below lower bound but SOC > 0. 

CentrifugalPump Model Updates
************************************
* New Feature: Added CentrifugalPumpBase class where wear rates are parameters instead of part of the state vector. 
    * Some users may use this class for prognostics, then use the parameter estimation tool occasionally to update the wear rates, which change very slowly.
* Bugfix: Fixed bug where some event states were returned as negative
* Bugfix: Fixed bug where some states were saved as parameters instead of part of the state. 
* Added example on use of CentrifugalPump Model (see `examples.sim_pump`)
* Performance improvements

PneumaticValve Model Updates
************************************
* New Feature: Added PneumaticValveBase class where wear rates are parameters instead of part of the state vector. 
    * Some users may use this class for prognostics, then use the parameter estimation tool occasionally to update the wear rates, which change very slowly.
* Added example on use of PneumaticValve Model (see `examples.sim_valve`)

Tips
----
* To predict a certain partial state (e.g., 50% SOH), create a new event (e.g., 'SOH_50') override the event_state and threshold_met equations to also predict that additional state
* If you're only doing diagnostics without prognostics- just define a next_state equation with no change of state and don't perform prediction. The state estimator can still be used to estimate if any of the events have occured. 
* Sudden events use a binary event_state (1=healthy, 0=failed)
* You can predict as many events as you would like, sometimes one event must happen before another, in this case the event occurance for event 1 can be a part of the equation for event 2 ('event 2': event_1 and [OTHER LOGIC])
