Release Notes
=======================

..  contents:: 
    :backlinks: top

Updates in V1.3
-----------------------
* **Surrogate Models** Added initial draft of new feature to generate surrogate models automatically from :class:`prog_models.PrognosticsModel`. (See :download:`examples.generate_surrogate <../examples/generate_surrogate.py>` example). Initial implementation uses Dynamic Mode Decomposition. Additional Surrogate Model Generation approaches will be explored for future releases. [Developed by NASA's DRF Project]
* **New Example Models** Added new :class:`prog_models.models.DCMotor`, :class:`prog_models.models.ESC`, and :class:`prog_models.models.Powertrain` models [Developed by NASA's SWS Project]
* **Datasets** Added new feature that allows users to access prognostic datasets programmatically (See :download:`examples.dataset <../examples/dataset.py>`)
* Added new :class:`prog_models.LinearModel` class - Linear Prognostics Models can be represented by a Linear Model. Similar to PrognosticsModels, LinearModels are created by subclassing the LinearModel class. Some algorithms will only work with Linear Models. See :download:`examples.linear_model <../examples/linear_model.py>` example for detail
* Added new StateContainer/InputContainer/OutputContainer objects for classes which allow for data access in matrix form and enforce expected keys. 
* Added new metric for SimResult: :py:func:`prog_models.sim_result.SimResult.monotonicity`.
* :py:func:`prog_models.sim_result.SimResult.plot` now automatically shows legends
* Added drag to :class:`prog_models.models.ThrownObject` model, making the model non-linear. Degree of nonlinearity can be effected using the model parameters (e.g., coefficient of drag cd).
* `observables` from previous releases are now called `performance_metrics`
* model.simulate_to* now returns named tuple, allowing for access by property name (e.g., result.states)
* Updates to :class:`prog_models.sim_result.SimResult`` and :class:`prog_models.sim_result.LazySimResult` for robustness
* Various performance improvements and bug fixes

Note
*****
Now input, states, and output should be represented by model.InputContainer, StateContainer, and OutputContainer, respectively

Note
*****
Python 3.6 is no longer supported.

Updates in V1.2 (Mini-Release)
------------------------------
* New Feature: Vectorized Models
    * Distributed models were vectorized to support vectorized sample-based prognostics approaches
* New Feature: Dynamic Step Sizes
    * Now step size can be a function of time or state
    * See `examples.dynamic_step_size` for more information
* New Feature: New method model.apply_bounds
    * This method allows for other classes to use applied bound limits
* Simulate_to* methods can now specify initial time. Also, outputs are now optional
* Various bug fixes

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
