Models
=============================================================
The Prognostics Model Package is distributed with a few pre-constructed models that can  be used in simulation or prognostics (with the prog_algs package). These models are summarized in the table below with additional detail in the following sections.

..  contents:: 
    :backlinks: top

Models Summary
-------------------------------------------------------------

+----------------------+----------------------------+-----------------------------------+---------------------------------+
|                      | Battery Model - Circuit    | Battery Model - Electro Chemistry | Centrifugal Pump                |
+======================+============================+===================================+=================================+
| Description          | Battery equivilant circuit | Battery Electro Chemistry Model   |                                 |
+----------------------+----------------------------+-----------------------------------+---------------------------------+
| Events               | End of Discharge (EOD)     | End of Discharge (EOD)            | * Impeller Wear Failure         |
|                      |                            |                                   | * Pump Oil Overheating          |
|                      |                            |                                   | * Radial Bering Overheat        |
|                      |                            |                                   | * Thrust Beiring Overheat       |
+----------------------+----------------------------+-----------------------------------+---------------------------------+
| Inputs               | Current (i)                | Current (i)                       | * Ambient Temperature-K (Tamb)  |
|                      |                            |                                   | * Voltage (V)                   |
|                      |                            |                                   | * Discharge Pressure-Pa (pdisch)|
|                      |                            |                                   | * Suction Pressure-Pa (psuc)    |
|                      |                            |                                   | * Sync Rotational Speed of      |
|                      |                            |                                   | * supply voltage-rad/sec (wsync)|
+----------------------+----------------------------+-----------------------------------+---------------------------------+
|Outputs / Measurements| Voltage (v), Temp °C (t)   | Voltage (v), Temp °C (t)          | * Discharge Flow- m^3/s (Qout)  |
|                      |                            |                                   | * Oil Temp - K (To)             |
|                      |                            |                                   | * Radial Bearing Temp - K (Tr)  |
|                      |                            |                                   | * Thrust Bearing Temp - K (Tt)  |
|                      |                            |                                   | * Mech rotation - rad/s (w)     |
+----------------------+----------------------------+-----------------------------------+---------------------------------+
| Notes                | Faster and less accurate   | Slower and more accurate than     |                                 |
|                      | than Electro Chem Model    | Circuit model                     |                                 |
+----------------------+----------------------------+-----------------------------------+---------------------------------+

Battery Model - Circuit
-------------------------------------------------------------

.. autoclass:: prog_models.models.battery_circuit.BatteryCircuit
   :members:
   :inherited-members:

Battery Model - Electro Chemistry
-------------------------------------------------------------

.. autoclass:: prog_models.models.battery_electrochem.BatteryElectroChem
   :members:
   :inherited-members:

 Pump Model
-------------------------------------------------------------

.. autoclass:: prog_models.models.centrifugal_pump.CentrifugalPump
   :members:
   :inherited-members: