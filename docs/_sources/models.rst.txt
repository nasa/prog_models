Models
=============================================================
The Prognostics Model Package is distributed with a few pre-constructed models that can  be used in simulation or prognostics (with the prog_algs package). These models are summarized in the table below with additional detail in the following sections.

..  contents:: 
    :backlinks: top

Models Summary
-------------------------------------------------------------

+----------------------+----------------------------+-----------------------------------+---------------------------------+---------------------------------+
|                      | Battery Model - Circuit    | Battery Model - Electro Chemistry | Centrifugal Pump                | Pneumatic Valve                 |
+======================+============================+===================================+=================================+=================================+
| Events               | End of Discharge (EOD)     | * End of Discharge (EOD)          | * Impeller Wear Failure         | * Leak-Bottom                   |
|                      |                            | * InsufficientCapacity            | * Pump Oil Overheating          | * Leak-Top                      |
|                      |                            |                                   | * Radial Bering Overheat        | * Leak-Internal                 |
|                      |                            |                                   | * Thrust Beiring Overheat       | * Spring Failure                |
|                      |                            |                                   |                                 | * Friction Failure              |
+----------------------+----------------------------+-----------------------------------+---------------------------------+---------------------------------+
| Inputs / Loading     | Current (i)                | Current (i)                       | * Ambient Temperature-K (Tamb)  | * Left Pressure-Pa (pL)         |
|                      |                            |                                   | * Voltage (V)                   | * Right Pressure-Pa (pR)        |
|                      |                            |                                   | * Discharge Pressure-Pa (pdisch)| * Bottom Port Pressure-Pa (uBot)|
|                      |                            |                                   | * Suction Pressure-Pa (psuc)    | * Top Port Pressure-Pa (uTop)   |
|                      |                            |                                   | * Sync Rotational Speed of      |                                 |
|                      |                            |                                   | * supply voltage-rad/sec (wsync)|                                 |
+----------------------+----------------------------+-----------------------------------+---------------------------------+---------------------------------+
|Outputs / Measurements| Voltage (v), Temp °C (t)   | Voltage (v), Temp °C (t)          | * Discharge Flow- m^3/s (Qout)  | * Florrate (Q)                  |
|                      |                            |                                   | * Oil Temp - K (To)             | * Is piston at bottom (iB)      |
|                      |                            |                                   | * Radial Bearing Temp - K (Tr)  | * Is piston at top (iT)         |
|                      |                            |                                   | * Thrust Bearing Temp - K (Tt)  | * Pressure at bottom - Pa (pB)  |
|                      |                            |                                   | * Mech rotation - rad/s (w)     | * Pressure at top - Pa (pT)     |
|                      |                            |                                   |                                 | * Position of piston - m (x)    |
+----------------------+----------------------------+-----------------------------------+---------------------------------+---------------------------------+
| Notes                | Faster and less accurate   | Slower and more accurate than     |                                 |                                 |
|                      | than Electro Chem Model    | Circuit model                     |                                 |                                 |
+----------------------+----------------------------+-----------------------------------+---------------------------------+---------------------------------+

Battery Model - Circuit
-------------------------------------------------------------

.. autoclass:: prog_models.models.BatteryCircuit
   :members:
   :inherited-members:

Battery Model - Electro Chemistry
-------------------------------------------------------------

.. autoclass:: prog_models.models.BatteryElectroChem
   :members:
   :inherited-members:

End of Discharge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.BatteryElectroChemEOD
   :members:
   :inherited-members:

End of Life (i.e., InsufficientCapacity)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.BatteryElectroChemEOL
   :members:
   :inherited-members:

End of Discharge, End of Life (i.e., InsufficientCapacity)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.BatteryElectroChemEODEOL
   :members:
   :inherited-members:



Pump Model
-------------------------------------------------------------

.. autoclass:: prog_models.models.CentrifugalPump

.. autoclass:: prog_models.models.CentrifugalPumpWithWear
   :members:
   :inherited-members:

Pump Model (Base)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.CentrifugalPumpBase
   :members:
   :inherited-members:

Pneumatic Valve
-------------------------------------------------------------

.. autoclass:: prog_models.models.PneumaticValve

.. autoclass:: prog_models.models.PneumaticValveWithWear
   :members:
   :inherited-members:

Pneumatic Valve (Base)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.PneumaticValveBase
   :members:
   :inherited-members:
