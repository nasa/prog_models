Models
=============================================================
The Prognostics Model Package is distributed with a few pre-constructed models that can  be used in simulation or prognostics (with the prog_algs package). These models are summarized in the table below with additional detail in the following sections.

..  contents:: 
    :backlinks: top

Models Summary
-------------------------------------------------------------

+----------------------+------------------------------------------------------+--------------------------------------------------------------------------+------------------------------------+----------------------------------------+
|                      | `Battery Model - Circuit <#battery-model-circuit>`__ | `Battery Model - Electro Chemistry <#battery-model-electro-chemistry>`__ | `Centrifugal Pump <#pump-model>`__ | `Pneumatic Valve <#pneumatic-valve>`__ |
+======================+======================================================+==========================================================================+====================================+========================================+
| Events               | End of Discharge (EOD)                               | * End of Discharge (EOD)                                                 | * Impeller Wear Failure            | * Leak-Bottom                          |
|                      |                                                      | * Insufficient Capacity                                                  | * Pump Oil Overheating             | * Leak-Top                             |
|                      |                                                      |                                                                          | * Radial Bering Overheat           | * Leak-Internal                        |
|                      |                                                      |                                                                          | * Thrust Beiring Overheat          | * Spring Failure                       |
|                      |                                                      |                                                                          |                                    | * Friction Failure                     |
+----------------------+------------------------------------------------------+--------------------------------------------------------------------------+------------------------------------+----------------------------------------+
| Inputs / Loading     | Current (i)                                          | Current (i)                                                              | * Ambient Temperature-K (Tamb)     | * Left Pressure-Pa (pL)                |
|                      |                                                      |                                                                          | * Voltage (V)                      | * Right Pressure-Pa (pR)               |
|                      |                                                      |                                                                          | * Discharge Pressure-Pa (pdisch)   | * Bottom Port Pressure-Pa (uBot)       |
|                      |                                                      |                                                                          | * Suction Pressure-Pa (psuc)       | * Top Port Pressure-Pa (uTop)          |
|                      |                                                      |                                                                          | * Sync Rotational Speed of         |                                        |
|                      |                                                      |                                                                          | * supply voltage-rad/sec (wsync)   |                                        |
+----------------------+------------------------------------------------------+--------------------------------------------------------------------------+------------------------------------+----------------------------------------+
|Outputs / Measurements| Voltage (v), Temp °C (t)                             | Voltage (v), Temp °C (t)                                                 | * Discharge Flow- m^3/s (Qout)     | * Florrate (Q)                         |
|                      |                                                      |                                                                          | * Oil Temp - K (To)                | * Is piston at bottom (iB)             |
|                      |                                                      |                                                                          | * Radial Bearing Temp - K (Tr)     | * Is piston at top (iT)                |
|                      |                                                      |                                                                          | * Thrust Bearing Temp - K (Tt)     | * Pressure at bottom - Pa (pB)         |
|                      |                                                      |                                                                          | * Mech rotation - rad/s (w)        | * Pressure at top - Pa (pT)            |
|                      |                                                      |                                                                          |                                    | * Position of piston - m (x)           |
+----------------------+------------------------------------------------------+--------------------------------------------------------------------------+------------------------------------+----------------------------------------+

Battery Model - Circuit
-------------------------------------------------------------

.. autoclass:: prog_models.models.BatteryCircuit

Battery Model - Electro Chemistry
-------------------------------------------------------------

There are three different flavors of Electro Chemistry Battery Models distributed with the package, described below

End of Discharge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.BatteryElectroChemEOD

End of Life (i.e., InsufficientCapacity)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.BatteryElectroChemEOL

End of Discharge, End of Life (i.e., InsufficientCapacity & EOD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.BatteryElectroChem

.. autoclass:: prog_models.models.BatteryElectroChemEODEOL


Pump Model
-------------------------------------------------------------

There are two variants of the pump model based on if the wear parameters are estimated as part of the state. The models are described below

Pump Model (Base)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.CentrifugalPumpBase

Pump Model (With Wear)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.CentrifugalPump

.. autoclass:: prog_models.models.CentrifugalPumpWithWear

Pneumatic Valve
-------------------------------------------------------------

There are two variants of the valve model based on if the wear parameters are estimated as part of the state. The models are described below

Pneumatic Valve (Base)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.PneumaticValveBase

Pneumatic Valve (With Wear)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.PneumaticValve

.. autoclass:: prog_models.models.PneumaticValveWithWear
