# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
from prog_models.models.battery_circuit import BatteryCircuit
from prog_models.models.battery_electrochem import BatteryElectroChem

class TestBattery(unittest.TestCase):
    def test_battery_circuit(self):
        batt = BatteryCircuit()
        # TODO(CT): More

    def test_battery_electrochem(self):
        batt = BatteryElectroChem()
        # TODO(CT): More