# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
from prog_models.models import BatteryCircuit, BatteryElectroChem, BatteryElectroChemEOL, BatteryElectroChemEOD, BatteryElectroChemEODEOL

def future_loading(t, x=None):
    # Variable (piece-wise) future loading scheme 
    if (t < 600):
        i = 2
    elif (t < 900):
        i = 1
    elif (t < 1800):
        i = 4
    elif (t < 3000):
        i = 2
    else:
        i = 3
    return {'i': i}

class TestBattery(unittest.TestCase):
    def test_battery_circuit(self):
        batt = BatteryCircuit()
        (times, inputs, states, outputs, event_states) = batt.simulate_to(200, future_loading, {'t': 18.95, 'v': 4.183})

    def test_battery_electrochem(self):
        batt = BatteryElectroChem()
        (times, inputs, states, outputs, event_states) = batt.simulate_to(200, future_loading, {'t': 18.95, 'v': 4.183})
        self.assertEqual(BatteryElectroChem, BatteryElectroChemEODEOL)

    def test_battery_electrochem_EOD(self):
        batt = BatteryElectroChemEOD()
        (times, inputs, states, outputs, event_states) = batt.simulate_to(200, future_loading, {'t': 18.95, 'v': 4.183})

    def test_battery_electrochem_EOL(self):
        batt = BatteryElectroChemEOL()
        (times, inputs, states, outputs, event_states) = batt.simulate_to(200, future_loading, {'t': 18.95, 'v': 4.183})

# This allows the module to be executed directly
def run_tests():
    unittest.main()
    
def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Battery models")
    result = runner.run(l.loadTestsFromTestCase(TestBattery)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
