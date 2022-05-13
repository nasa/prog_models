# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

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

        # check warning raised when changing overwritten parameter
        with self.assertWarns(UserWarning):
            batt.parameters['Ro'] = 10
        
        with self.assertWarns(UserWarning):
            batt.parameters['qMobile'] = 10

        with self.assertWarns(UserWarning):
            batt.parameters['tDiffusion'] = 10

    def test_battery_electrochem_EOD(self):
        batt = BatteryElectroChemEOD()
        (times, inputs, states, outputs, event_states) = batt.simulate_to(200, future_loading, {'t': 18.95, 'v': 4.183})

    def test_battery_electrochem_EOL(self):
        batt = BatteryElectroChemEOL()
        (times, inputs, states, outputs, event_states) = batt.simulate_to(200, future_loading, {'t': 18.95, 'v': 4.183})

    def test_batt_namedtuple_access(self):
        batt = BatteryElectroChemEOL()
        named_results = batt.simulate_to(200, future_loading, {'t': 18.95, 'v': 4.183})
        # Can't test for equality, sim values different each run. Test assignment
        times = named_results.times
        inputs = named_results.inputs
        states = named_results.states
        outputs = named_results.outputs
        event_states = named_results.event_states

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
