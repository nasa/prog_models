#from prog_models.models import BatteryCircuit

from math import inf
import numpy as np
import pandas as pd


test_param = pd.DataFrame({  # Set to defaults
        'V0': [4.183],
        'Rp': [1e4],
        'qMax': [7856.3254],
        'CMax': [7777],
        'VEOD': [3.0],
        # Voltage above EOD after which voltage will be considered in SOC calculation
        'VDropoff': [0.1],
        # Capacitance
        'Cb0': [1878.155726],
        'Cbp0': [-230],
        'Cbp1': [1.2],
        'Cbp2': [2079.9],
        'Cbp3': [27.055726],
        # R-C Pairs
        'Rs': [0.0538926],
        'Cs': [234.387],
        'Rcp0': [0.0697776],
        'Rcp1': [1.50528e-17],
        'Rcp2': [37.223],
        'Ccp': [14.8223],
        # Temperature Parameters
        'Ta': [292.1],
        'Jt': [800],
        'ha': [0.5],
        'hcp': [19],
        'hcs': [1]
    })

print(test_param)

#print(test_param.iloc[0,1])

next_test = pd.DataFrame({
            'tb': [292.1],
            'qb': [7856.3254],
            'qcp': [0],
            'qcs': [0]
        })
print(next_test)