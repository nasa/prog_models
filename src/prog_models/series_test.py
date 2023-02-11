# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

#from prognostics_model_df import PrognosticsModel

import pandas as pd
from prog_models.sim_result_df import SimResultDF

# battery circuit data


saved_times = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]
saved_inputs = [{'i': 2.0}, {'i': 2.0}, {'i': 2.0}, {'i': 2.0}, {'i': 2.0}, {'i': 2.0}, {'i': 2.0}, {'i': 2.0}, {'i': 2.0}, {'i': 2.0}, {'i': 2.0}]
saved_states = [{'tb': 292.1, 'qb': 7856.3254, 'qcp': 0.0, 'qcs': 0.0}, {'tb': 292.22644204150873, 'qb': 7816.317409660615, 'qcp': 2.0689378843844732, 'qcs': 20.41326287692566}, {'tb': 292.3610888245951, 'qb': 7776.309533253024, 'qcp': 2.0689354519476315, 'qcs': 24.33552608303554}, {'tb': 292.49484966409074, 'qb': 7736.301684463998, 'qcp': 2.068934546518363, 'qcs': 25.089157295472866}, {'tb': 292.6271089174722, 'qb': 7696.293849418812, 'qcp': 2.0689339377980582, 'qcs': 25.233957148618543}, {'tb': 292.7577558424454, 'qb': 7656.286025387842, 'qcp': 2.068933389371525, 'qcs': 25.261774177023153}, {'tb': 292.8867852787106, 'qb': 7616.2782117833385, 'qcp': 2.0689328557918225, 'qcs': 25.267113779720262}, {'tb': 293.0142124723636, 'qb': 7576.270408429509, 'qcp': 2.068932328305495, 'qcs': 25.26813453545151}, {'tb': 293.14005640048686, 'qb': 7536.262615230134, 'qcp': 2.0689318052097034, 'qcs': 25.26832549315709}, {'tb': 293.2643365580705, 'qb': 7496.254832104665, 'qcp': 2.0689312861565456, 'qcs': 25.268357049951057}, {'tb': 293.3870723426943, 'qb': 7456.247058975978, 'qcp': 2.068930771058666, 'qcs': 25.268358017893604}]
saved_outputs = [{'t': 292.1, 'v': 4.1828702595530105}, {'t': 292.22644204150873, 'v': 3.950865081712307}, {'t': 292.3610888245951, 'v': 3.9288409870976757}, {'t': 292.49484966409074, 'v': 3.9203754582002106}, {'t': 292.6271089174722, 'v': 3.914547058110963}, {'t': 292.7577558424454, 'v': 3.9092570975639114}, {'t': 292.8867852787106, 'v': 3.904102121211562}, {'t': 293.0142124723636, 'v': 3.8990044036060914}, {'t': 293.14005640048686, 'v': 3.8939488081511775}, {'t': 293.2643365580705, 'v': 3.8889322262832393}, {'t': 293.3870723426943, 'v': 3.883953862519164}]
saved_event_states = [{'EOD': 1.0000418413269898}, {'EOD': 0.9948974424148921}, {'EOD': 0.9897530581526326}, {'EOD': 0.9846086774416868}, {'EOD': 0.9794642984979828}, {'EOD': 0.9743199209705339}, {'EOD': 0.9691755447837648}, {'EOD': 0.9640311699150712}, {'EOD': 0.958886796352081}, {'EOD': 0.9537424240844369}, {'EOD': 0.9485980531022217}]
result_list = [saved_inputs, saved_states, saved_outputs, saved_event_states]

pd_list = SimResultDF(saved_times, result_list)
print(pd_list.data_df)

saved_times2 = [x+220 for x in saved_times] # create new list of timestamps


pd_list1 = SimResultDF(saved_times2, result_list)   # create new SimResultDF object
print(type(pd_list.data_df))

pd_list.extend(pd_list1)    # test extend for dataframes

print(pd_list.data_df)

print(pd_list.dfpop('i', 100.0))  # test for pop
print(pd_list)




