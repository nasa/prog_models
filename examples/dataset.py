# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example downloading and using a NASA prognostics dataset.

In this example, a battery dataset is downloaded from the NASA PCoE data repository. This dataset is then accessed and plotted. 
"""

DATASET_ID = 1

def run_example():
    # Step 1: Download and import the dataset for a single battery
    from prog_models.datasets import nasa_battery
    (desc, data) = nasa_battery.load_data(DATASET_ID)

    # Step 2: Access the dataset description
    print(f'Dataset {DATASET_ID}')
    print(desc['description'])
    print(f'Procedure: {desc["procedure"]}')

    # Step 3: Access the dataset data
    # Data is in format [run_id][time][variable]
    # For the battery the variables are 
    #    0: relativeTime (since beginning of run)
    #    1: current (amps)
    #    2: voltage
    #    3: temperature (°C)
    # so that data[a][b, 3] is the temperature at time index b (relative to the start of the run) for run 1
    print(f'Number of runs: {len(data)}')
    print(f'Analyzing run 4')
    print(f'number of time indices: {len(data[4])}')
    print(f"Details of run 4: {desc['runs'][4]}")

    # Step 4: Plot the dataset
    # TODO

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()
