# Prognostics Model Python Package
[![CodeFactor](https://www.codefactor.io/repository/github/nasa/prog_models/badge)](https://www.codefactor.io/repository/github/nasa/prog_models)
[![GitHub License](https://img.shields.io/badge/License-NOSA-green)](https://github.com/nasa/prog_models/blob/master/license.pdf)
[![GitHub Releases](https://img.shields.io/github/release/nasa/prog_models.svg)](https://github.com/nasa/prog_models/releases)

The NASA Prognostic Model Package is a Python framework focused on defining and building models for prognostics (computation of remaining useful life) of engineering systems, and provides a set of prognostics models for select components developed within this framework, suitable for use in prognostics applications for these components.

This is designed to be used with the [Prognostics Algorithms Package](https://github.com/nasa/prog_algs).

## Installation 
`pip3 install prog_models`

## [Documentation](https://nasa.github.io/prog_models/)
See documentation [here](https://nasa.github.io/prog_models/)
 
## Repository Directory Structure 
Here is the directory structure for the github repository 
 
`src/prog_models/` - The prognostics model python package<br />
&nbsp;&nbsp; |-`models/` - Example models<br /> 
&nbsp;&nbsp; |-`prognostics_model.py` - Physics-based model superclass of degraded system behavior<br />
&nbsp;&nbsp; |-`sim_result.py` - Class for storing the result of a simulation (used by `prognostics_model`)<br />
&nbsp;&nbsp; |-`visualize.py` - Visualization tools<br />
`docs/` - Project documentation<br />
`sphinx_config/` - Configuration for automatic documentation generation<br />
`examples/` - Example Python scripts using prog_models<br />
`tests/` - Tests for prog_models<br />
`README.md` - The readme (this file)<br />
`requirements.txt` - Python library dependiencies required to be met to use this package<br />
`prog_model_template.py` - Template for Prognostics Model<br />
`tutorial.ipynb` - Tutorial (Juypter Notebook)

## Citing this repository
Use the following to cite this repository:

```
@misc{2021_nasa_prog_models,
    author    = {Christopher Teubert and Matteo Corbetta and Chetan Kulkarni and Matthew Daigle},
    title     = {Prognostics Models Python Package},
    month     = Aug,
    year      = 2021,
    version   = {1.1},
    url       = {https://github.com/nasa/prog_models}
    }
```

The corresponding reference should look like this:

C. Teubert, M. Corbetta, C. Kulkarni, and M. Daigle, Prognostics Models Python Package, v1.1, Aug. 2021. URL https://github.com/nasa/prog_models.

## Acknowledgements
The structure and algorithms of this package are strongly inspired by the [MATLAB Prognostics Model Library](https://github.com/nasa/PrognosticsModelLibrary). We would like to recognize Matthew Daigle and the rest of the team that contributed to the Prognostics Model Library for the contributions their work on the MATLAB library made to the design of prog_models

## Notices

Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

## Disclaimers

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
