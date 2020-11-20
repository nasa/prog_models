Developers Guide
================

This document includes some details relevant for developers. 

..  contents:: 
    :backlinks: top

Branching Strategy
------------------
Our project is following the git strategy described `here <https://nvie.com/posts/a-successful-git-branching-model/>`_. Release branches are not required. Details specific to each branch are described below. 

`master`: Every merge into the master branch is done using a pull request (never commiting directly), is assigned a release number, and must comply with the release checklist. The release checklist is a software assurance tool. 

`dev`: Every commit on the dev branch should be functional. All unit tests must function before commiting to dev or merging another branch. 

`Feature Branches`: These branches include changes specific to a new feature. Before merging into dev unit tests should all run, tests should be added for the feature, and documentation should be updated, as appropriate.

Release Checklist
*****************
* Code review - all software must be checked by someone other than the author
* Check that each new feature has a corresponding tests
* Run unit tests `python -m tests`
* Check documents- see if any updates are required
* Rebuild sphinx documents: `sphinx-build sphinx-config/ docs/`
* Write release notes
* For releases adding new features- ensure that NASA release process has been followed

NPR 7150
--------
* Software Classification: Class-E (Research Software)
* Safety Criticality: Not Safety Critical 

Compliance Notation Legend
**************************
* FC: Fully Compliant
* T: Tailored (Specific tailoring described in mitigation) `SWE-121 <https://swehb.nasa.gov/display/7150/SWE-121+-+Document+Alternate+Requirements>`_
* PC: Partially Compliant
* NC: Not Compliant
* NA: Not Applicable

Compliance Matrix
*****************
+-------+----------------------------------+------------+---------------------+
| SWE # | Description                      | Compliance | Evidence            |
+=======+==================================+============+=====================+
| 033   | Assess aquisiton Options         | FC         | See section below   |
+-------+----------------------------------+------------+---------------------+
| 013   | Maintain Software Plans          | FC         | This document       |
+-------+----------------------------------+------------+---------------------+
| 042   | Electronic Accesss to Source     | FC         | This repo           |
+-------+----------------------------------+------------+---------------------+
| 139   | Comply with 7150                 | FC         | This document       |
+-------+----------------------------------+------------+---------------------+
| 121   | Tailored Reqs                    | NA         | No tailoring        |
+-------+----------------------------------+------------+---------------------+
| 125   | Compliance Matrix                | FC         | This document       |
+-------+----------------------------------+------------+---------------------+
| 029   | Software Classification          | FC         | This document       |
+-------+----------------------------------+------------+---------------------+
| 022   | Software Assurance               | FC         | This document       |
+-------+----------------------------------+------------+---------------------+
| 205   | Safety Cricial Software          | FC         | See above           |
+-------+----------------------------------+------------+---------------------+
| 023   | Safety Critical Reqs             | NA         | Not safety critical |
+-------+----------------------------------+------------+---------------------+
| 206   | Autogen Software                 | NA         | No autogen          |
+-------+----------------------------------+------------+---------------------+
| 148   | Software Catolog                 | FC         | Will be added       |
+-------+----------------------------------+------------+---------------------+
| 156   | Perform CyberSecurity Assessment | FC         | See section below   |
+-------+----------------------------------+------------+---------------------+

Aquisition Options
******************
Assessed, there are some existing prognostics tools, but no general python package that can support model-based prognostics like we need. 

Cybersecurity Assessment 
************************
Assessed, no significant Cybersecurity concerns were identified- research software. 