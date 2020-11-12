# Prognostics Model Library Software Plan

## Software Goals
- Mirror Prognostics Model Matlab Library
- Ability to simulate prognostics components to failure

## Development Plan
- Oct-Nov: First Implementation
- Dec-Jan: Testing and verification 

## Release & Maintenance  Plan
Release open source on Github.com, after which use github ticketing system for bug-tracking and maintenance. 

## Software Assurance
* Branching strategy: master/trunk: Main branch for software that has undergone a release and has a release # associated. pull requests into master require release review (see release checklist)
  * dev: working code that has undergone a code review
  * feature/bugfix branches: working branches for in-progress work
* Testing- release requires sufficent tests- part of review

### Release Checklist
* Target branch: master, source: dev
* Includes adequate tests
* Includes adequate documentation
* Includes examples
* requirements.txt up to date
