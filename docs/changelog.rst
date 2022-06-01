##########
Change log
##########

5-31-22:
--------
- Added soil moisture at bottom boundary as a parameter in the model configuration
- Fix initial conditions calculation

  - initial soil water potential calculated based on initial soil water content specified in config file
  - initial water potential in roots and xylem calculated assuming hydrostatic conditions

5-30-22:
--------
- Added automated testing
- Fixed bug in PM transpiration
- Updates for name change of optimization package

5-22-22:
--------
- Bug fixes for Feddes water stress function
- Moved root functions to a separate module
- Documentation pages moved from GitHub pages to readthedocs

5-19-22:
--------
- FETCH3 can use the optimization configuration file format for standalone runs, so you can use the same configuration
  file for both optimization runs and standalone runs.
- Updated notebooks for example model output and optimization results

5-18-22:
--------
- Many documentation updates
- Bug fixes
- Updates to conda environment

- Updates to model output files:

  - added H, K, and Capac to the soil, root, and canopy outputs (so these variables are no
    longer only in the ds_all.nc output)
  - added metadata (including units) to output files


5-7-22:
-------
- Performance improvements: Replaced scipy.lingalg.pinv2 with torch.linalg.lstqr for ~75% performance improvement
- Updates to required input parameters (uses DBH, sapwood depth to calculate sapwood area)
- Updates to use new version of optimization package



.. include:: roadmap.rst
