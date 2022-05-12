#######
Roadmap
#######


In progress:
------------
- Examples for exploring model output
- Code refactoring
- Better documentation of model inputs, outputs, and units


To be added soon:
-----------------
- Additional optimization objective function options



Future:
-------
- code performance enhancements
- multi-objective optimization
- plot-scale version of model & optimization
- Improvements to NHL transpiration module



Change log
----------


5-7-22:
-------
- Performance improvements: Replaced scipy.lingalg.pinv2 with torch.linalg.lstqr for ~75% performance improvement
- Updates to required input parameters (uses DBH, sapwood depth to calculate sapwood area)
- Updates to use new version of optimization package