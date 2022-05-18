##########
Change log
##########


5-7-22:
-------
- Performance improvements: Replaced scipy.lingalg.pinv2 with torch.linalg.lstqr for ~75% performance improvement
- Updates to required input parameters (uses DBH, sapwood depth to calculate sapwood area)
- Updates to use new version of optimization package



.. include:: roadmap.rst
