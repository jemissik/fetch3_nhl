.. FETCH3 documentation master file, created by
   sphinx-quickstart on Tue Feb 15 12:21:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FETCH3's documentation!
==================================

********
Overview
********

FETCH3 is a tree hydrodynamic model for water fluxes across the soil-plant-atmosphere continuum. FETCH3 simulates water
transport through the soil, roots, and xylem as flow through porous media. The model resolves water potentials along the
vertical dimension, and stomatal response is linked to xylem water potential.


There are 2 different transpiration schemes available:

- **Penman-Monteith transpiration(PM)** - Simple transpiration scheme used in *Silva et al, 2022*. Transpiration is
  calculated using the Penman-Monteith equation and disctributed through the canopy based on the leaf area density.
  Stomatal conductance is adjusted using Jarvis functions for incoming shortwave radiation, air temperature, vapor
  pressure deficit, and xylem water potential.
- **Non-hydraulically limited transpiration (NHL)** - Replicates the more complex canopy transpiration scheme that was
  used in FETCH2 (*Mirfenderesgi et al, 2016*). Non-hydraulically limited transpiration is calculated considering
  stomatal conductance as a function of atmospheric demand and photosynthetic capacity, but without any limitations
  imposed by soil moisture. The non-hydraulically limited transpiration is then reduced using a function of xylem water
  potential. This scheme considers the different aerodynamic characteristics at different layers in the canopy, and
  radiation attenuation through the canopy.

To start using the model, see the :ref:`Getting Started` page!

**********
References
**********

**FETCH3:**

Silva, M., Matheny, A. M., Pauwels, V. R. N., Triadis, D., Missik, J. E.,
Bohrer, G., and Daly, E.: Tree hydrodynamic modelling of the soil–plant–atmosphere
continuum using FETCH3, Geosci. Model Dev., 15, 2619–2634, https://doi.org/10.5194/gmd-15-2619-2022, 2022.

**FETCH2:**

Mirfenderesgi, G., Bohrer, G., Matheny, A. M., Fatichi, S., Frasson, R. P. de M.,
& Schäfer, K. V. R. (2016). Tree level hydrodynamic approach for resolving aboveground
water storage and stomatal conductance and modeling the effects of tree hydraulic strategy.
Journal of Geophysical Research: Biogeosciences, 121(7), 1792–1813. https://doi.org/10.1002/2016JG003467

Mirfenderesgi, G., Matheny, A. M., & Bohrer, G. (2019). Hydrodynamic Trait Coordination and Cost-Benefit
Trade-offs in Trees. Ecohydrology, 12(1), e2041. https://doi.org/10.1002/eco.2041


**FETCH:**

Bohrer, G., Mourad, H., Laursen, T. A., Drewry, D., Avissar, R., Poggi, D., et al. (2005).
Finite element tree crown hydrodynamics model (FETCH) using porous media flow within branching
elements: A new representation of tree hydrodynamics. Water Resources Research, 41(11).
https://doi.org/10.1029/2005WR004181


********
Contents
********

.. toctree::
   :maxdepth: 2

   getting_started
   user_guide/index
   scaling
   roadmap
   developer_guide
   FETCH3 code reference <ftch/fetch3>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
