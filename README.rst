BOLOS
=====

``BOLOS`` is a BOLtzmann equation solver Open Source library.  

This package provides a pure Python library for the solution of the 
Boltzmann equation for electrons in a non-thermal plasma.  It builds upon
previous work, mostly by G. J. M. Hagelaar and L. C. Pitchford [HP2005]_, 
who developed `Bolsig+`_.  ``BOLOS`` is a multiplatform, open source 
implementation of a similar algorithm compatible with the `Bolsig+`_ 
cross-section input format.


The code was originally developed by `Alejandro Luque <http://www.iaa.es/~aluque>`_ at the 
`Instituto de Astrofísica de Andalucía <http://www.iaa.es>`_ (IAA), `CSIC <http://www.csic.es>`_, 
later updated by `Bang-Shiuh Chen <https://orcid.org/0000-0003-0437-3659>`_. 
This version is maintained by `Alejandro Roa <https://orcid.org/0009-0005-7243-9834>`_ and the 
`Nonequilibrium Plasma Team <https://em2c.centralesupelec.fr/Axe_Plasma>`_ at Laboratoire EM2C, CNRS, CentraleSupélec.
The current version adds new features compared to the original code, including the 
ability to add Coulomb collisions while keeping speeds relatively low by using 
`Numba <https://numba.pydata.org/>`_ to accelerate the code. 
The code is still under development, and new features will be added in the future.
The code is licensed under the `LGPLv2 License`_. 
The `source code`_ can be obtained from
GitHub, which also hosts the `bug tracker`_.


.. _LGPLv2 License: http://www.gnu.org/licenses/lgpl-2.0.html
.. _BOLSIG+: http://www.bolsig.laplace.univ-tlse.fr/
.. _homepage: http://pypi.python.org/pypi/bolos/
.. _source code: https://github.com/aleroaq/bolos
.. _bug tracker: https://github.com/aleroaq/bolos/issues
.. [HP2005] *Solving the Boltzmann equation to obtain electron transport coefficients and rate coefficients for fluid models*, G. J. M. Hagelaar and L. C. Pitchford, Plasma Sources Sci. Technol. **14** (2005) 722–733.



