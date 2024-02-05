============
Introduction
============
 
The purpose of this module is to implement a set of useful utilities for dealing with coherent objects in LES data (in particular from MONC). 

===============
Version History
===============

Latest version is 0.3.0

.. topic:: New at 0.3.0

	#. Use of scikit-image library tools for object labelling and bounds.
	#. Greatly speeded up code to find and merge objects spanning the domain boundary.
	#. Hence overall speedup of label_3D_cyclic and so get_object_labels of about 2 orders of magnitude.
	#. Use of loguru.logger.

.. topic:: New at 0.2

    #. Complete re-structuring. Extracted from Subfilter repository.


