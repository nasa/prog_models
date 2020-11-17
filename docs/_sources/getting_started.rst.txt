Getting Started
===============

The NASA Prognostics Models Package is a python modeling framework focused on defining and building models for prognostics (computation of remaining useful life) of engineering systems, and provides a set of prognostics models for select components developed within this framework, suitable for use in prognostics applications for these components.

The foundation of this package is the class :class:`prog_models.prognostics_model.PrognosticsModel`. This class defines the model interface and provides tools for analysis and simulation. New models must either be a subclass of this model or use the model generator method (:py:meth:`prog_models.prognostics_model.PrognosticsModel.generate_model`)

Installing 
-----------

Use 
----


Extending
----------
There are two methods of creating new models: 1. Implementing a new subclass of one of the base model clases (:class:`prog_models.prognostics_model.PrognosticsModel` or :class:`prog_models.deriv_prog_model.DerivProgModel`) or 2. use the model generator method (:py:meth:`prog_models.prognostics_model.PrognosticsModel.generate_model`). These methods are described more below.

1. Subclass Method
********************
The first method for creating a new prog_model is by creating a new subclass of one of the base model classes. The base model classes are described below:

.. autoclass:: prog_models.prognostics_model.PrognosticsModel
|
.. autoclass:: prog_models.deriv_prog_model.DerivProgModel

To generate a new model create a new class for your model that inherits from one of these two base classes, whichever is more appropriate for your application. Alternatively, you can copy one of the templates :class:`prog_model_template.ProgModelTemplate` or :class:`deriv_model_template.ProgModelTemplate`, replacing the methods with logic defining your specific model.

The analysis and simulation tools defined in :class:`prog_models.prognostics_model.PrognosticsModel` will then work with your new model. 

2. Model Generator Method
*************************
The second way to generate a new model is using the model generator method :py:meth:`prog_models.prognostics_model.PrognosticsModel.generate_model`. Pass a map of the keys for input, state, output, events (optional), and the required transition equations into the method, and it will return a constructed model. See :py:meth:`prog_models.prognostics_model.PrognosticsModel.generate_model` for more detail.