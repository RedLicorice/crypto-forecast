## Pipelines
A pipeline is a python file exporting a "PARAMETER_GRID" as well as an "estimator" object.

A GridSearchCV is executed using parameters from the grid to find the best performing model, according
to the specified scoring function.