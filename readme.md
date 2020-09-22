## Pipelines
A pipeline is a python file exporting a "PARAMETER_GRID" as well as an "estimator" object.

GridSearchCV is executed using parameters from the grid to find the best performing model, according
to the specified scoring function.

The pipeline's estimator can also be a wrapper method such as RFE or SelectFromModel, in this case
the class lib.selection_pipeline.SelectionPipeline should be 
used instead of sklearn.pipeline.Pipeline