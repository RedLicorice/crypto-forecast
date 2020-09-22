## Installation
Create conda environment

`conda env create -f environment.yml`

Activate conda environment

`conda activate crypto-forecast`

Sync conda environment with yml (to update changes):

`conda env update --file local.yml`

## Usage
### Preprocess data
Use scripts in `./scripts/preprocessing` for data ingestion, resulting datasets and json indexes are stored in `.data/preprocessed`

### Assemble datasets
Use `python build_datasets.py` to build the merged datasets and index.

Resulting built datasets are stored in `./data/datasets/`

There are three types of datasets:
- Simple: features are not processed, just stuck together with no lag.
- ATSA: features are processed according to ATSA's publication
- Improved: features processed and engineered in order to ease model's interpretation

Datasets are composed of:
- One or more indexes, containing information about file paths and available features
- A set of CSV and Excel files for features, one for each symbol
- A set of CSV and Excel files for target features, one for each symbol

### Build model
Use `python build_model.py -d <dataset> -p <pipeline>` to build models for the dataset index.

`<dataset>` is formatted as `<dataset_name>.<dataset_index>`

Additional parameters are:
- `-e` experiment name, default is `experiment_<date>_<time>`
- `--cv` number of folds to use for cross validation
- `--njobs` number of jobs to dispatch for cross validation, default is cpu_count()
- `--scoring` scoring function to use in cross validation, default is 'precision'
- `--test-size` test size for data split, default is 0.3
- `--use-target` Target to use for classification, default depends on pipeline being used
- `--expanding-window` Whether to use TimeSeriesSplit instead of StratifiedKFold as cross validation provider, default is 0/False

### Benchmark
Use `python bench_models.py` to benchmark pipelines with datasets and produce a report.

Reports are stored in `./benchmarks/<benchmark_name>`

Additional parameters are:
- `-n` benchmark name, default is `benchmark`

## Pipelines
A pipeline is a python file exposing `TARGET`, `PARAMETER_GRID` and `estimator` attributes.

- `TARGET` is the column from the target dataset to use for classification (problem definition)
- `PARAMETER_GRID` is the parameter grid used in grid search
- `estimator` is the estimator whose parameters are to be tuned

The pipeline's estimator can also be a wrapper method such as `RFE` or `SelectFromModel`, in this case
the class `lib.selection_pipeline.SelectionPipeline` should be 
used instead of `sklearn.pipeline.Pipeline` to expose the required attributes.