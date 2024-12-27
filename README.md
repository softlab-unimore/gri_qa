# gri_qa

## How to run the experiments

Run one of the `test` files in the main directory. Specify the name of the dataset you want to test, e.g. `gri-qa_extra.csv`. The name must exactly match the file name inside the `dataset` dir

```
python3 [TEST_SCRIPT] --dataset [DATASET_NAME]
```

The script will save the results in `results/[DATASET_NAME]` dir. The results include the CO2 emissions of the model and the csv predictions with/without the boolean matches.

## How to evaluate the experiments

Run the `extract_metrics.py` script. The script will run the exact match metric on the results provided by the models specified in the first few lines of `extract_metrics.py` (inside main). Be sure that the model you want to evaluate is listed. This time, when specifying the dataset, use either `extra`, `quant` or `rel` to evaluate the models on `gri-qa_extra.csv`, `gri-qa_quant.csv`, `gri-qa_rel.csv` respectively.

```
python3 extract_metrics.py --dataset ['extra' | 'quant' | 'rel']
```

The results will be saved inside the file `metrics.csv` in `results/[DATASET_NAME]`.