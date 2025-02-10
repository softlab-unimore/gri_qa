## How to run the experiments

Move the directory `../dataset/` into this directory.  
Copy the file `./sample_config/config.ini` into this directory, and fill the empty fields (i.e. your openai key, which is needed to run `test_openapi.py` and `test_openapi_chainofthought.py`).  
Run one of the `test` files in the main directory. Specify the name of the dataset you want to test, e.g. `gri-qa_extra.csv`. The name must exactly match the file name inside the `dataset` dir.

```
python3 {TEST_SCRIPT} --dataset {DATASET_NAME} --type [one-table | multi-table]
```

The script will save the results in `results/{DATASET_NAME}` dir. The results include the CO2 emissions of the model and the csv predictions with/without the boolean matches.

## How to evaluate the experiments

Run the `extract_metrics.py` script. The script will run the exact match metric on the results provided by the models specified in the first few lines of `extract_metrics.py` (inside main, variable `models`). Be sure that the model you want to evaluate is listed. This time, when specifying the dataset, use either `extra`, `quant`, `rel`, `multistep` or `multitable{num_tables}_[rel | quant | multistep]` to evaluate the models on `gri-qa_extra.csv`, `gri-qa_quant.csv`, `gri-qa_rel.csv` or `gri-qa_multitable{num_tables}_[rel | quant | multistep].csv` respectively.

```
python3 extract_metrics.py --dataset [extra | quant | rel | multistep | multitable{num_tables}-quant | multitable{num_tables}-rel | multitable{num_tables}-multistep] --type [one-table | multi-table]
```

The results will be saved inside the file `metrics.csv` in `results/[DATASET_NAME]`.

## Models tested

- [TheFinAI/finma-7b-full](https://huggingface.co/TheFinAI/finma-7b-full)  
- [next-tat/tat-llm-7b-fft](https://huggingface.co/next-tat/tat-llm-7b-fft)  
- [osunlp/TableLlama](https://huggingface.co/osunlp/TableLlama)  
- [neulab/omnitab-large](https://huggingface.co/neulab/omnitab-large)  
- [microsoft/tapex-large](https://huggingface.co/microsoft/tapex-large)  
- [gpt-4o-mini](https://platform.openai.com/docs/models)
