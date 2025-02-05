# Table Extractor



This directory contains the code to extract meaningful **financial and non-financial tables** from company reports (pdf files)  



## How to run



### Setup



Go to the root of the directory, create a conda virtual environment and activate it

```
conda create -n env
conda activate env
```

Install the [Tesseract OCR Engine](https://github.com/tesseract-ocr/tesseract), needed to use one [Unstructured](https://github.com/Unstructured-IO/unstructured) for table extraction.  
Then, copy the file `sample_config/.env` to the root of the repository as `.env` and fill the missing values.  
Build and run the docker-compose file. The container hosts the pgvector database containing the embeddings extracted from the pdf files

```
sudo docker compose build
sudo docker compose up
```



### Creation of the document embeddings

#### Create dense embeddings

To store the semantic embeddings in the database, run

```
PYTHONHASHSEED=0 python3 main.py --pdf {PDF_PATH} --embed --use_dense --model_name {MODEL_NAME}
```

where

1. `PYTHONHASHSEED=0` is an environment variable making the `hash` function **deterministic**. `hash` is used to parse document chunks, producing an **unique id** which is used as primary key inside the embedding database. In this way, the system **avoids loading multiple times the same documents** if the above command is run repeatedly;
2. `PDF_PATH` can be either a single pdf file or a directory storing pdf files;
3. `--embed` and `--use_dense` indicate that the system should embed the documents using the model `MODEL_NAME` (taken from Huggingface). By default, `MODEL_NAME="intfloat/multilingual-e5-large-instruct"`.

#### Create sparse embeddings

To store the data for the sparse embedding, run

```
PYTHONHASHSEED=0 python3 main.py --pdf {PDF_PATH} --embed --use_sparse --syn_model_name {SYN_MODEL_NAME}
```

where

1. `PYTHONHASHSEED=0` same as above;
2. `PDF_PATH` can be either a single pdf file or a directory storing pdf files;
3. `--embed` and `--use_sparse` indicate that the system should embed the documents using the model `SYN_MODEL_NAME`. By default, `SYN_MODEL_NAME="tf_idf"`.

#### Query the embeddings

To query the embeddings, run

```
python3 main.py --pdf {PDF_PATH} --query {QUERY_STRING} --use_dense --model_name {MODEL_NAME} --k {TOP_K_RESULTS}
```

This command will return the top-k results obtained from the dense (semantic) query.

To do the same for the sparse (syntactic) embeddings, run

```
python3 main.py --pdf {PDF_PATH} --query {QUERY_STRING} --use_sparse --syn_model_name {MODEL_NAME} --k {TOP_K_RESULTS}
```

#### Query the embeddings with ensemble method

The ensemble method leverages both semantic and syntactic retrieval modes to further improve the system. To use the ensemble, run

```
python3 main.py --pdf {PDF_PATH} --query {QUERY_STRING} --use_ensemble --model_name {MODEL_NAME} --syn_model_name {SYN_MODEL_NAME} --k {TOP_K_RESULTS} --lambda {LAMBDA_VALUE}
```

The additional parameter `--lambda` is a scalar value that controls the importance of syntactic features over semantic ones. The higher the value, the more we give importance to the `SYN_MODEL_NAME` (e.g. tf_idf)

### Extract the tables

Run the previous commands by specifying, as `{PDF_PATH}`, the path to a directory containing several pdfs.  
The script `main.py` will automatically extract the tables from the relevant pages and store them in `table_dataset/`.
