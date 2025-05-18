# GRI-QA: a Comprehensive Benchmark for Table Question Answering over Environmental Data

**GRI-QA** is a benchmark for Single- and Multi-Table Question Answering over environmental data available at [dataset](./dataset/) and in the following 🤗 [repository](https://huggingface.co/datasets/lucacontalbo/GRI-QA). The benchmark is composed by several question types:  
- **extractive** questions, divided in the datasets  
  - *extra*: questions that require the identification of relevant span(s) in a table;  
  - *hier*: same as *extra*, but on hierarchical rows;  
- **calculated** questions, split in  
  - datasets to test reasoning on single tables:  
    - *rel*: requires the identification of relations between cells;  
    - *quant*: requires the computation of quantitative results;  
    - *step*: questions combining the operations of the *rel* and *quant*;  
  - and datasets to test reasoning on multiple tables (2,3 or 5 tables):  
    - *mrel*  
    - *mquant*  
    - *mstep*  

For more information, please refer to our paper.

# Repository organization

- `dataset/` holds the whole GRI-QA benchmark, as well as the sampled datasets used for the human baseline;
- `table_extraction/` holds the code to extract, given a GRI (Global Reporting Initiative) description, the relevant tables from corporate reports (pdf files);
- `test/` holds the code to reproduce the results in Section 4 of the paper.

# License

GRI-QA is under the [MIT license](./LICENSE)

# Citation

Paper to be released soon at ACL 2025
