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

```
@inproceedings{contalbo-etal-2025-gri,
    title = "{GRI}-{QA}: a Comprehensive Benchmark for Table Question Answering over Environmental Data",
    author = "Contalbo, Michele Luca  and
      Pederzoli, Sara  and
      Buono, Francesco Del  and
      Valeria, Venturelli  and
      Guerra, Francesco  and
      Paganelli, Matteo",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.814/",
    pages = "15764--15779",
    ISBN = "979-8-89176-256-5",
    abstract = "Assessing corporate environmental sustainability with Table Question Answering systems is challenging due to complex tables, specialized terminology, and the variety of questions they must handle. In this paper, we introduce GRI-QA, a test benchmark designed to evaluate Table QA approaches in the environmental domain. Using GRI standards, we extract and annotate tables from non-financial corporate reports, generating question-answer pairs through a hybrid LLM-human approach. The benchmark includes eight datasets, categorized by the types of operations required, including operations on multiple tables from multiple documents. Our evaluation reveals a significant gap between human and model performance, particularly in multi-step reasoning, highlighting the relevance of the benchmark and the need for further research in domain-specific Table QA. Code and benchmark datasets are available at https://github.com/softlab-unimore/gri{\_}qa."
}
```
