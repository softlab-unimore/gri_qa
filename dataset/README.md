# How to load the dataset

The `one-table` and `multi-table` directories contain the datasets.  
Each dataset contains the links to the tables used by that specific sample in the columns `pdf_name`, `page_nbr` and `table_nbr`. For each sample, you must remove the `".pdf"` from the `pdf_name` string, and access the corresponding tables on the following path `annotation/[pdf_name]/[page_nbr]_[table_nbr].csv`. 
