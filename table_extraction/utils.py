import argparse
import logging

def check_args(args):
    if len(args["load_query_from_file"]) > 0 and args["load_query_from_file"].split(".")[-1] != "json":
        raise ValueError(f"argument \"load_query_from_file\" should either be empty or a json file")
    if sum([args["use_dense"], args["use_sparse"], args["use_ensemble"]]) != 1:
        raise ValueError(f"only one argument between \"use_dense\", \"use_sparse\" or \"use_ensemble\" must be specified")
    if args["query"] != '' and args["embed"]:
        raise ValueError(f"you can't embed and query at the same time. Please specify only one argument between \"--embed\" and \"--query\"")
    if args["use_ensemble"] and args["embed"]:
        raise ValueError(f"cannot embed using the ensemble model. The ensemble model can only be used for querying, after the vector and sparse stores are loaded into the db")

def init_args():
    parser = argparse.ArgumentParser(prog='BPER Table Extractor', description='Extract tables from companies non financial statements')

    parser.add_argument('-m', '--method', choices=["page"], default="page", type=str,
                        help='extraction method')
    parser.add_argument('-p', '--pdf', type=str, required=True, default='',
                        help='relative URI of the pdf file to analyze')
    parser.add_argument('-q', '--query', type=str, required=False, default='',
                        help=' query to be sent to the vector store')
    parser.add_argument('-e', '--embed', action="store_true", default=False, 
                        help='embed documents')
    parser.add_argument('-d', '--use_dense', action="store_true", default=False, 
                        help='use vector store')
    parser.add_argument('-s', '--use_sparse', action="store_true", default=False, 
                        help='use sparse store')
    parser.add_argument('-E', '--use_ensemble', action="store_true", default=False, 
                        help='use ensemble method')
    parser.add_argument('-M', '--model_name', type=str, required=False, default="sentence-transformers/all-mpnet-base-v2", 
                        help='name of the neural model to use')
    parser.add_argument('-S', '--syn_model_name', type=str, required=False, default="tf_idf", 
                        help='name of the sparse model to use')
    parser.add_argument('-L', '--lambda', type=float, required=False, default=0.3, 
                        help='integration scalar for syntactic features. Only usable with --use_ensemble')
    parser.add_argument('-f', '--load_query_from_file', type=str, required=False, default="", 
                        help='path of the GRI/Description file')

    args = vars(parser.parse_args())
    check_args(args)

    return args