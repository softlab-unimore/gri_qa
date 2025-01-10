import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pprint import pprint
from transformers import RobertaModel, BertModel
from codecarbon import EmissionsTracker

from tag_op.tagop.util import create_logger, set_environment
from tag_op.data.tatqa_batch_gen import TaTQATestBatchGen
from tag_op.data.data_util import get_op_1, get_op_2, get_arithmetic_op_index_1, get_arithmetic_op_index_2
from tag_op.data.data_util import get_op_3, get_arithmetic_op_index_3
from tag_op.tagop.modeling_tagop import TagopModel
from tag_op.tagop.model import TagopPredictModel


def tagop_predictor(
        test_data_path: str,
        save_dir: str,
        roberta_model: str,
        model_path: str,
) -> dict:
    args = {
        # options.add_data_args(parser)
        'gpu_num': torch.cuda.device_count(),
        'test_data_path': test_data_path,
        'save_dir': save_dir,
        'log_file': 'train.log',
        # options.add_bert_args(parser)
        'bert_learning_rate': None,
        'bert_weight_decay': None,
        'roberta_model': roberta_model,
        # argparse.ArgumentParser
        'batch_size': 32,
        'model_path': model_path,
        'mode': 1,
        'op_mode': 0,
        'ablation_mode': 0,
        'encoder': 'roberta',
    }

    args = pd.Series(args)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    args.cuda = args.gpu_num > 0

    logger = create_logger("TagOp Predictor", log_file=os.path.join(args.save_dir, args.log_file))

    pprint(args)
    set_environment(args.cuda)

    dev_itr = TaTQATestBatchGen(
        test_data_path=args.test_data_path,
        batch_size=args.batch_size,
        cuda=args.cuda,
    )

    if args.encoder == 'roberta':
        bert_model = RobertaModel.from_pretrained(args.roberta_model)
    elif args.encoder == 'bert':
        bert_model = BertModel.from_pretrained('bert-large-uncased')
    else:
        raise ValueError(f"Invalid encoder: {args.encoder}")

    if args.ablation_mode == 0:
        operators = [1 for _ in range(10)]
        arithmetic_op_index = [3, 4, 6, 7, 8, 9]
    elif args.ablation_mode == 1:
        operators = get_op_1(args.op_mode)
    elif args.ablation_mode == 2:
        operators = get_op_2(args.op_mode)
    else:
        operators = get_op_3(args.op_mode)

    if args.ablation_mode == 1:
        arithmetic_op_index = get_arithmetic_op_index_1(args.op_mode)
    elif args.ablation_mode == 2:
        arithmetic_op_index = get_arithmetic_op_index_2(args.op_mode)
    else:
        arithmetic_op_index = get_arithmetic_op_index_3(args.op_mode)

    dataset_name = os.path.basename(args.test_data_path.replace('.pkl', ''))
    carbon_dir = str(os.path.join(args.save_dir, 'codecarbon', dataset_name))
    os.makedirs(carbon_dir, exist_ok=True)
    tracker = EmissionsTracker(output_dir=carbon_dir)
    tracker.start()

    network = TagopModel(
        encoder=bert_model,
        config=bert_model.config,
        bsz=None,
        operator_classes=len(operators),
        scale_classes=5,
        operator_criterion=nn.CrossEntropyLoss(),
        scale_criterion=nn.CrossEntropyLoss(),
        arithmetic_op_index=arithmetic_op_index,
        op_mode=args.op_mode,
        ablation_mode=args.ablation_mode,
    )
    network.load_state_dict(torch.load(os.path.join(args.model_path, "checkpoint_best.pt"), weights_only=True),
                            strict=False)
    model = TagopPredictModel(args, network)
    logger.info("Below are the result on Dev set...")
    model.reset()
    model.avg_reset()
    pred_json = model.predict(dev_itr)

    tracker.stop()

    return pred_json


def dict_to_df(dict_preds: dict) -> pd.DataFrame:
    """
    Convert tagop predictions into a DataFrame with IDX and Value columns.
    The list values are concatenated into a string with proper handling of different data types.

    Args:
        dict_preds (dict): Path

    Returns:
        pandas.DataFrame: DataFrame with IDX and Value columns
    """
    rows = []

    for key, value in dict_preds.items():
        # Handle different types of values and their concatenation
        if isinstance(value, list):
            # Convert all elements to strings and handle nested lists
            processed_values = []

            for item in value:
                if isinstance(item, list):
                    # Join nested list elements with spaces
                    processed_values.append(' '.join(str(x) for x in item))
                else:
                    # Add non-list items directly
                    processed_values.append(str(item))

            # Join all processed values with a space
            concatenated_value = ' '.join(processed_values)

            rows.append({
                'IDX': key,
                'Value': concatenated_value
            })

    return pd.DataFrame(rows)


def dict_to_qa(dict_preds: dict, qa_file: str) -> pd.DataFrame:
    """Merges prediction dictionary with Q&A dataset from CSV file.

    Args:
        dict_preds (dict): Predictions dictionary with indices as keys
        qa_file (str): Path to CSV file with 'question' and 'value' columns

    Returns:
        pd.DataFrame: DataFrame with columns [index, question, value, response]
    """
    # Convert predictions to DataFrame and set index
    df_preds = dict_to_df(dict_preds)
    df_preds['IDX'] = df_preds['IDX'].astype(int)
    assert df_preds['IDX'].is_unique
    df_preds = df_preds.set_index('IDX', drop=True)
    preds = df_preds['Value'].to_dict()

    # Load and process Q&A data
    df_true = pd.read_csv(qa_file)
    df_true['index'] = np.arange(len(df_true))
    df_true = df_true[['index', 'question', 'value']]
    df_true['response'] = df_true['index'].apply(lambda x: preds[x] if x in preds else None)

    return df_true


# #### Checkpoint
# You may download this checkpoint of the trained TagOp model (Save in the same directory as the script):
# [TagOp Checkpoint](https://drive.google.com/file/d/1Ttyh1xyulsGcOt_JmFsAhPuxx7G3fyha/view?usp=share_link)

def main():
    # Step 1: Predict using TagOp
    # ToDo: Use script test_tagop_prepare_dataset.py to prepare the dataset
    dict_preds = tagop_predictor(
        test_data_path='./dataset_tagop/cache/qa_tagop_dataset.pkl',
        save_dir='./results/',
        roberta_model='./dataset_tagop/roberta.large/',  # ToDo: Download the model
        model_path='./tagop_checkpoint/'  # ToDo: Download the checkpoint
    )

    # Step 2: Save predictions
    qa_file = './dataset/one-table/gri-qa_extra.csv'
    df_preds = dict_to_qa(dict_preds, qa_file)
    df_preds.to_csv('./results/one-table/extra/tagop.csv', index=False)

    print('Hello World!')


if __name__ == '__main__':
    main()
