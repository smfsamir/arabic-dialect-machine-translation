#!/usr/bin/env python
# coding=utf-8

"""
This code is based on Huggingface seq2seq example
***************************************************************
Updated by AbdelRahim Elmadany
University of British Columbia, Vancouver Campus, BC, Canada
NLP Lab, April, 2021
***************************************************************
Updates:
--------
   - add early stopping
   - evaluate based on bleu (for MT tasks), and rouge (for summarization tasks), and macro F1 score (for other NLP tasks)
   - save the best epoch based on the evaluation score
   - Tested on mT5, Arabic T5 (Twitter, MSA, and Twitter plus MSA) models
"""
import codecs
from rouge_score import rouge_scorer, scoring
from eval_squad import *
import regex
import shutil, glob
import sacrebleu
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix

import transformers
from filelock import FileLock
from transformers import (
    IntervalStrategy,
    EarlyStoppingCallback,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed, MBart50Tokenizer
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
# cache_dir="/home/elmadany/T5/cache_dir" #AMD
# cache_dir="/project/6007993/elmadany/T5/cache_dir" #Cedar
# cache_dir="/scratch/st-amuham01-1/elmadany/T5/cache_dir" #Sockeye
# cache_dir="/scratch/st-amuham01-1/elmadany/T5/moatez_cache"
cache_dir="/home/fsamir8/scratch/ara_dial_translation"
#with FileLock(".lock") as lock:
#    nltk.download("punkt", quiet=True)


logger = logging.getLogger(__name__)



def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "a") as f:
        json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(default = "google/mt5-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task: str = field(
        default="translation",
        metadata={
            "help": "The name of the task, should be summarization (or summarization_{dataset} for evaluating "
            "pegasus) or translation (or translation_{xx}_to_{yy})."
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default="en",
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default="hi-Devanagari",
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default="/home/ganeshjw/projects/rrg-mageed/ganeshjw/projects/MT5/mt5_train.json", metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default="/home/ganeshjw/projects/rrg-mageed/ganeshjw/projects/MT5/mt5_dev.json",
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge/sacreblue) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge/sacreblue) on "
            "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    source_lang: Optional[str] = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: Optional[str] = field(default=None, metadata={"help": "Target language id for translation."})
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["tsv", "csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["tsv","csv", "json"], "`validation_file` should be a csv or a json file."
#         if not self.task.startswith("summarization") and not self.task.startswith("translation"):
#             raise ValueError(
#                 "`task` should be summarization, summarization_{dataset}, translation or translation_{xx}_to_{yy}."
#             )
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def main():
    early_stopping_num=20
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    #automatic checking last checkpoint
    ckpt_num=0
    for ck in glob.glob(training_args.output_dir+"/checkpoint-*"):
        current_ckpt_num=int(ck.split("/")[-1].split("-")[-1])
        if current_ckpt_num>ckpt_num:
            ckpt_num=current_ckpt_num
    if ckpt_num>0:
        last_checkpoint=training_args.output_dir+"/checkpoint-"+str(ckpt_num)
    print ("last_checkpoint", last_checkpoint)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
    # second column for the summaries (unless you specify column names for this with the `text_column` and
    # `summary_column` arguments).
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        if "tsv" in extension:
            print ("[INFO] loading from TSV")
            datasets = load_dataset("csv", delimiter="\t", data_files=data_files, cache_dir=cache_dir)    
        else:
            datasets = load_dataset(extension, data_files=data_files, cache_dir=cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        local_files_only = True
    )
    if "mbart-large-50" not in model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            local_files_only = True
        )
    else:
        tokenizer = MBart50Tokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            local_files_only = True
        )
        tokenizer.src_lang = data_args.source_lang
        tokenizer.tgt_lang = data_args.target_lang
        print (">>>>>>>data_args.source_lang", data_args.source_lang)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        local_files_only = True
    )

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        assert (
            data_args.target_lang is not None and data_args.source_lang is not None
        ), "mBart requires --target_lang and --source_lang"
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
       
    # To serialize preprocess_function below, each of those four variables needs to be defined (even if we won't use
    # them all).
    source_lang, target_lang, text_column, summary_column = None, None, None, None

    if data_args.task.startswith("summarize"):
        # Get the column names for input/target.
        dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
        if data_args.text_column is None:
            text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            text_column = data_args.text_column
        if data_args.summary_column is None:
            summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            summary_column = data_args.summary_column
    # else:
    #     # Get the language codes for input/target.
    #     lang_search = re.match("translation_([a-z]+)_to_([a-z]+)", data_args.task)
    #     if data_args.source_lang is not None:
    #         source_lang = data_args.source_lang.split("_")[0]
    #     else:
    #         assert (
    #             lang_search is not None
    #         ), "Provide a source language via --source_lang or rename your task 'translation_xx_to_yy'."
    #         source_lang = lang_search.groups()[0]

    #     if data_args.target_lang is not None:
    #         target_lang = data_args.target_lang.split("_")[0]
    #     else:
    #         assert (
    #             lang_search is not None
    #         ), "Provide a target language via --target_lang or rename your task 'translation_xx_to_yy'."
    #         target_lang = lang_search.groups()[1]

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warn(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
#         if data_args.task.startswith("translation"):
            #inputs = [ex[source_lang] for ex in examples["translation"]]
            #targets = [ex[target_lang] for ex in examples["translation"]]
        inputs = examples[data_args.text_column]
        targets = examples[data_args.summary_column]
#         else:
#             inputs = examples[text_column]
#             targets = examples[summary_column]
        inputs = [str(prefix) + str(inp) for inp in inputs]
        targets = [str(tgt) for tgt in targets]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        train_dataset = datasets["train"]
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    # if "summarize" in data_args.task:
    #     metric_name = "rouge"
    # elif "translate" in data_args.task:
    #     metric_name = "bleu"
    # else:
    if training_args.metric_for_best_model is not None:
        metric_name = regex.sub("eval_","",training_args.metric_for_best_model)
    else:
        metric_name ='f1'
#     metric_name = "f1" #"rouge" if data_args.task.startswith("summarization") else "sacrebleu"
    print ("[INFO] evlaute using ", metric_name, "score", "task name:", data_args.task)
    # metric = load_metric("f1", cache_dir=cache_dir)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        # if metric_name == "rouge":
        #     preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        #     labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        # el
        if metric_name == "QA_f1":
            labels = [label.split('*****') for label in labels]
#         elif metric_name == "bleu":  # sacrebleu
#             labels = [[label] for label in labels]
        # else:
        #     labels = [label for label in labels]
        elif metric_name == "bleucased":
            labels = [label.lower() for label in labels]
            preds = [pred.lower() for pred in preds]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        if metric_name == "QA_f1":
            print (decoded_labels)
            print (decoded_preds)
            res = evaluate_squad(decoded_labels, decoded_preds)
            result =  {
                'QA_Exact_match': res["exact_match"],
                'QA_f1': res["f1"]#,
                #'Exact_sentence': res["exact_sentence"]
                }
        elif "rouge" in metric_name:
            use_agregator= True
            rouge_types = ["rouge1", "rouge2", "rougeL"]#, "rougeLsum"]
            scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
            if use_agregator:
                aggregator = scoring.BootstrapAggregator()
            else:
                scores = []

            for ref, pred in zip(decoded_labels, decoded_preds):
                score = scorer.score(ref, pred)
                if use_agregator:
                    aggregator.add_scores(score)
                else:
                    scores.append(score)

            if use_agregator:
                result = aggregator.aggregate()
            else:
                result = {}
                for key in scores[0]:
                    result[key] = list(score[key] for score in scores)
            
            # result = []#metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            # Extract a few results from ROUGE
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        elif metric_name == "bleu":
#             result = metric.compute(predictions=decoded_preds, references=[decoded_labels])
            bleu_results = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
            bleu_score = bleu_results.score
            result = {"bleu": bleu_score}
        elif metric_name == "bleucased":
#             result = metric.compute(predictions=decoded_preds, references=[decoded_labels])
            bleu_results = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
            bleu_score = bleu_results.score
            result = {"bleucased": bleu_score}
        else:
            accuracy = accuracy_score(decoded_labels, decoded_preds)
            f1 = f1_score(decoded_labels, decoded_preds, average='macro') 
            recall = recall_score(decoded_labels, decoded_preds, average='macro')
            precision = precision_score(decoded_labels, decoded_preds, average='macro')
            result =  {
                'accuracy': accuracy,
                'f1_macro': f1,
                'precision': precision,
                'recall': recall
                }
            #--------- compute FNP for sentiment only ---------------------------
            if "ArBench_Senti" in data_args.task:
                report = classification_report(decoded_labels, decoded_preds)
                print (report)
                F1PN_n="neg" #Negative class
                F1PN_p="pos" #Positive class
                F1PN_u="neut" #Nutral class
                F_NP=0.0
                AvgRec=0.0
                for line in regex.split("[\r\n\f]+", report):
                    line_info=regex.split("\s+", line)
                    if len(line_info) <5:
                        continue
                    report_label=line_info[1]
                    report_P=line_info[2]
                    report_R=line_info[3]
                    report_F1=line_info[4]
                    if str(F1PN_n) in report_label:
                        print ("*** Negative class P,R,F1",report_label,report_P, report_R, report_F1)
                        rN=float(report_R)
                        Nf1=float(report_F1)
                    elif str(F1PN_p) in report_label:
                        print ("*** Positive class P,R,F1",report_label,report_P, report_R, report_F1)
                        rP=float(report_R)
                        Pf1=float(report_F1)
                    elif str(F1PN_u) in report_label:
                        print ("*** EVAL Neutral class P,R,F1",report_label,report_P, report_R, report_F1)
                        rU=float(report_R)
                        Uf1=float(report_F1)
                if str(F1PN_u)!="":
                    AvgRec=(rN+rP+rU)/3 #compute AvgRecall
                F_NP=(Nf1+Pf1)/2 #compute F1PN
                result['sentiment_F_NP']=F_NP
                result['sentiment_AvgRec']=AvgRec
            #------------------------------------------------------
            # report = classification_report(all_label, all_pred)
            
        

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )
    print ("[INFO] early_stopping_num=", early_stopping_num)
    # trainer.add_callback(EarlyStoppingCallback(early_stopping_num)) #number of patient epochs before early stopping
    all_metrics = {}
    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
            print ("***** checkpoint= resume_from_checkpoint", checkpoint)
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
            print ("***** checkpoint= model_name_or_path", checkpoint)
        else:
            checkpoint = None
            print ("***** checkpoint=", checkpoint)

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        if trainer.is_world_process_zero():
            metrics_formatted = trainer.metrics_format(metrics)
            logger.info("***** train metrics *****")
            k_width = max(len(str(x)) for x in metrics_formatted.keys())
            v_width = max(len(str(x)) for x in metrics_formatted.values())
            for key in sorted(metrics_formatted.keys()):
                logger.info(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")
            save_json(metrics, os.path.join(training_args.output_dir, "train_results.json"))
            all_metrics.update(metrics)

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
        )
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        if trainer.is_world_process_zero():
            metrics_formatted = trainer.metrics_format(metrics)
            logger.info("***** val metrics *****")
            k_width = max(len(str(x)) for x in metrics_formatted.keys())
            v_width = max(len(str(x)) for x in metrics_formatted.values())
            for key in sorted(metrics_formatted.keys()):
                logger.info(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")
            save_json(metrics, os.path.join(training_args.output_dir, "eval_results.json"))
            all_metrics.update(metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = test_results.metrics
        max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(test_dataset)
        metrics["test_samples"] = min(max_test_samples, len(test_dataset))
        metrics["ckpt_name"] = model_args.model_name_or_path.split("/")[-1]
        output_init = data_args.test_file.split("/")[-1]+"_"+metrics["ckpt_name"]
        if trainer.is_world_process_zero():
            metrics_formatted = trainer.metrics_format(metrics)
            logger.info("***** test metrics *****")
            k_width = max(len(str(x)) for x in metrics_formatted.keys())
            v_width = max(len(str(x)) for x in metrics_formatted.values())
            for key in sorted(metrics_formatted.keys()):
                logger.info(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")
            save_json(metrics, os.path.join(training_args.output_dir, data_args.test_file.split("/")[-1]+"_test_results.json"))
            all_metrics.update(metrics)

            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                
                test_preds = [pred.strip().encode("utf-8") for pred in test_preds]
                print (test_preds[0])
                print (test_preds[1])
                output_test_preds_file = os.path.join(training_args.output_dir, output_init+"_test_preds.txt")
                with open(output_test_preds_file, "w", encoding='utf8') as writer:
                    for pred in test_preds:
                        writer.write(str(codecs.decode(pred)) +'\n')

    if trainer.is_world_process_zero():
        save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))
    #----------delete eval checkpoints
    # for ck in glob.glob(training_args.output_dir+"/checkpoint-*"):
    #     shutil.rmtree(ck)
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()









