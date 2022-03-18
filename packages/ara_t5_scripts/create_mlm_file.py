from tokenize import Whitespace
from typing import Optional
import pandas as pd
import argparse


DATA_DIR = "/project/rrg-mageed/DataBank/DIA_AR_data_bin"
TGT_FNAME = "mlm_test_data.tsv"

def get_masked_string(line:str, tokenizer: Whitespace, p: Optional[float] = None) -> str:
    """Mask the string using T5 style masking. See example in 
    https://huggingface.co/docs/transformers/model_doc/t5.

    Args:
        line (str): line of text
        p (_type_, optional): percentage of tokens to mask. 

    Returns:
        str: Lines masked with "<extra_id_1>", etc.
    """
    tokens = list(map(lambda x: x[0], tokenizer.pre_tokenize(line)))
    



def main(args):
    src_file = f"{DATA_DIR}/{args.src_file}"
    frame = pd.read_csv(src_file, sep='\t')
    frame = frame[frame["label"] == "MSA"]
    with open(f"{DATA_DIR}/{TGT_FNAME}", 'w') as tgt_tsv:

        tgt_tsv.write(f"\n")







if __name__ = "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_file", metavar='src_file', type=str, nargs=1)
    main(parser.parse_args())

