import argparse
from transformers import AutoModelForSeq2SeqLM 

def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/fsamir8/projects/rrg-mageed/DataBank/MSA_Dia_Style_Tansfer/araT5_binary_msa-da-SUPER-ARLU"
    )
    print(model)


def main(args):
    if args.load_model:
        load_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', action='store_true')



