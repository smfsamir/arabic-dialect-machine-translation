from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# TODO: run this 
from typing import *
def get_vocab(get_line_iterator) -> Set[str]:
    """Get the vocabulary for language/dialect.

    Args:
        dia_fname (str): _description_
        label (_type_): _description_
    
    Returns:
        Vocabulary for the language or dialect.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]"])
    tokenizer.pre_tokenizer = Whitespace()

    iterator = get_line_iterator()
    tokenizer.train_from_iterator(iterator, trainer)
    
    vocab = set([])
    iterator = get_line_iterator()
    for line in iterator:
        tokens = tokenizer.encode(line).tokens
        vocab.update(tokens)
    return vocab