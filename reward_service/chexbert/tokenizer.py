"""CheXbert tokenization utilities.

Inlined from CheXbert/src/bert_tokenizer.py — accepts lists of strings
directly instead of CSV paths.
"""

from typing import List

import pandas as pd
from transformers import BertTokenizer


def tokenize_texts(texts: List[str], tokenizer: BertTokenizer) -> List[List[int]]:
    """Tokenize a list of report strings using a BERT tokenizer.

    Args:
        texts: Raw report strings.
        tokenizer: A pre-loaded BertTokenizer.

    Returns:
        List of token-id lists, each capped at 512 tokens.
    """
    # Clean up whitespace (mirrors get_impressions_from_csv)
    imp = pd.Series(texts)
    imp = imp.str.strip()
    imp = imp.replace("\n", " ", regex=True)
    imp = imp.replace(r"\s+", " ", regex=True)
    imp = imp.str.strip()

    encoded = []
    for i in range(len(imp)):
        text = str(imp.iloc[i])
        tokenized = tokenizer.tokenize(text)
        if tokenized:
            res = tokenizer.encode(
                tokenized,
                add_special_tokens=True,
                is_split_into_words=True,
            )
            if len(res) > 512:
                res = res[:511] + [tokenizer.sep_token_id]
            encoded.append(res)
        else:
            encoded.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
    return encoded
