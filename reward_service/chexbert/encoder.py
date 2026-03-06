"""CheXbert encoder — inlined from CheXbert/src/.

Provides:
- bert_encoder nn.Module (from models/bert_encoder.py)
- InMemoryDataset (replaces unlabeled_dataset.py)
- collate_fn_no_labels (from encode.py)
- encode_texts() — high-level function that returns CLS embeddings
"""

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

from reward_service.chexbert.tokenizer import tokenize_texts

# --- Constants (from constants.py) ---
PAD_IDX = 0
BATCH_SIZE = 18
CONDITIONS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding",
]


# --- Model (from models/bert_encoder.py) ---
class bert_encoder(nn.Module):
    def __init__(self, logits=False, p=0.1, clinical=False,
                 freeze_embeddings=False, pretrain_path=None,
                 bert_model_path=None):
        super().__init__()
        if pretrain_path is not None:
            self.bert = BertModel.from_pretrained(pretrain_path)
        elif bert_model_path is not None:
            self.bert = BertModel.from_pretrained(bert_model_path)
        else:
            raise ValueError("bert_model_path is required (local absolute path)")

        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
        self.logits = logits
        self.dropout = nn.Dropout(p)
        hidden_size = self.bert.pooler.dense.in_features
        self.linear_heads = nn.ModuleList(
            [nn.Linear(hidden_size, 4, bias=True) for _ in range(13)]
        )
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

    def forward(self, source_padded, attention_mask):
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        out = cls_hidden
        if self.logits:
            out = []
            for i in range(14):
                out.append(self.linear_heads[i](cls_hidden))
        return out


# --- Dataset (replaces unlabeled_dataset.py) ---
class InMemoryDataset(Dataset):
    """Dataset that takes pre-tokenized token-id lists."""

    def __init__(self, encoded_impressions: List[List[int]]):
        self.encoded_imp = encoded_impressions

    def __len__(self):
        return len(self.encoded_imp)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imp = torch.LongTensor(self.encoded_imp[idx])
        return {"imp": imp, "len": imp.shape[0], "idx": idx}


# --- Collate (from encode.py) ---
def collate_fn_no_labels(sample_list):
    tensor_list = [s["imp"] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(
        tensor_list, batch_first=True, padding_value=PAD_IDX
    )
    len_list = [s["len"] for s in sample_list]
    idx_list = [s["idx"] for s in sample_list]
    return {"imp": batched_imp, "len": len_list, "idx": idx_list}


# --- Attention masks (from utils.py) ---
def generate_attention_masks(batch, source_lengths, device):
    masks = torch.ones(batch.size(0), batch.size(1), dtype=torch.float)
    for idx, src_len in enumerate(source_lengths):
        masks[idx, src_len:] = 0
    return masks.to(device)


# --- High-level encode function ---
class CheXbertEncoder:
    """Pre-loaded CheXbert model that produces CLS embeddings from text."""

    def __init__(self, checkpoint_path: str, bert_model_path: str, device: torch.device):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)

        model = bert_encoder(logits=False, bert_model_path=bert_model_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle DataParallel state dict (keys prefixed with "module.")
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)

        model = model.to(device)
        model.eval()
        self.model = model

    @torch.no_grad()
    def encode(self, texts: List[str]) -> Dict[int, torch.Tensor]:
        """Encode texts and return {index: cls_embedding} dict."""
        encoded = tokenize_texts(texts, self.tokenizer)
        dataset = InMemoryDataset(encoded)
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, collate_fn=collate_fn_no_labels,
        )
        rep = {}
        for data in loader:
            batch = data["imp"].to(self.device)
            src_len = data["len"]
            attn_mask = generate_attention_masks(batch, src_len, self.device)
            out = self.model(batch, attn_mask)
            for idx_val, j in zip(data["idx"], range(len(out))):
                rep[idx_val] = out[j].cpu()
        return rep
