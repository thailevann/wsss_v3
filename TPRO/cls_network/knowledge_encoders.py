"""
Load knowledge features: clinical_bert (pkl) or bio_clinical_bert (pkl or HuggingFace).
"""
from __future__ import annotations

import os
import pickle as pkl
import torch


def load_knowledge_features(
    encoder_type: str,
    feature_path: str,
    knowledge_features_base_dir: str | None = None,
    bio_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    dataset_name: str | None = None,
) -> torch.Tensor:
    """
    Load k_fea tensor (num_tokens, dim).
    encoder_type: 'clinical_bert' | 'bio_clinical_bert'
    feature_path: e.g. 'clinical_bert/bcss_knowledge_fea' or 'bio_clinical_bert/bcss_knowledge_fea'
    """
    base = knowledge_features_base_dir or "./text&features"
    pkl_path = os.path.join(base, "text_features", feature_path + ".pkl")
    if os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as f:
            k_fea = pkl.load(f)
        if isinstance(k_fea, torch.Tensor):
            return k_fea.cpu()
        return torch.from_numpy(k_fea).float() if hasattr(k_fea, "__array__") else torch.tensor(k_fea).float()

    if encoder_type == "bio_clinical_bert":
        try:
            return _load_bio_clinical_bert_fea(
                pkl_path=pkl_path,
                model_name=bio_model_name,
                dataset_name=dataset_name or "bcss",
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Knowledge features not found at {pkl_path}. "
                "Generate pkl first (e.g. run script to extract Bio_ClinicalBERT features) or use clinical_bert."
            ) from e

    raise FileNotFoundError(f"Knowledge features not found: {pkl_path}")


def _load_bio_clinical_bert_fea(
    pkl_path: str,
    model_name: str,
    dataset_name: str,
) -> torch.Tensor:
    """Optional: load from HuggingFace and save pkl. Requires transformers."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise FileNotFoundError(
            f"Pkl not found at {pkl_path}. Install transformers and run feature extraction, or use pre-made pkl."
        )
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_path = os.path.join(
        os.path.dirname(pkl_path).replace("text_features", "text"),
        os.path.basename(pkl_path).replace("_fea.pkl", "_knowledge.txt"),
    )
    text_path = os.path.join(os.path.dirname(os.path.dirname(pkl_path)), "text", dataset_name + "_knowledge.txt")
    if not os.path.isfile(text_path):
        raise FileNotFoundError(f"Text file not found: {text_path}")
    with open(text_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    inputs = tokenizer(lines, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        out = model(**inputs)
        fea = out.last_hidden_state[:, 0]
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as f:
        pkl.dump(fea.cpu(), f)
    return fea.cpu()
