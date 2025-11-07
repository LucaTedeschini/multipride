import random
import numpy as np
import torch
import torch.nn as nn
from transformers import Trainer
import pandas as pd

from include.losses import FocalLoss

import logging
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import (
    Trainer,
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- Custom Trainer for weighted loss (pretraining stage) ---
class WeightedTrainer(Trainer):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        use_focal_loss: bool = False,
        gamma: float = 3.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        self.use_focal_loss = use_focal_loss
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # Added so pylance will stop complaining
        assert labels is not None

        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        if self.use_focal_loss:
            loss_fct = FocalLoss(gamma=self.gamma, weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    

def compute_class_weights_from_series(s: pd.Series) -> torch.Tensor:
    vc = s.value_counts().to_dict()
    total = sum(vc.values())
    weights = [total / vc[i] for i in sorted(vc.keys())]
    return torch.tensor(weights, dtype=torch.float)


# --- Metrics ---
def compute_metrics_from_logits(logits, labels) -> Dict[str, float]:
    preds = np.argmax(logits, axis=-1)
    acc = float(accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds, average="binary"))
    precision = float(precision_score(labels, preds, average="binary"))
    recall = float(recall_score(labels, preds, average="binary"))
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    return compute_metrics_from_logits(logits, labels)


def compute_label_proportions(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, label_column: str, logger: logging.Logger
):
    def label_props(df: pd.DataFrame) -> Dict[int, float]:
        counts = df[label_column].value_counts(normalize=True).to_dict()
        return {k: counts.get(k, 0.0) for k in sorted(counts.keys())}

    train_props = label_props(train_df)
    val_props = label_props(val_df)
    test_props = label_props(test_df)

    logger.info(f"Label proportions in training set: {train_props}")
    logger.info(f"Label proportions in validation set: {val_props}")
    logger.info(f"Label proportions in test set: {test_props}")

# --- Main flow functions ---
def load_augmented_df(lang: str, logger: logging.Logger) -> pd.DataFrame:
    files = {
        "it": Path("dataset/augmented_it.csv"),
        "es": Path("dataset/augmented_es.csv"),
    }
    if lang == "both":
        df_it = pd.read_csv(files["it"])
        df_es = pd.read_csv(files["es"])
        df = pd.concat([df_it, df_es], ignore_index=True)
    else:
        df = pd.read_csv(files[lang])
    df = df.fillna({"bio": ""})
    df["bio"] = df["bio"].replace("", "[NO BIO]")  # Use special token for missing bios
    logger.info(f"Loaded dataset for lang={lang}, length={len(df)}")
    return df

def tokenize_function_single(tokenizer, max_length=128):
    """For models that use single concatenated text+bio input (pretrain stage)"""

    def fn(batch):
        return tokenizer(
            batch["text"],
            batch["bio"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return fn


def prepare_hf_datasets(
    df: pd.DataFrame,
    tokenizer,
    label_column: str,
    logger: logging.Logger,
    val_size=0.15,
    test_size=0.15,
    seed=42,
):
    # First split: separate test set
    train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df[label_column], random_state=seed)

    # Second split: separate train and validation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size / (1 - test_size),  # Adjust proportion
        stratify=train_val_df[label_column],
        random_state=seed,
    )

    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    compute_label_proportions(train_df, val_df, test_df, label_column, logger)

    train_ds = HFDataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = HFDataset.from_pandas(val_df.reset_index(drop=True))
    test_ds = HFDataset.from_pandas(test_df.reset_index(drop=True))

    tok = tokenize_function_single(tokenizer)
    format_columns = ["input_ids", "attention_mask", "labels"]

    train_ds = train_ds.map(tok, batched=True)
    val_ds = val_ds.map(tok, batched=True)
    test_ds = test_ds.map(tok, batched=True)

    # Rename label column to "labels" which Trainer expects
    if label_column != "labels":
        train_ds = train_ds.rename_column(label_column, "labels")
        val_ds = val_ds.rename_column(label_column, "labels")
        test_ds = test_ds.rename_column(label_column, "labels")

    train_ds.set_format(type="torch", columns=format_columns)
    val_ds.set_format(type="torch", columns=format_columns)
    test_ds.set_format(type="torch", columns=format_columns)

    logger.info(f"Prepared HF datasets and tokenized.")
    return train_ds, val_ds, test_ds, train_df, val_df, test_df




def prepare_hf_weighted_datasets(
    df: pd.DataFrame,
    tokenizer,
    label_column: str,
    logger: logging.Logger,
    val_size=0.15,
    test_size=0.15,
    seed=42,
):
    # First split: separate test set
    train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df[label_column], random_state=seed)

    # Second split: separate train and validation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size / (1 - test_size),
        stratify=train_val_df[label_column],
        random_state=seed,
    )

    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    compute_label_proportions(train_df, val_df, test_df, label_column, logger)

    # Create datasets
    train_ds = HFDataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = HFDataset.from_pandas(val_df.reset_index(drop=True))
    test_ds = HFDataset.from_pandas(test_df.reset_index(drop=True))

    tok = tokenize_function_single(tokenizer)
    format_columns = ["input_ids", "attention_mask", "labels"]

    train_ds = train_ds.map(tok, batched=True)
    val_ds = val_ds.map(tok, batched=True)
    test_ds = test_ds.map(tok, batched=True)

    # Rename label column to "labels"
    if label_column != "labels":
        train_ds = train_ds.rename_column(label_column, "labels")
        val_ds = val_ds.rename_column(label_column, "labels")
        test_ds = test_ds.rename_column(label_column, "labels")

    train_ds.set_format(type="torch", columns=format_columns)
    val_ds.set_format(type="torch", columns=format_columns)
    test_ds.set_format(type="torch", columns=format_columns)

    # Compute class weights for WeightedRandomSampler
    train_labels = train_df[label_column].values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts

    # Assign weight to each sample based on its class
    sample_weights = class_weights[train_labels]
    sample_weights = list(sample_weights)

    # Create WeightedRandomSampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    logger.info(f"Created WeightedRandomSampler with class weights: {class_weights}")
    logger.info(f"Prepared HF datasets and tokenized.")

    return train_ds, val_ds, test_ds, train_df, val_df, test_df, sampler


def compile_configuration(config, logger):

    if not hasattr(config, "lpft"):
        logger.warning("lpft not set by config file. Assuming standard training")
        config.lpft = False

    if not hasattr(config, "val_size_pretrain"):
        logger.warning("val_size_pretrain not set by config file. Setting to default value 0.15")
        config.val_size_pretrain = 0.15
    
    if not hasattr(config, "test_size_pretrain"):
        logger.warning("test_size_pretrain not set by config file. Setting to default value 0.15")
        config.test_size_pretrain = 0.15
    
    if not hasattr(config, "lr_pretrain"):
        logger.warning("lr_pretrain not set by config file. Setting to default value 2e-5")
        config.lr_pretrain = 2e-5

    if not hasattr(config, "batch_size_pretrain"):
        logger.warning("batch_size_pretrain not set by config file. Setting to default value 8")
        config.batch_size_pretrain = 8

    if not hasattr(config, "weight_decay_pretrain"):
        logger.warning("weight_decay_pretrain not set by config file. Setting to default value 0.1")
        config.weight_decay_pretrain = 0.1
    
    # Different setting based on LPFT

    if config.lpft:
        if not hasattr(config, "val_size_lpft"):
            logger.warning("val_size_lpft not set by config file. Setting to default value 0.15")
            config.val_size_lpft = 0.15

        if not hasattr(config, "test_size_lpft"):
            logger.warning("test_size_lpft not set by config file. Setting to default value 0.15")
            config.test_size_lpft = 0.15

        if not hasattr(config, "lr_lp"):
            logger.warning("lr_lp not set by config file. Setting to default value 2e-5")
            config.lr_lp = 2e-5


        if not hasattr(config, "batch_size_lp"):
            logger.warning("batch_size_lp not set by config file. Setting to default value 8")
            config.batch_size_lp = 8
        
        if not hasattr(config, "weight_decay_lp"):
            logger.warning("weight_decay_lp not set by config file. Setting to default value 0.1")
            config.weight_decay_lp = 0.1

        if not hasattr(config, "lr_ft"):
            logger.warning("lr_ft not set by config file. Setting to default value 5e-6")
            config.lr_ft = 2e-5


        if not hasattr(config, "batch_size_ft"):
            logger.warning("batch_size_ft not set by config file. Setting to default value 8")
            config.batch_size_ft = 8
        
        if not hasattr(config, "weight_decay_ft"):
            logger.warning("weight_decay_ft not set by config file. Setting to default value 0.1")
            config.weight_decay_ft = 0.1


    else:
        if not hasattr(config, "val_size_mainstage"):
            logger.warning("val_size_mainstage not set by config file. Setting to default value 0.15")
            config.val_size_mainstage = 0.15

        if not hasattr(config, "test_size_mainstage"):
            logger.warning("test_size_mainstage not set by config file. Setting to default value 0.15")
            config.test_size_mainstage = 0.15

        if not hasattr(config, "lr_mainstage"):
            logger.warning("lr_mainstage not set by config file. Setting to default value 2e-5")
            config.lr_mainstage = 2e-5


        if not hasattr(config, "batch_size_mainstage"):
            logger.warning("batch_size_mainstage not set by config file. Setting to default value 8")
            config.batch_size_mainstage = 8
        
        if not hasattr(config, "weight_decay_mainstage"):
            logger.warning("weight_decay_mainstage not set by config file. Setting to default value 0.1")
            config.weight_decay_mainstage = 0.1

    return config

    