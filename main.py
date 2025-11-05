import argparse
import gc
import logging
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset as HFDataset
from pysentimiento.preprocessing import preprocess_tweet  # Used by spanish model
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# --- Configuration / constants ---
MODELS = [
    "nickprock/setfit-italian-hate-speech",
    "pysentimiento/robertuito-base-cased",
    "Twitter/twhin-bert-base",
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights_from_series(s: pd.Series) -> torch.Tensor:
    vc = s.value_counts().to_dict()
    total = sum(vc.values())
    weights = [total / vc[i] for i in sorted(vc.keys())]
    return torch.tensor(weights, dtype=torch.float)


class FocalLoss(nn.Module):
    def __init__(self, gamma=3.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)  # (batch,)
        p_t = torch.exp(-ce_loss)
        loss = ((1 - p_t) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# --- Custom Trainer for weighted loss (pretraining stage) ---
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, use_focal_loss: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        self.use_focal_loss = use_focal_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        if self.use_focal_loss:
            loss_fct = FocalLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# --- Dual encoder model ---
class DualEncoderForSequenceClassification(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config, use_focal_loss: bool = False):
        super().__init__(config)
        self.num_labels = config.num_labels
        # instantiate two encoders from the pretrained config
        self.encoder_text = AutoModel.from_config(config)
        self.encoder_bio = AutoModel.from_config(config)
        hidden_size = config.hidden_size

        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1), nn.Sigmoid()
        )
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.classifier = nn.Linear(hidden_size, config.num_labels)
        self.use_focal_loss = use_focal_loss
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=True):
        # both encoders receive the same input (text and bio were concatenated/tokenized as pair)
        out_text = self.encoder_text(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        out_bio = self.encoder_bio(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)

        h_text = out_text.last_hidden_state[:, 0]
        h_bio = out_bio.last_hidden_state[:, 0]

        combined = torch.cat([h_text, h_bio], dim=-1)
        gate = self.gate_layer(combined)  # shape (batch, 1)
        h_final = gate * h_text + (1 - gate) * h_bio
        pooled = self.dropout(h_final)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            cw = None
            if hasattr(self.config, "class_weights") and self.config.class_weights is not None:
                cw = torch.tensor(self.config.class_weights, device=logits.device, dtype=torch.float)
            if self.use_focal_loss:
                loss_fct = FocalLoss(weight=cw)
            else:
                loss_fct = nn.CrossEntropyLoss(weight=cw)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)


# --- Metrics ---
def compute_metrics_from_logits(logits, labels) -> Dict[str, float]:
    preds = np.argmax(logits, axis=-1)
    acc = float(accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds, average="binary"))
    return {"accuracy": acc, "f1": f1}


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    return compute_metrics_from_logits(logits, labels)


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


def preprocess_df_texts(df: pd.DataFrame, spanish: bool):
    if spanish:
        df = df.copy()
        df["text"] = df["text"].apply(lambda x: preprocess_tweet(x, lang="es"))
        df["bio"] = df["bio"].apply(lambda x: preprocess_tweet(x, lang="es"))
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


def train_pretrain_stage(args, logger):
    # Load tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    df = load_augmented_df(args.lang, logger)
    df = preprocess_df_texts(df, spanish=(args.lang in ["es", "both"]))

    # Use single input mode (dual_input=False) for pretrain
    train_ds, val_ds, test_ds, train_df, val_df, test_df = prepare_hf_datasets(
        df,
        tokenizer,
        label_column="lgbt",
        logger=logger,
        val_size=0.15,
        test_size=0.15,
        seed=args.seed,
    )

    # compute class weights
    class_weights = compute_class_weights_from_series(train_df["lgbt"])
    logger.info(f"Pretrain class weights: {class_weights.tolist()}")

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model.to(args.device)

    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR / "lgbt_pretrain" / f"{args.lang}" / NOW),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        # gradient_accumulation_steps=4,  # effective batch size 32
        num_train_epochs=1 if args.fast_dev else 10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(OUTPUT_DIR / "lgbt_pretrain" / f"{args.lang}" / NOW),
        logging_steps=50,
        save_total_limit=2,
        seed=args.seed,
        report_to="tensorboard",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        use_focal_loss=True,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    logger.info("Starting pre-training...")
    trainer.train()
    logger.info("Pre-training finished.")
    return trainer, tokenizer, train_df, val_df, test_df, df


def train_main_stage(args, logger, pretrain_trainer, tokenizer, full_df, freeze_bio_encoder: bool = False):
    train_ds, val_ds, test_ds, train_df, val_df, test_df = prepare_hf_datasets(
        full_df,
        tokenizer,
        label_column="label",
        logger=logger,
        val_size=0.15,
        test_size=0.15,
        seed=args.seed,
    )

    class_weights = compute_class_weights_from_series(train_df["label"])
    logger.info(f"Main task class weights: {class_weights.tolist()}")

    # Build Dual Encoder
    config = AutoConfig.from_pretrained(args.model, num_labels=2)
    config.class_weights = class_weights.tolist()
    combined = DualEncoderForSequenceClassification(config, use_focal_loss=True)

    # Load base encoder weights for text encoder (fresh from pretrained)
    base_model = AutoModel.from_pretrained(args.model)
    combined.encoder_text.load_state_dict(base_model.state_dict(), strict=True)
    logger.info("Loaded fresh pretrained weights into encoder_text")

    if pretrain_trainer is not None:
        # Load the LGBT-trained encoder weights into encoder_bio
        pretrain_state = pretrain_trainer.model.state_dict()

        # Extract encoder weights (model-specific, adjust prefix as needed)
        encoder_prefix = None
        for key in pretrain_state.keys():
            if "embeddings" in key:
                encoder_prefix = key.split(".")[0]
                break

        if encoder_prefix:
            encoder_state = {
                k.replace(f"{encoder_prefix}.", ""): v
                for k, v in pretrain_state.items()
                if k.startswith(f"{encoder_prefix}.")
            }

            # Load into encoder_bio
            missing, unexpected = combined.encoder_bio.load_state_dict(encoder_state, strict=False)
            logger.info(f"Loaded LGBT-pretrained weights into encoder_bio")
            logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            logger.info(f"Missing keys: {missing}")
            logger.info(f"Unexpected keys: {unexpected}")
        else:
            logger.warning("Could not determine encoder prefix, initializing encoder_bio with random weights")
    else:
        logger.info("No pretrain_trainer provided, initializing encoder_bio with base pretrained weights")
        combined.encoder_bio.load_state_dict(base_model.state_dict(), strict=True)

    # Freeze bio encoder if specified
    if freeze_bio_encoder:
        for param in combined.encoder_bio.parameters():
            param.requires_grad = False
        logger.info("Bio encoder is frozen (not fine-tuned)")
    else:
        logger.info("Both encoders will be fine-tuned (not frozen)")

    # Cleanup
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    combined.to(args.device)

    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR / "dual_encoder" / f"{args.lang}" / NOW),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        # gradient_accumulation_steps=4,  # effective batch size 32
        num_train_epochs=1 if args.fast_dev else 10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(OUTPUT_DIR / "dual_encoder" / f"{args.lang}" / NOW),
        logging_steps=50,
        save_total_limit=2,
        seed=args.seed,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=combined,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting main task training...")
    trainer.train()
    logger.info("Main task training finished.")
    return trainer, test_ds


def evaluate_and_save(trainer: Trainer, test_dataset: HFDataset, logger: logging.Logger, out_prefix="results"):
    preds_out = trainer.predict(test_dataset)
    metrics = getattr(preds_out, "metrics", {}) or {}
    logger.info(f"Test set metrics: {metrics}")

    # log accuracy/f1 if available under common names
    acc = metrics.get("test_accuracy") or metrics.get("accuracy")
    f1 = metrics.get("test_f1") or metrics.get("f1")
    if acc is not None and f1 is not None:
        logger.info(f"Test set accuracy: {acc:.4f}, F1: {f1:.4f}")

    # If predictions exist, perform error analysis and confusion matrix
    if getattr(preds_out, "predictions", None) is not None and getattr(preds_out, "label_ids", None) is not None:
        logits = preds_out.predictions
        labels = preds_out.label_ids
        preds = np.argmax(logits, axis=-1)

        # compute probabilities / confidences
        try:
            probs = softmax(logits, axis=1) if logits.ndim == 2 else logits
            confidences = np.max(probs, axis=1)
        except Exception:
            confidences = np.zeros_like(preds, dtype=float)

        # convert HF Dataset to pandas safely without mutating format
        try:
            results = test_dataset.to_pandas()
        except Exception:
            # fallback
            ds_pandas = test_dataset.with_format("pandas")
            results = ds_pandas.to_pandas()

        results = results.reset_index(drop=True)
        results["predicted_label"] = preds
        results["true_label"] = labels
        results["confidence"] = confidences
        results["is_correct"] = results["true_label"] == results["predicted_label"]

        out_csv = RESULTS_DIR / f"{out_prefix}_error_analysis.csv"
        results.to_csv(out_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Error analysis saved to {out_csv}")

        cm = confusion_matrix(labels, preds)
        logger.info(f"Confusion matrix:\n{cm}")

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-Reclamatory", "Reclamatory"],
            yticklabels=["Non-Reclamatory", "Reclamatory"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(RESULTS_DIR / f"{out_prefix}_confusion.png")
        plt.close()
    else:
        logger.warning("Predictions or label_ids not found in trainer.predict output; skipping error analysis.")


def main():
    parser = argparse.ArgumentParser(description="Pre-train and fine-tune models for reclaim/labels")
    parser.add_argument("--lang", choices=["it", "es", "both"], default="it")
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast-dev", dest="fast_dev", action="store_true", help="Run fewer epochs for quick tests")
    parser.add_argument(
        "--fresh",
        dest="fresh",
        action="store_true",
        help="Delete previous logs and results and start fresh.",
    )
    parser.add_argument(
        "--skip-pretrain",
        dest="skip_pretrain",
        action="store_true",
        help="Skip pretraining stage and train dual encoder from scratch",
    )
    args = parser.parse_args()

    # Setup logging and directories
    global LOGS_DIR, OUTPUT_DIR, RESULTS_DIR, NOW
    LOGS_DIR = Path("./logs").absolute()
    OUTPUT_DIR = Path("./output").absolute()
    RESULTS_DIR = Path("./results").absolute()
    NOW = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Fresh start
    if args.fresh:
        if LOGS_DIR.exists():
            shutil.rmtree(LOGS_DIR)
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        if RESULTS_DIR.exists():
            shutil.rmtree(RESULTS_DIR)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = logging.getLogger("multipride")
    logger.handlers.clear()  # Clear any existing handlers
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh_path = LOGS_DIR / f"run_{NOW}.log"
    fh = logging.FileHandler(fh_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    logger.info(f"Device: {args.device}  Model: {args.model}  Lang: {args.lang}")

    if not args.skip_pretrain:
        # Pretrain Stage
        pretrain_trainer, tokenizer, train_df, val_df, test_df_pretrain, full_df = train_pretrain_stage(args, logger)

        # Evaluate pretrain stage on test set
        logger.info("Evaluating pretrain stage on test set...")
        tokenizer_temp = AutoTokenizer.from_pretrained(args.model)
        df_temp = load_augmented_df(args.lang, logger)
        df_temp = preprocess_df_texts(df_temp, spanish=(args.lang in ["es", "both"]))
        _, _, pretrain_test_ds, _, _, _ = prepare_hf_datasets(
            df_temp,
            tokenizer_temp,
            label_column="lgbt",
            logger=logger,
            seed=args.seed,
        )
        evaluate_and_save(pretrain_trainer, pretrain_test_ds, logger, out_prefix="lgbt_pretrain")
    else:
        logger.info("Skipping pretrain stage")
        pretrain_trainer = None
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        full_df = load_augmented_df(args.lang, logger)
        full_df = preprocess_df_texts(full_df, spanish=(args.lang in ["es", "both"]))

    # Main Stage
    main_trainer, test_dataset = train_main_stage(args, logger, pretrain_trainer, tokenizer, full_df)

    # Evaluate on test set
    logger.info("Evaluating main model on test set...")
    evaluate_and_save(main_trainer, test_dataset, logger, out_prefix="dual_encoder")

    logger.info("All done.")


if __name__ == "__main__":
    main()
