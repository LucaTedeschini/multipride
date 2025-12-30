import gc
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset as HFDataset
from omegaconf import OmegaConf
from scipy.special import softmax
from sklearn.metrics import (
    confusion_matrix,
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import include.constants as constants
from include.networks import DualEncoderForSequenceClassification
from include.utilities import (
    WeightedTrainer,
    compute_class_weights_from_series,
    compute_metrics,
    load_augmented_df,
    load_test_df,
    prepare_hf_datasets,
    prepare_hf_weighted_datasets,
    prepare_test_dataset,
)

def compute_baseline(conf, logger):
    # Load tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(conf.model)
    df = load_augmented_df(conf.lang, logger)

    train_ds, val_ds, test_ds, train_df, val_df, test_df = prepare_hf_datasets(
        df,
        tokenizer,
        label_column="label",
        logger=logger,
        val_size=conf.val_size_pretrain,
        test_size=conf.test_size_pretrain,
        seed=conf.seed,
    )

    # compute class weights
    class_weights = compute_class_weights_from_series(train_df["label"])
    logger.info(f"Pretrain class weights: {class_weights.tolist()}")

    model = AutoModelForSequenceClassification.from_pretrained(conf.model, num_labels=2)
    model.to(conf.device)

    training_args = TrainingArguments(
        output_dir=str(constants.RESULTS_DIR / "baseline" / f"{conf.lang}" / constants.NOW),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=conf.lr_pretrain,
        per_device_train_batch_size=conf.batch_size_pretrain,
        per_device_eval_batch_size=conf.batch_size_pretrain,
        num_train_epochs=1 if conf.fast_dev else 10,
        weight_decay=conf.weight_decay_pretrain,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(constants.OUTPUT_DIR / "baseline" / f"{conf.lang}" / constants.NOW),
        logging_steps=50,
        save_total_limit=1,
        seed=conf.seed,
        report_to="tensorboard",
        push_to_hub=False,
        hub_private_repo = True,
        hub_model_id=f"MultiPRIDE-baseline-{conf.lang}",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        use_focal_loss=conf.use_focal_loss,
        gamma=conf.gamma,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    logger.info("Starting pre-training...")
    trainer.train()
    trainer.push_to_hub(f"MultiPRIDE-LGBT-Baseline-{conf.lang}")
    logger.info("Pre-training finished.")
    return trainer, tokenizer, train_df, val_df, test_df, df


def train_pretrain_stage(conf, logger):
    # Load tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(conf.model)
    df = load_augmented_df(conf.lang, logger)

    # Use single input mode (dual_input=False) for pretrain
    train_ds, val_ds, test_ds, train_df, val_df, test_df = prepare_hf_datasets(
        df,
        tokenizer,
        label_column="lgbt",
        logger=logger,
        val_size=conf.val_size_pretrain,
        test_size=conf.test_size_pretrain,
        seed=conf.seed,
    )

    # compute class weights
    class_weights = compute_class_weights_from_series(train_df["lgbt"])
    logger.info(f"Pretrain class weights: {class_weights.tolist()}")

    model = AutoModelForSequenceClassification.from_pretrained(conf.model, num_labels=2)
    model.to(conf.device)

    training_args = TrainingArguments(
        output_dir=str(constants.RESULTS_DIR / "lgbt_pretrain" / f"{conf.lang}" / constants.NOW),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=conf.lr_pretrain,
        per_device_train_batch_size=conf.batch_size_pretrain,
        per_device_eval_batch_size=conf.batch_size_pretrain,
        # gradient_accumulation_steps=4,  # effective batch size 32
        num_train_epochs=1 if conf.fast_dev else 10,
        weight_decay=conf.weight_decay_pretrain,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(constants.OUTPUT_DIR / "lgbt_pretrain" / f"{conf.lang}" / constants.NOW),
        logging_steps=50,
        save_total_limit=1,
        seed=conf.seed,
        report_to="tensorboard",
        push_to_hub=False,
        hub_private_repo = True,
        hub_model_id=f"MultiPRIDE-LGBT-Pretrain-{conf.lang}",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        use_focal_loss=conf.use_focal_loss,
        gamma=conf.gamma,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    logger.info("Starting pre-training...")
    trainer.train()
    trainer.push_to_hub(f"MultiPRIDE-LGBT-Pretrain-{conf.lang}")
    logger.info("Pre-training finished.")
    return trainer, tokenizer, train_df, val_df, test_df, df


def train_main_stage(conf, logger, pretrain_trainer, tokenizer, full_df, freeze_bio_encoder: bool = False):
    train_ds, val_ds, test_ds, train_df, val_df, test_df, sampler = prepare_hf_weighted_datasets(
        full_df,
        tokenizer,
        label_column="label",
        logger=logger,
        val_size=conf.val_size_mainstage,
        test_size=conf.test_size_mainstage,
        seed=conf.seed,
    )

    class_weights = compute_class_weights_from_series(train_df["label"])
    logger.info(f"Main task class weights: {class_weights.tolist()}")

    # Build Dual Encoder
    config = AutoConfig.from_pretrained(conf.model, num_labels=2, dtype=torch.bfloat16)
    config.class_weights = class_weights.tolist()
    combined = DualEncoderForSequenceClassification(
        config,
        use_focal_loss=conf.use_focal_loss,
        gamma=conf.gamma,
    )

    # Load base encoder weights for text encoder (fresh from pretrained)
    base_model = AutoModel.from_pretrained(conf.model)
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

    combined.to(conf.device)

    training_args = TrainingArguments(
        output_dir=str(constants.RESULTS_DIR / "dual_encoder" / f"{conf.lang}" / constants.NOW),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=conf.lr_mainstage,
        per_device_train_batch_size=conf.batch_size_mainstage,
        per_device_eval_batch_size=conf.batch_size_mainstage,
        # gradient_accumulation_steps=4,  # effective batch size 32
        num_train_epochs=1 if conf.fast_dev else 10,
        weight_decay=conf.weight_decay_mainstage,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(constants.OUTPUT_DIR / "dual_encoder" / f"{conf.lang}" / constants.NOW),
        logging_steps=50,
        save_total_limit=1,
        seed=conf.seed,
        report_to="tensorboard",
        push_to_hub=False,
        hub_private_repo = True,
        hub_model_id=f"MultiPRIDE-DualEncoder-MainStage-{conf.lang}",
    )

    trainer = Trainer(
        model=combined,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    if conf.weighted_sampling:
        logger.info("Using WeightedRandomSampler for training")

        # Replace the default sampler
        def get_train_sampler(dataset):
            return sampler

        trainer._get_train_sampler = get_train_sampler

    logger.info("Starting main task training...")
    trainer.train()
    trainer.push_to_hub(f"MultiPRIDE-DualEncoder-MainStage-{conf.lang}")
    logger.info("Main task training finished.")
    return trainer, test_ds


def train_main_stage_LPFT(conf, logger, pretrain_trainer, tokenizer, full_df, freeze_bio_encoder: bool = False):
    train_ds, val_ds, test_ds, train_df, val_df, test_df, sampler = prepare_hf_weighted_datasets(
        full_df,
        tokenizer,
        label_column="label",
        logger=logger,
        val_size=conf.val_size_lpft,
        test_size=conf.test_size_lpft,
        seed=conf.seed,
    )

    class_weights = compute_class_weights_from_series(train_df["label"])
    logger.info(f"Main task class weights: {class_weights.tolist()}")

    # Build Dual Encoder
    config = AutoConfig.from_pretrained(conf.model, num_labels=2, dtype=torch.bfloat16)
    config.class_weights = class_weights.tolist()
    combined = DualEncoderForSequenceClassification(
        config,
        use_focal_loss=conf.use_focal_loss,
        gamma=conf.gamma,
    )

    # Load base encoder weights for text encoder (fresh from pretrained)
    base_model = AutoModel.from_pretrained(conf.model)
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

    combined.to(conf.device)

    for param in combined.encoder_text.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=str(constants.RESULTS_DIR / "dual_encoder" / f"{conf.lang}" / constants.NOW),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=conf.lr_lp,
        per_device_train_batch_size=conf.batch_size_lp,
        per_device_eval_batch_size=conf.batch_size_lp,
        # gradient_accumulation_steps=4,  # effective batch size 32
        num_train_epochs=1 if conf.fast_dev else 10,
        weight_decay=conf.weight_decay_lp,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(constants.OUTPUT_DIR / "dual_encoder" / f"{conf.lang}" / constants.NOW),
        logging_steps=50,
        save_total_limit=1,
        seed=conf.seed,
        report_to="tensorboard",
        push_to_hub=False,
        hub_private_repo = True,
        hub_model_id=f"MultiPRIDE-DualEncoder-LPFT-{conf.lang}"
    )

    trainer = Trainer(
        model=combined,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    if conf.weighted_sampling:
        logger.info("Using WeightedRandomSampler for training")

        # Replace the default sampler
        def get_train_sampler(dataset):
            return sampler

        trainer._get_train_sampler = get_train_sampler

    logger.info("Starting linear prober training...")
    trainer.train()

    trainer.push_to_hub(f"MultiPRIDE-DualEncoder-LPFT-{conf.lang}")
    # Fine tuning
    for param in combined.encoder_text.parameters():
        param.requires_grad = True

    training_args = TrainingArguments(
        output_dir=str(constants.RESULTS_DIR / "dual_encoder" / f"{conf.lang}" / constants.NOW),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=conf.lr_ft,
        per_device_train_batch_size=conf.batch_size_ft,
        per_device_eval_batch_size=conf.batch_size_ft,
        # gradient_accumulation_steps=4,  # effective batch size 32
        num_train_epochs=1 if conf.fast_dev else 10,
        weight_decay=conf.weight_decay_ft,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(constants.OUTPUT_DIR / "dual_encoder" / f"{conf.lang}" / constants.NOW),
        logging_steps=50,
        save_total_limit=1,
        seed=conf.seed,
        report_to="tensorboard",
        push_to_hub=False,
        hub_private_repo = True,
        hub_model_id=f"MultiPRIDE-DualEncoder-MainStage-FT-{conf.lang}"

    )

    trainer = Trainer(
        model=combined,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    if conf.weighted_sampling:
        logger.info("Using WeightedRandomSampler for training")

        # Replace the default sampler
        def get_train_sampler(dataset):
            return sampler

        trainer._get_train_sampler = get_train_sampler

    logger.info("Starting fine tuning training...")
    trainer.train()
    trainer.push_to_hub(f"MultiPRIDE-DualEncoder-MainStage-FT-{conf.lang}")

    logger.info("Main task training finished.")
    return trainer, test_ds


def evaluate_and_save(
    conf, trainer: Trainer, test_dataset: HFDataset, logger: logging.Logger, out_prefix="results"
) -> np.ndarray:
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

        out_csv = constants.RESULTS_DIR / f"{out_prefix}_error_analysis.csv"
        results.to_csv(out_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Error analysis saved to {out_csv}")

        cm = confusion_matrix(labels, preds, normalize="all")
        logger.info(f"Confusion matrix:\n{cm}")

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=["Non-Reclamatory", "Reclamatory"],
            yticklabels=["Non-Reclamatory", "Reclamatory"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(constants.RESULTS_DIR / f"{out_prefix}_{conf.name}_{conf.seed}_confusion.png")
        plt.close()

        # Saving F1 and Acc metrics on file
        with open(f"{constants.RESULTS_DIR}/{out_prefix}_{conf.name}_{conf.seed}.csv", "w") as f:
            f.write("f1,acc\n")
            f.write(f"{f1},{acc}\n")

    else:
        logger.warning("Predictions or label_ids not found in trainer.predict output; skipping error analysis.")

    return cm


def run_test_evaluation(main_trainer: Trainer, logger: logging.Logger, tokenizer, lang: str) -> pd.DataFrame:
    # Load the correct dataset (italian / spanish)
    # Build the dataloader
    # Run the inference
    # Save results
    if lang == "it":
        # load italian
        df = load_test_df(lang, logger)
    elif lang == "es":
        # load spanish
        df = load_test_df(lang, logger)
    else:
        raise Exception("Language not recognized!")

    dataloader = prepare_test_dataset(df, tokenizer, logger)
    raw_out = main_trainer.predict(dataloader)
    logits = raw_out.predictions
    predictions = np.argmax(logits, axis=-1)
    df["label"] = predictions

    return df
