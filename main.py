import argparse
import logging
import shutil
import torch
from transformers import (
    AutoTokenizer,
)

import include.constants as constants
from include.constants import MODELS
from include.utilities import (
    set_seed,
    load_augmented_df,
    prepare_hf_datasets,
)
from include.training import (
    train_pretrain_stage,
    train_main_stage,
    evaluate_and_save
)


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
    parser.add_argument(
        "--freeze-bio-encoder",
        dest="freeze_bio_encoder",
        action="store_true",
        help="Freeze the bio encoder during main task training",
    )
    parser.add_argument(
        "--use-focal-loss",
        dest="use_focal_loss",
        action="store_true",
        help="Use Focal Loss instead of Cross-Entropy Loss",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for Focal Loss",
    )
    parser.add_argument(
        "--weighted-sampling",
        dest="weighted_sampling",
        action="store_true",
        help="Use WeightedRandomSampler with replacement for dual encoder training",
    )
    args = parser.parse_args()

    # Fresh start
    if args.fresh:
        if constants.LOGS_DIR.exists():
            shutil.rmtree(constants.LOGS_DIR)
        if constants.OUTPUT_DIR.exists():
            shutil.rmtree(constants.OUTPUT_DIR)
        if constants.RESULTS_DIR.exists():
            shutil.rmtree(constants.RESULTS_DIR)

    constants.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    constants.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    constants.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    fh_path = constants.LOGS_DIR / f"run_{constants.NOW}.log"
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

    # Main Stage
    main_trainer, test_dataset = train_main_stage(
        args, logger, pretrain_trainer, tokenizer, full_df, freeze_bio_encoder=args.freeze_bio_encoder
    )

    # Evaluate on test set
    logger.info("Evaluating main model on test set...")
    evaluate_and_save(main_trainer, test_dataset, logger, out_prefix="dual_encoder")

    logger.info("All done.")


if __name__ == "__main__":
    main()
