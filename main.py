import argparse
import logging
import shutil
import torch
from omegaconf import OmegaConf
from transformers import (
    AutoTokenizer,
)

import include.constants as constants
from include.constants import MODELS
from include.utilities import (
    set_seed,
    load_augmented_df,
    prepare_hf_datasets,
    compile_configuration
)
from include.training import (
    train_pretrain_stage,
    train_main_stage,
    evaluate_and_save,
    train_main_stage_LPFT,
    run_test_evaluation
)
import os


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
    parser.add_argument(
        "--config-file",
        dest="config_file",
        type=str,
        help="Pass a path to a config.yaml file. !!The config file could override other configurations!!",
    )
    parser.add_argument(
        "--is_evaluation",
        dest="is_evaluation",
        type=bool,
        help="Run the trained model on the real test set, and save results",
        default=False
    )
    args = parser.parse_args()

    if args.config_file:
        conf = OmegaConf.load(args.config_file)
    else:
        conf = OmegaConf.create()
        conf.lang = args.lang
        conf.seed = args.seed
        conf.model = args.model
        conf.skip_pretrain = args.skip_pretrain
        conf.freeze_bio_encoder = args.freeze_bio_encoder
        conf.use_focal_loss = args.use_focal_loss
        conf.gamma = args.gamma
        conf.weighted_sampling = args.weighted_sampling
    
    # README: these parameters are not overrided by the config file
    conf.fast_dev = args.fast_dev
    conf.fresh = args.fresh
    conf.is_evaluation = args.is_evaluation


    # Fresh start
    if conf.fresh:
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

    conf.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(conf.seed)
    logger.info(f"Device: {conf.device}  Model: {conf.model}  Lang: {conf.lang}")

    # Check configuration file
    conf = compile_configuration(conf, logger)


    if not conf.skip_pretrain:
        # Pretrain Stage
        pretrain_trainer, tokenizer, train_df, val_df, test_df_pretrain, full_df = train_pretrain_stage(conf, logger)

        # Evaluate pretrain stage on test set
        logger.info("Evaluating pretrain stage on test set...")
        tokenizer_temp = AutoTokenizer.from_pretrained(conf.model)
        df_temp = load_augmented_df(conf.lang, logger)
        _, _, pretrain_test_ds, _, _, _ = prepare_hf_datasets(
            df_temp,
            tokenizer_temp,
            label_column="lgbt",
            logger=logger,
            seed=conf.seed,
        )
        evaluate_and_save(pretrain_trainer, pretrain_test_ds, logger, out_prefix="lgbt_pretrain")
    else:
        logger.info("Skipping pretrain stage")
        pretrain_trainer = None
        tokenizer = AutoTokenizer.from_pretrained(conf.model)
        full_df = load_augmented_df(conf.lang, logger)

    if not conf.lpft:
        # Main Stage
        main_trainer, test_dataset = train_main_stage(
            conf, logger, pretrain_trainer, tokenizer, full_df, freeze_bio_encoder=conf.freeze_bio_encoder
        )
    else:
        # Main Stage (LP-FT)
        main_trainer, test_dataset = train_main_stage_LPFT(
            conf, logger, pretrain_trainer, tokenizer, full_df, freeze_bio_encoder=conf.freeze_bio_encoder
        )

    # Evaluate on test set
    logger.info("Evaluating main model on test set...")
    evaluate_and_save(main_trainer, test_dataset, logger, out_prefix="dual_encoder")

    if conf.is_evaluation:
        # Run the trained model on the test set
        logger.info("\n\n======== STARTING EVALUATION ON TEST SET ========\n\n")
        result_df = run_test_evaluation(main_trainer, logger, tokenizer, conf.lang)
        raw_results = result_df
        submission_results = result_df[["id", "label", "lang"]]
        readable_results = result_df[["text", "label"]]
        
        print(submission_results)
        os.makedirs("submission", exist_ok=True)
        logging.info(f"SAVING PREDICTIONS TO submission/{conf.lang}.tsv...")
        submission_results.to_csv(f"submission/{conf.lang}.tsv", sep="\t", index=False)
        raw_results.to_csv(f"submission/raw_{conf.lang}.tsv", sep="\t", index=False)
        readable_results.to_csv(f"submission/readable_{conf.lang}.tsv", sep="\t", index=False)
        logging.info("DONE!")

    logger.info("All done.")


if __name__ == "__main__":
    main()
