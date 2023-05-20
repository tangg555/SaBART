"""
@Desc:
@Reference:
- logger and WandLogger
Weights and Biases is a third-party logger
https://pytorch-lightning.readthedocs.io/en/latest/common/loggers.html
@Notes:

"""

import sys
import glob
import os
from pathlib import Path
import numpy as np
import pytorch_lightning as pl

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.configuration.chatbot.config_args import parse_args_for_config
from src.utils.wrapper import print_done
from src.utils.string_utils import are_same_strings
from src.models.basic_pl_trainer import BasicPLTrainer
from src.modules.pl_callbacks import Seq2SeqLoggingCallback, Seq2SeqCheckpointCallback
from src.models.chabot import (
    MyBart,
    OneHopGNNBart,
    OneHopGNNBartNoAgg,
    OneHopGNNBartNoChild,
    OneHopGNNBartEH,
)

class ChatbotTrainer(BasicPLTrainer):
    def __init__(self, args, trainer_name="gnn-bart-trainer"):
        # parameters
        super().__init__(args, trainer_name=trainer_name)

        self._init_model(self.args)
        self._init_logger(self.args, self.model)
        self._init_pl_trainer(self.args, self.model, self.logger)

    @print_done(desc="Creating directories and fix random seeds")
    def _init_args(self, args):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(42)
        pl.seed_everything(args.seed, workers=True)  # reproducibility

    @print_done(desc="initialize model")
    def _init_model(self, args):
        # automatically download from huggingface project
        print(f"model_path: {args.model_name_or_path}")
        print(args.model_name)
        # ============= bart ===============
        if are_same_strings(args.model_name, "chatbot_onehop"):
            self.model: OneHopGNNBart = OneHopGNNBart(args)
        elif are_same_strings(args.model_name, "chatbot_onehop_no_agg"):
            self.model: OneHopGNNBartNoAgg = OneHopGNNBartNoAgg(args)
        elif are_same_strings(args.model_name, "chatbot_onehop_no_child"):
            self.model: OneHopGNNBartNoChild = OneHopGNNBartNoChild(args)
        elif are_same_strings(args.model_name, "chatbot_onehop_eh"):
            self.model: OneHopGNNBartEH = OneHopGNNBartEH(args)
        elif are_same_strings(args.model_name, "mybart"):
            self.model: MyBart = MyBart(args)
        else:
            raise NotImplementedError(f"args.model_name: {args.model_name}")

    @print_done("set up pytorch lightning trainer")
    def _init_pl_trainer(self, args, model, logger):
        extra_callbacks = []
        self.checkpoint_callback = Seq2SeqCheckpointCallback(
            output_dir=self.save_dir,
            experiment_name=self.experiment_name,
            monitor="val_loss",
            save_top_k=args.save_top_k,
            every_n_val_epochs=args.every_n_val_epochs,
            verbose=args.ckpt_verbose,
        )

        self.pl_trainer: pl.Trainer = pl.Trainer.from_argparse_args(
            args,
            enable_model_summary=False,
            callbacks=[self.checkpoint_callback, Seq2SeqLoggingCallback(), pl.callbacks.ModelSummary(max_depth=1)]
                      + extra_callbacks,
            logger=logger,
            **self.train_params,
        )

    def train(self):
        self.auto_find_lr_rate()
        self.auto_find_batch_size()

        self.pl_trainer.logger.log_hyperparams(self.args)

        if self.checkpoints:
            # training
            best_ckpt = self.checkpoints[-1]
            self.pl_trainer.fit(self.model, ckpt_path=best_ckpt)
        else:
            self.pl_trainer.fit(self.model)


if __name__ == '__main__':
    hparams = parse_args_for_config()
    trainer = ChatbotTrainer(hparams)
    trainer.train()
