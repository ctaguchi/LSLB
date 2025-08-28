# This program is the latest as of July 25, 2024.
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from transformers.optimization import Adafactor
import re
import os
import yaml
import json
from typing import Tuple, List, Union, Dict, Literal
import pandas as pd
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import sacrebleu
import evaluate
from mbrs.metrics import MetricBLEU
from mbrs.decoders import DecoderMBR
import wandb
wandb.login()

# local
import utils

def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="600m",
                        help="Specify the pretrained NLLB model.")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help="Batch size during the training.")
    parser.add_argument("-s", "--src_lang", type=str, default="eng_Latn",
                        help="The language code of the source language.")
    parser.add_argument("-t", "--tgt_lang", type=str, default="kac_Latn",
                        help="The language code of the target language.")
    parser.add_argument("-e", "--num_epochs", type=int, default=10,
                        help="Number of epochs to run.")
    parser.add_argument("-d", "--data", nargs="+",
                        help="The datasets to be used during the training. paradisec, nllb, dict")
    parser.add_argument("--eval_data", nargs="+",
                        help="The evaluation datasets to be used for validation. paradisec, flores, conv")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--steps", type=int, default=100_000,
                        help="Number of steps to run.")
    parser.add_argument("--validation_method", type=str, default="epoch",
                        help="When to run validation. `epoch` or `steps` (every 1k steps).")
    parser.add_argument("--max_tgt_token_length", type=int, default=30,
                        help="Max token length for training samples of the target language.")
    parser.add_argument("--max_src_token_length", type=int, default=30,
                        help="Max token length for training samples of the source language.")
    parser.add_argument("--max_src_subtok_length", type=int, default=35,
                        help="Max subword token length for training samples of the source language.")
    parser.add_argument("--training_scheme", default="steps",
                        help="Training scheme. `steps` or `epochs`.")
    parser.add_argument("--model_save_path", type=str, required=True,
                        help="Path to save the trained model.")
    parser.add_argument("--wandb_project_name", type=str, default="nllb-kac")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--multi_gpu", action="store_true", help="data parallelism has not been tested yet")
    parser.add_argument("--lr_scheduler", default=None, help="Options: `multiplicativelr`")
    parser.add_argument("--multiplicativelr_lambda", type=float, default=0.99)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--num_saved_checkpoint", type=int, default=2)
    parser.add_argument("--bidirectional", action="store_true",
                        help="If true, the training goes through both translation directions.")
    parser.add_argument("--ddp", action="store_true",
                        help="If true, it'll be multi-GPU training with Distributed Data Parallel.")
    parser.add_argument("--japanese", action="store_true",
                        help="if true, Jinghpaw<->Japanese training data will be added.")
    parser.add_argument("--mbr", action="store_true",
                        help="If true, MBR decoding will be used for translation generation.")
    parser.add_argument("--generation_config", type=str,
                        help="Generation config file for MBR.")
    parser.add_argument("--curriculum_learning", action="store_true",
                        help="Load the datasets for curriculum learning.")
    args = parser.parse_args()
    return args


class MTTrainer():
    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 batch_size: int,
                 num_epochs: int,
                 train_dataloaders: List[DataLoader],
                 eval_dataloaders: Dict[str, DataLoader],
                 model,
                 tokenizer,
                 optimizer,
                 scheduler,
                 model_save_path: str,
                 warmup: bool = False,
                 warmup_steps: int = 0,
                 warmup_scheduler = None,
                 early_stopping: bool = False,
                 early_stopper = None,
                 max_length: int = 128,
                 max_translation_input_length: int = 1024,
                 num_beams: int = 4,
                 mbr: bool = None,
                 decoder: DecoderMBR = None,
                 generation_config: str = None,
                 validation_method: Literal["steps", "epoch"] = "steps",
                 wandb_run = None,
                 ddp: bool = False):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_dataloaders = train_dataloaders
        self.eval_dataloaders = eval_dataloaders
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_save_path = model_save_path
        self.warmup = warmup
        self.warmup_steps = warmup_steps
        self.warmup_scheduler = warmup_scheduler
        self.early_stopping = early_stopping
        self.early_stopper = early_stopper
        self.max_length = max_length
        self.max_translation_input_length = max_translation_input_length
        self.num_beams = num_beams
        self.mbr = mbr
        self.decoder = decoder
        self.generation_config = generation_config
        self.validation_method = validation_method
        self.wandb_run = wandb_run
        self.ddp = ddp
        
        self.losses = []
        self.step = 0
        self.epoch = 1
        self.min_devloss = float("inf")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.chrf_calc = sacrebleu.CHRF(word_order=2)
        self.bleu_calc = sacrebleu.BLEU()
        self.meteor_calc = evaluate.load("meteor")

    def train(self) -> None:
        """Training."""
        self.model.train()
        for i in range(self.num_epochs):
            self.training_epoch_loop()
            self.epoch += 1

    def ddp_train(self, rank, world_size) -> None:
        """Multi-GPU training with Distributed Data Parallel."""
        utils.ddp_setup(rank, world_size)
        model = self.model.to(rank)
        model = DDP(model, device_ids=[rank])
        optimizer = Adafactor(
            [p for p in model.parameters() if p.requires_grad],
            scale_parameter=False,
            relative_step=False,
            lr=1e-4,
            clip_threshold=1.0,
            weight_decay=1e-3,
        )
        for i in range(args.num_epochs):
            training_epoch_loop(i)

    def training_epoch_loop(self):
        """A training loop."""
        print(f"Epoch {self.epoch}")
        self.model.train()
        step_per_epoch = 0
        x, y, loss = None, None, None
        if isinstance(self.train_dataloaders, list):
            for dataloaders in self.train_dataloaders:
                # `dataloaders` contain DataLoaders from different language pairs(List[DataLoader])
                for dataloader in dataloaders:
                    for xx, yy in dataloader:
                        # the first 8 characters express a lang code as defined in `utils.CustomDataset`.
                        self.train_batch_step(xx, yy)
                        step_per_epoch += 1

                        if self.validation_method == "steps" and self.step % 1000 == 0:
                            self.validation(self.step)
        elif isinstance(self.train_dataloaders, DataLoader):
            for xx, yy in self.train_dataloaders:
                self.train_batch_step(xx, yy)
                step_per_epoch += 1
                if self.validation_method == "steps" and self.step % 1000 == 0:
                    self.validation(self.step)
        else:
            raise TypeError("Unknown type for `train_dataloaders`", type(train_dataloaders))        
                
        # The end of an epoch; slightly decrease the learning rate
        if self.warmup:
            with self.warmup_scheduler.dampening():
                if self.warmup_scheduler.last_step + 1 > self.warmup_steps:
                    self.scheduler.step()
        else:
            if self.scheduler:
                self.scheduler.step()        

        if self.validation_method == "epoch":
            self.validation(self.epoch)

        checkpoint_save_path = os.path.join(self.model_save_path, "checkpoint-" + str(self.step))
        if not os.path.exists(checkpoint_save_path):
            os.makedirs(checkpoint_save_path)
        self.model.eval()
        self.log_translation(checkpoint_save_path)

    def train_batch_step(self, xx, yy) -> None:
        """A training step per batch."""
        try:
            self.tokenizer.src_lang = xx[0][:8]
            xx = [x[8:] for x in xx] # a batch
            x = self.tokenizer(xx,
                               return_tensors="pt",
                               padding=True,
                               truncation=True,
                               max_length=self.max_length).to(self.device)
            self.tokenizer.src_lang = yy[0][:8]
            yy = [y[8:] for y in yy] # batch
            y = self.tokenizer(yy,
                               return_tensors="pt",
                               padding=True,
                               truncation=True,
                               max_length=self.max_length).to(self.device)
            y.input_ids[y.input_ids == self.tokenizer.pad_token_id] = -100
            output = self.model(**x, labels=y.input_ids)
            loss = output.loss
            loss.backward()
            self.losses.append(loss.item())
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.ddp:
                utils.ddp_cleanup()
                
            if self.warmup:
                with self.warmup_scheduler.dampening():
                    pass
                
            self.wandb_run.log({"train-loss": loss,
                                "learning-rate": self.optimizer.param_groups[0]["lr"],
                                "epoch": self.epoch})
            self.step += 1

        except (ValueError, TypeError, RuntimeError) as e:
            self.optimizer.zero_grad(set_to_none=True)
            x, y, loss = None, None, None
            utils.cleanup()
            print("error", e)
            pass

    def validation(self,
                   progress: int) -> None:
        """Validation.
        args:
        - progress (int): the index of the current step or epoch.
        """
        self.model.eval()
        if self.validation_method == "epoch":
            print(f"Evaluating epoch {progress}...")
        elif self.validation_method == "steps":
            print(f"Evaluating step {progress}...")
        else:
            raise NotImplementedError

        checkpoint_save_path = os.path.join(self.model_save_path, "checkpoint-" + str(self.step))
        if not os.path.exists(checkpoint_save_path):
            os.makedirs(checkpoint_save_path)

        devloss = 0.0
        for dataloader in self.eval_dataloaders.values():
            for xx, yy in dataloader:
                xx = [x[8:] for x in xx if x.startswith(self.src_lang)]
                yy = [y[8:] for y in yy if y.startswith(self.tgt_lang)]
                loss = self.calc_validation_loss(xx, yy)
                devloss += loss.item()
        print("devloss:", devloss)
        self.wandb_run.log({"eval-loss": devloss})

        if self.min_devloss <= devloss:
            if self.early_stopping and self.early_stopper.early_stop(devloss):
                print("Validation loss worsened. Early-stopping.")
                exit()
        else: # new record
            self.min_devloss = devloss
            self.model.save_pretrained(checkpoint_save_path)
            self.tokenizer.save_pretrained(checkpoint_save_path)
            if self.ddp:
                dist.barrier()

    def log_translation(self, checkpoint_save_path: str) -> None:
        """Log translation results."""
        stats = dict()
        translations = dict()
        for name, dl in self.eval_dataloaders.items():
            print("Evaluating on", name)
            chrf, bleu, meteor, preds = self.dev_eval(dl)
            stats[name] = {"chrf": chrf,
                           "bleu": bleu,
                           "meteor": meteor}
            translations[name] = preds

        with open(os.path.join(checkpoint_save_path, "stats.csv"), "w") as f:
            print("Eval_data,Metric,Score", file=f)
            for name, score_dict in stats.items():
                for metric, score in score_dict.items():
                    print(f"{name},{metric},{score}", file=f)
                    self.wandb_run.log({f"eval-{name}-{metric}": score})
        if self.tgt_lang == "kac_Latn":
            suffix = "kac"
        elif self.tgt_lang == "eng_Latn":
            suffix = "en"
        else:
            raise NotImplementedError

        for name, preds in translations.items():
            with open(os.path.join(f"{checkpoint_save_path}",
                                   f"{name}_translations.{suffix}"),
                      "w") as f:
                for p in preds:
                    print(p, file=f)

    def dev_eval(self, dataloader: DataLoader) -> Tuple[float, float, float, list]:
        preds = []
        refs = []
        for xx, yy in dataloader:
            xx = [x[8:] for x in xx if x.startswith(self.src_lang)]
            yy = [y[8:] for y in yy if y.startswith(self.tgt_lang)]
            preds += self.translate(xx)
            refs += yy
        chrf_result = self.chrf_calc.corpus_score(preds, [refs])
        bleu_result = self.bleu_calc.corpus_score(preds, [refs])
        meteor_result: dict = self.meteor_calc.compute(predictions=preds,
                                                       references=[[r] for r in refs])
        return chrf_result.score, bleu_result.score, meteor_result["meteor"].item(), preds

    def translate(self,
                  xx: Union[list, str],
                  a: int = 32,
                  b: int = 3,
                  **kwargs) -> Union[list, str]:
        """Translate a sentence.
        For info on generation config, see https://huggingface.co/docs/transformers/v4.45.1/en/generation_strategies
        """
        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang
        with torch.no_grad():
            inputs = self.tokenizer(xx,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_translation_input_length).to(self.device)
            result = self.model.generate(**inputs,
                                         forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
                                         max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
                                         generation_config=self.generation_config,
                                         **kwargs)
            translation = self.tokenizer.batch_decode(result, skip_special_tokens=True)
            
        if self.mbr:
            # mbr decoding; `translation` is a list of translations with multiple candidates
            mbr_translation = []
            hypotheses = [translation[i:i+self.batch_size] for i in range(0, len(translation), self.batch_size)]
            for source, hyps in zip(xx, hypotheses):
                output = decoder.decode(hyps, hyps, source=source, nbest=1)
                mbr_translation.append(output)
        else:
            return translation

    def calc_validation_loss(self,
                             xx: Union[str, list],
                             yy: Union[str, list]) -> float:
        """Compute the validation loss."""
        with torch.no_grad():
            self.tokenizer.src_lang = self.src_lang
            inputs = self.tokenizer(xx,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_translation_input_length).to(self.device)
            self.tokenizer.src_lang = self.tgt_lang
            y = self.tokenizer(yy,
                               return_tensors="pt",
                               padding=True,
                               truncation=True,
                               max_length=self.max_translation_input_length).to(self.device)
            y.input_ids[y.input_ids == self.tokenizer.pad_token_id] = -100
            output = self.model(**inputs, labels=y.input_ids)
        return output.loss

model_map = {"600m": "facebook/nllb-200-distilled-600M",
             "1.3b": "facebook/nllb-200-1.3B",
             "3.3b": "facebook/nllb-200-3.3B"}

def main():
    args = get_args()
    print("Loading models...")
    tokenizer = NllbTokenizer.from_pretrained(model_map[args.model])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_map[args.model])
    generation_config = GenerationConfig.from_pretrained(model_map[args.model])
    if args.mbr:
        with open(args.generation_config, "r") as f:
            mbr_gc = json.load(f)
        generation_config.update(**mbr_gc)
        mbr_cfg = MetricBLEU.Config()
        mbr_metric = MetricBLEU(mbr_cfg)
        decoder_cfg = DecoderMBR.Config()
        mbr_decoder = DecoderMBR(decoder_cfg, mbr_metric)
    print("Model loaded")

    print("Loading data...")
    datafetcher = utils.DataFetcher(args.src_lang,
                                    args.tgt_lang,
                                    args.batch_size,
                                    args.japanese,
                                    args.bidirectional,
                                    args.curriculum_learning)
    train_data, dev_data = datafetcher.fetch_data(args.data)
    train_dataloader, dev_dataloaders = datafetcher.get_dataloaders(train_data, dev_data) # Tuple[DataLoader, Dict[str, DataLoader]]
    print("Data loaded")

    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
        )
    if args.lr_scheduler == "multiplicativelr":
        lmbda = lambda epoch: args.multiplicativelr_lambda
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    else:
        scheduler = None

    if args.warmup:
        import pytorch_warmup as warmup
        if args.warmup_steps is None:
            if args.bilingual:
                args.warmup_steps = int((len(train_df) * 2 / args.batch_size) // 2)
            else:
                args.warmup_steps = int((len(train_df) / args.batch_size) // 2)
        warmup_scheduler = warmup.LinearWarmup(optimizer, args.warmup_steps)
    else:
        warmup_scheduler = None
        # raise NotImplementedError
    if args.early_stopping:
        early_stopper = utils.EarlyStopper(patience=3, min_delta=10)
    else:
        early_stopper = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    wandb_run = wandb.init(project=args.wandb_project_name,
                     config={"learning-rate": args.learning_rate,
                             "epochs": args.num_epochs})
    if args.ddp:
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        mp.spawn(ddp_train,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
    print("Starting training...")
    trainer = MTTrainer(src_lang=args.src_lang,
                        tgt_lang=args.tgt_lang,
                        batch_size=args.batch_size,
                        num_epochs=args.num_epochs,
                        train_dataloaders=train_dataloader,
                        eval_dataloaders=dev_dataloaders,
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        model_save_path=args.model_save_path,
                        warmup=args.warmup,
                        warmup_steps=args.warmup_steps,
                        warmup_scheduler=warmup_scheduler,
                        early_stopping=args.early_stopping,
                        early_stopper=early_stopper,
                        mbr=args.mbr,
                        generation_config=generation_config,
                        validation_method=args.validation_method,
                        wandb_run=wandb_run,
                        ddp=args.ddp)
    print("Training started")
    trainer.train()

if __name__ == "__main__":
    main()
