import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset

import argparse
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from transformers import T5ForConditionalGeneration, T5Tokenizer

from collator import monoT5Collator
from dataset import RerankerDataset

import peft

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default='sonoisa/t5-base-japanese', type=str, required=False,
                        help="Base model to fine tune.")
    parser.add_argument("--triples_path", default=None, type=str, required=False,
                        help="Triples.tsv path")
    parser.add_argument("--output_model_path", default=None, type=str, required=True,
                        help="Path for trained model and checkpoints.")
    parser.add_argument("--save_every_n_steps", default=0, type=int, required=False,
                        help="Save every N steps. (recommended 10000)")
    parser.add_argument("--logging_steps", default=100, type=int, required=False,
                        help="Logging steps parameter.")
    parser.add_argument("--per_device_train_batch_size", default=8, type=int, required=False,
                        help="Per device batch size parameter.")
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int, required=False,
                        help="Gradient accumulation parameter.")
    parser.add_argument("--learning_rate", default=3e-4, type=float, required=False,
                        help="Learning rate parameter.")
    parser.add_argument("--epochs", default=10, type=int, required=False,
                        help="Number of epochs to train")

    parser.add_argument("--max_steps", default=None, type=int, required=False,
                        help="Number of epochs to train")
    
    parser.add_argument("--dataset_name", default=None, type=str, required=True,
                        help="Triples.tsv path")

    parser.add_argument("--local_data", action='store_true',
                        help="using local data")

    parser.add_argument("--dataset_config", default=None, type=str, required=False,
                        help="Triples.tsv path")

    parser.add_argument("--dataset_split", default='train', type=str, required=False,
                        help="Triples.tsv path")

    device = torch.device('cuda')
    torch.manual_seed(123)
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.base_model)

    model = T5ForConditionalGeneration.from_pretrained(args.base_model)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    peft_config = peft.LoraConfig(
        task_type=peft.TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        inference_mode=False,
    )

    model = peft.get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset_train = RerankerDataset(args)

    if args.save_every_n_steps:
        steps = args.save_every_n_steps
        strategy = 'steps'
    else:
        steps = 1
        strategy = 'epoch'

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_path,
        do_train=True,
        save_strategy=strategy,
        save_steps=steps, 
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=5e-5,
        num_train_epochs=1,
        warmup_steps=1000,
        adafactor=True,
        seed=1,
        # disable_tqdm=False,
        load_best_model_at_end=False,
        predict_with_generate=True,
        dataloader_pin_memory=False,
        fp16=False,
        bf16=True,
        max_steps=args.max_steps,
        gradient_checkpointing=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset_train,
        tokenizer=tokenizer,
        data_collator=monoT5Collator(args,tokenizer),
    )

    trainer.train()

    trainer.save_model(args.output_model_path)
    trainer.save_state()

if __name__ == "__main__":
    main()
