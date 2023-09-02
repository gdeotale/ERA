from model import build_transformer 
from dataset import BilingualDataset, causal_mask 
from config import get_config, get_weights_file_path

import torchtext.datasets as datasets 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split 
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm 
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset 
from tokenizers import Tokenizer 
from tokenizers.models import WordLevel 
from tokenizers.trainers import WordLevelTrainer 
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
# from torch.utils.tensorboard import Summarywriter
from tensorboardX import SummaryWriter

class Lit_translation(LightningModule):
    def __init__(self, data_dir='PASCAL_VOC', hidden_size=16, learning_rate=2e-4):

        super().__init__()

        # Build tokenizers
        self.ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
        self.tokenizer_src = get_or_build_tokenizer(config, self.ds_raw, config['lang_src'])
        self.tokenizer_tgt = get_or_build_tokenizer(config, self.ds_raw, config['lang_tgt'])
        self.vocab_src_len=self.tokenizer_src.get_vocab_size()
        self.vocab_tgt_len=self.tokenizer_tgt.get_vocab_size()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)
        


    def forward(self, x):
        model = build_transformer(self.vocab_src_len, self.vocab_tgt_len, config['seq_len'], config['seq_len'], d_model=config['d_model'])
        return model

    def training_step(self, batch, batch_idx):
        encoder_input = batch['encoder_input'] # (b, seq_1en)
        decoder_input = batch['decoder_input'] # (B, seq_len)
        encoder_mask = batch['encoder_mask'] # (B, 1, 1, seq_Len)
        decoder_mask = batch['decoder_mask'] #- (B, 1, seq Len, seq_len)

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = model.encode(encoder_input, encoder_mask) # (B, sea_len, a _model)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        
        proj_output = model.project(decoder_output) # (B, seqlen, vocab_size)

        # Compare the output with the label
        label = batch['label']

        # Compute the loss using a simple cross entropy
        loss = self.loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))


        return loss

    def validation_step(self, batch, batch_idx):
        self.batch = batch
        self.encoder_input = batch['encoder_input'] # (b, seq_1en)
        decoder_input = batch['decoder_input'] # (B, seq_len)
        self.encoder_mask = batch['encoder_mask'] # (B, 1, 1, seq_Len)
        decoder_mask = batch['decoder_mask'] #- (B, 1, seq Len, seq_len)

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = model.encode(self.encoder_input, self.encoder_mask) # (B, sea_len, a _model)
        decoder_output = model.decode(encoder_output, self.encoder_mask, decoder_input, decoder_mask)
        
        proj_output = model.project(decoder_output) # (B, seqlen, vocab_size)

        # Compare the output with the label
        label = batch['label']

        # Compute the loss using a simple cross entropy
        loss = self.loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
        )

        self.log("val_loss", loss, prog_bar=True)
        garbage_collection_cuda()
        return loss

    def on_validation_epoch_end(self):
        # Get the learning rate from the optimizer
        optimizer = self.optimizers()
        epoch = self.current_epoch
    
        if config.SAVE_MODEL:
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)


        # check that the batch size is 1
        assert encoder_input.size(
            0) == 1, "Batch size must be 1 for validation"

        model_out = greedy_decode(model, self.encoder_input, self.encoder_mask, self.tokenizer_src, self.tokenizer_tgt, max_len, device)

        source_text = self.batch["src_text"][0]
        target_text = self.batch["tgt_text"][0]
        model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())
        
        source_texts.append(source_text)
        expected.append(target_text)
        predicted.append (model_out_text)

        # Print the source, target and model output
        print_msg('-'*console_width)
        print_msg(f"{f'SOURCE: ':>12}{source_text}")
        print_msg(f"{f'TARGET: ':>12}{target_text}")
        print_msg (f"{f'PREDICTED: ':>12}{model_out_text}")
        garbage_collection_cuda()


    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
        
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################
    def setup(self, stage=None):
        
        # Keep 90% for training, 10% for validation
        train_ds_size = int(0.9 * len(self.ds_raw))
        val_ds_size = len(self.ds_raw) - self.train_ds_size
        train_ds_raw, val_ds_raw = random_split(self.ds_raw, [self.train_ds_size, self.val_ds_size])
    
        self.train_ds = BilingualDataset(train_ds_raw, self.tokenizer_src, self.tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len']) 
        self.val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
        
        # Find the maximum length of each sentence in the source and target sentence
        max_len_src = 0
        max_len_tgt = 0

        for item in ds_raw:
            src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
            tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
        print(f'Max length of source sentence: {max_len_src}')
        print(f'Max length of target sentence: {max_len_tgt}')

    def train_dataloader(self):

        return DataLoader(self.train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=config.NUM_WORKERS)

    def val_dataloader(self):

        return DataLoader(self.val_ds, batch_size=1, shuffle=True, num_workers=config.NUM_WORKERS)

    def test_dataloader(self):

        return DataLoader(self.val_ds, batch_size=1, shuffle=True, num_workers=config.NUM_WORKERS)

    def get_optimizer(self):
      return self.optimizers()