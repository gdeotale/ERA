from model import build_transformer 
from dataset import BilingualDataset, causal_mask 
from config import get_config, get_weights_file_path
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.memory import garbage_collection_cuda

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
#from torch.utils.tensorboard import Summarywriter
#from tensorboardX import SummaryWriter

def get_all_sentences(ds, lang):
        for item in ds:
            yield item['translation'][lang]

def get_or_build_tokenizer (config, ds, lang):
        tokenizer_path = Path(config['tokenizer_file'].format(lang))
        if not Path.exists(tokenizer_path):
            # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
            tokenizer = Tokenizer(WordLevel (unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
            tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer
        
def greedy_decode (model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode (source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item())], dim=1
            )
        
        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)

class lit_module(LightningModule):
    def __init__(self, data_dir='PASCAL_VOC', hidden_size=16, learning_rate=2e-4):

        super().__init__()
        self.config = get_config()
        # Build tokenizers
        self.ds_raw = load_dataset('opus_books', f"{self.config['lang_src']}-{self.config['lang_tgt']}", split='train')
        self.tokenizer_src = get_or_build_tokenizer(self.config, self.ds_raw, self.config['lang_src'])
        self.tokenizer_tgt = get_or_build_tokenizer(self.config, self.ds_raw, self.config['lang_tgt'])
        self.vocab_src_len=self.tokenizer_src.get_vocab_size()
        self.vocab_tgt_len=self.tokenizer_tgt.get_vocab_size()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)
        
        self.model = build_transformer(self.vocab_src_len, self.vocab_tgt_len, self.config['seq_len'], self.config['seq_len'], d_model=self.config['d_model'])
        
        
    def forward(self, x):
        model = self.model()
        return model

    def training_step(self, batch, batch_idx):
        encoder_input = batch['encoder_input'] # (b, seq_1en)
        decoder_input = batch['decoder_input'] # (B, seq_len)
        encoder_mask = batch['encoder_mask'] # (B, 1, 1, seq_Len)
        decoder_mask = batch['decoder_mask'] #- (B, 1, seq Len, seq_len)

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = self.model.encode(encoder_input, encoder_mask) # (B, sea_len, a _model)
        decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        
        proj_output = self.model.project(decoder_output) # (B, seqlen, vocab_size)

        # Compare the output with the label
        label = batch['label']

        # Compute the loss using a simple cross entropy
        loss = self.loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))


        return loss

    def validation_step(self, batch, batch_idx):
        self.batch = batch
        self.encoder_input = batch['encoder_input'] # (b, seq_1en)
        decoder_input = batch['decoder_input'] # (B, seq_len)
        self.encoder_mask = batch['encoder_mask'] # (B, 1, 1, seq_Len)
        decoder_mask = batch['decoder_mask'] #- (B, 1, seq Len, seq_len)

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = self.model.encode(self.encoder_input, self.encoder_mask) # (B, sea_len, a _model)
        decoder_output = self.model.decode(encoder_output, self.encoder_mask, decoder_input, decoder_mask)
        
        proj_output = self.model.project(decoder_output) # (B, seqlen, vocab_size)

        # Compare the output with the label
        label = batch['label']

        # Compute the loss using a simple cross entropy
        loss = self.loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))

        self.log("val_loss", loss, prog_bar=True)
        garbage_collection_cuda()
        return loss

    def on_validation_epoch_end(self):
        # Get the learning rate from the optimizer
        optimizer = self.optimizers()
        epoch = self.current_epoch
    
        model_filename = get_weights_file_path(self.config, f"{epoch:02d}")
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_filename)


        # check that the batch size is 1
        assert self.encoder_input.size(
            0) == 1, "Batch size must be 1 for validation"

        model_out = greedy_decode(self.model, self.encoder_input, self.encoder_mask, self.tokenizer_src, self.tokenizer_tgt, self.config['seq_len'])

        source_text = self.batch["src_text"][0]
        target_text = self.batch["tgt_text"][0]
        model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())
        
        #source_texts.append(source_text)
        #expected.append(target_text)
        #predicted.append (model_out_text)
        # Print the source, target and model output
        #print_msg('-'*console_width)
        print("Source:"+source_text)
        print("TARGET:"+target_text)
        print("PREDICTED:"+model_out_text)
        garbage_collection_cuda()


    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], eps=1e-9)
        
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################
    def setup(self, stage=None):
        
        # Keep 90% for training, 10% for validation
        train_ds_size = int(0.9 * len(self.ds_raw))
        val_ds_size = len(self.ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(self.ds_raw, [train_ds_size, val_ds_size])
    
        self.train_ds = BilingualDataset(train_ds_raw, self.tokenizer_src, self.tokenizer_tgt, self.config['lang_src'], self.config['lang_tgt'], self.config['seq_len']) 
        self.val_ds = BilingualDataset(val_ds_raw, self.tokenizer_src, self.tokenizer_tgt, self.config['lang_src'], self.config['lang_tgt'], self.config['seq_len'])
        
        # Find the maximum length of each sentence in the source and target sentence
        max_len_src = 0
        max_len_tgt = 0

        for item in self.ds_raw:
            src_ids = self.tokenizer_src.encode(item['translation'][self.config['lang_src']]).ids
            tgt_ids = self.tokenizer_tgt.encode(item['translation'][self.config['lang_tgt']]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
        print(f'Max length of source sentence: {max_len_src}')
        print(f'Max length of target sentence: {max_len_tgt}')

    def train_dataloader(self):

        return DataLoader(self.train_ds, batch_size=self.config['batch_size'], shuffle=True, num_workers=8)

    def val_dataloader(self):

        return DataLoader(self.val_ds, batch_size=1, shuffle=True, num_workers=8)

    def test_dataloader(self):

        return DataLoader(self.val_ds, batch_size=1, shuffle=True, num_workers=48)

    def get_optimizer(self):
      return self.optimizers()