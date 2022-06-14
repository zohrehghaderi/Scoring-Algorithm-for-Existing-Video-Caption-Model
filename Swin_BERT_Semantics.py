from pytorch_lightning.utilities.types import EVAL_DATALOADERS

from Mlpmax import MLP
import torch.nn as nn
import pytorch_lightning as pl
from typing import Union, List, Any, Callable, Optional
import torch
from torch.utils.data import DataLoader
from Encoder import swin_encoder
Prepare_inputs_sos import SOSSwinBert
from transformers import BertTokenizer, BertConfig
from NLP_metrics import convert_list_to_string, nlp_metric_bert


PAD_token = 0
EOS_token = 2
SOS_token = 3

beam_num = 3


class Swin_BERT_Semantics(pl.LightningModule):

    def __init__(self, swin_freeze, in_size, hidden_sizes, out_size, drop_swin, max_length,
    mlp_freeze, drop_mlp, drop_bert, checkpoint_encoder):
        super(Swin_BERT_Semantics, self).__init__()
       

             
        self.max_length = max_length
        

        #Swin-Encoder
        self.checkpoint_encoder = checkpoint_encoder
        self.encoder = swin_encoder(device=self.device, drop=self.drop_swin, checkpoint_encoder=self.checkpoint_encoder)
        self.encoder_change = nn.Linear(1024, 768)
        if swin_freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False


        #BERT-Decoder
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained("bert-base-uncased", bos_token_id=101, pad_token_id=0, eos_token_id=102)
        config.is_decoder = True
        config.add_cross_attention = True
        config.hidden_dropout_prob = drop_bert
        config.attention_probs_dropout_prob = drop_bert
        self.decoder = SOSSwinBert.from_pretrained('bert-base-uncased', config=config)

        #Semantics-Network
        self.net = MLP(in_size=in_size, hidden_sizes=hidden_sizes, out_size=out_size,
                       dropout_p=self.drop_mlp, have_last_bn=True)
        
        if mlp_freeze:
            for param in self.net.parameters():
                param.requires_grad = False

 

    def forward(self, enc_inputs):
        self.encoder.eval()
        self.net.eval()
        self.decoder.eval()

        batch_size = enc_inputs.shape[0]
        #encode video
        enc_outputs = self.encoder(enc_inputs)
        enc_outputs_vision = self.encoder_change(enc_outputs)
        semantics = self.net(enc_outputs)

        #prepare SOS, ourput encoder to generate text
        seq_input = torch.zeros(batch_size, 1, dtype=torch.int, device=self.device)
        seq_input[:, 0] = self.decoder.config.bos_token_id
        expanded_return_idx = (torch.arange(seq_input.shape[0]).view(-1, 1).repeat(1, beam_num).view(-1).to(self.device))
        encoder_hidden_states = enc_outputs_vision.index_select(0, expanded_return_idx)
        mask_p = torch.ones(batch_size, 1, dtype=torch.int, device=self.device)
        semantics = semantics.index_select(0, expanded_return_idx)
        semantics = semantics.unsqueeze(1)
        model_kwargs = {"encoder_hidden_states": encoder_hidden_states, "inputs_embeds": semantics,"attention_mask": mask_p}

        #generate text
        outputs = self.decoder.generate(input_ids=seq_input,
                                        bos_token_id=self.decoder.config.bos_token_id,
                                        eos_token_id=self.decoder.config.eos_token_id,
                                        pad_token_id=self.decoder.config.pad_token_id,
                                        max_length=self.max_length,
                                        num_beams=beam_num,
                                        num_return_sequences=1, **model_kwargs)

        return outputs

    