'''
for inference only
'''
from collections import OrderedDict
from torch import nn
import torch
from .image_encoder_decoder import Encoder, Decoder
from .lookup_free_quantize import LFQ


class VQModel(nn.Module):

    def __init__(self,
                 resolution=128,
                 ### Quantize Related
                 n_embed=262144,
                 embed_dim=18,
                 sample_minimization_weight=1.0,
                 batch_maximization_weight=1.0,
                 ckpt_path=None,
                 ignore_keys=[],
                 use_ema = False,
                 token_factorization = False,
                 ):
        super().__init__()
        ddconfig = {
            "double_z": False,
            "z_channels": 18,
            "resolution": resolution,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1,2,2,4],  # num_down = len(ch_mult)-1
            "num_res_blocks": 2,
        }
        if ckpt_path and '256' in ckpt_path:
            ddconfig['resolution'] = 256
        self.use_ema = use_ema
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = LFQ(dim=embed_dim, codebook_size=n_embed, 
                            sample_minimization_weight=sample_minimization_weight, 
                            batch_maximization_weight=batch_maximization_weight,
                            token_factorization = token_factorization)
       
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=None)
    
    def init_from_ckpt(self, path, ignore_keys=list(), stage=None):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        ema_mapping = {}
        new_params = OrderedDict()
        if stage == "transformer": ### directly use ema encoder and decoder parameter
            if self.use_ema:
                for k, v in sd.items(): 
                    if "encoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v   
                        s_name = k.replace('.', '')
                        ema_mapping.update({s_name: k})
                        continue
                    if "decoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v 
                        s_name = k.replace(".", "")
                        ema_mapping.update({s_name: k})
                        continue 
            else: #also only load the Generator
                for k, v in sd.items():
                    if "encoder" in k:
                        new_params[k] = v
                    elif "decoder" in k:
                        new_params[k] = v
            missing_keys, unexpected_keys = self.load_state_dict(new_params, strict=False)
        else: ## simple resume
            missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        (quant, emb_loss, info) = self.quantize(h, return_loss_breakdown=False, return_loss=False)
        ### using token factorization the info is a tuple (each for embedding)
        return quant, emb_loss, info

    def decode(self, quant):
        dec = self.decoder(quant)
        return dec

    def forward(self, input):
        quant, diff, _, = self.encode(input)
        # print(quant)
        # print(f'quant: {quant.shape}, diff: {diff.shape}')
        dec = self.decode(quant)
        return dec
