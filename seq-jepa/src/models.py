
from utils import off_diagonal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import torchvision
import math


class TransformerEncoder_(nn.Module):
    def __init__(self, emb_dim, num_heads, num_enc_layers, mlp_ratio=4, post_norm=True, return_attention_weights=False):
        super(TransformerEncoder_, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * mlp_ratio,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)
        self.post_norm = post_norm
        self.return_attention_weights = return_attention_weights
        if self.post_norm:
            self.norm = nn.LayerNorm(emb_dim)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attention_weights = None
        for i, layer in enumerate(self.transformer_encoder.layers):
            if i == len(self.transformer_encoder.layers) - 1 and self.return_attention_weights:
                src, attention_weights = self._extract_attention_weights(layer, src, src_mask, src_key_padding_mask)
            else:
                src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        if self.post_norm:
            src = self.norm(src)
        return src, attention_weights
    
    def _extract_attention_weights(self, layer, src, src_mask, src_key_padding_mask):
        attn_output, attn_output_weights = layer.self_attn(src, src, src, 
                                                           attn_mask=src_mask, 
                                                           key_padding_mask=src_key_padding_mask,
                                                           need_weights=True,
                                                           average_attn_weights=False)
        
        x = src
        if layer.norm_first:
            x = x + layer.dropout1(attn_output)
            x = x + layer._ff_block(layer.norm2(x))
        else:
            x = layer.norm1(x + layer.dropout1(attn_output))
            x = layer.norm2(x + layer._ff_block(x))
        return x, attn_output_weights



###### SeqJEPA_PLS ######
class SeqJEPA_PLS(nn.Module):
    def __init__(self, fovea_size, img_size, ema, ema_decay=0.996, **kwargs):
        super().__init__()
        self.fovea_size = fovea_size
        self.img_size = img_size
        self.ema = ema
        self.ema_decay = ema_decay
        self.n_channels = kwargs["n_channels"]
        self.num_classes = kwargs["num_classes"]
        self.num_heads = kwargs["num_heads"]
        self.num_enc_layers = kwargs["num_enc_layers"]
        self.action_projdim = kwargs["act_projdim"]
        self.action_latentdim = kwargs["act_latentdim"]
        self.act_cond = kwargs["act_cond"]
        self.learn_act_emb = kwargs["learn_act_emb"]
        self.cifar_resnet = kwargs["cifar_resnet"]
        self.criterion = nn.CosineSimilarity(dim=1)

        self.encoder = torchvision.models.resnet18(pretrained=False, zero_init_residual=True)
        self.res_out_dim = self.encoder.fc.in_features
        self.encoder.fc = torch.nn.Identity()
        
        if self.cifar_resnet:
            self.encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = torch.nn.Identity()
        if self.ema:
            self.target_encoder = copy.deepcopy(self.encoder)
            for param in self.target_encoder.parameters():
                param.requires_grad = False
        else:
            print("Not using EMA")
        
        if self.learn_act_emb:
            self.action_proj = nn.Sequential(
                nn.Linear(self.action_latentdim, self.action_projdim, bias=False), torch.nn.BatchNorm1d(self.action_projdim, affine=False))
            self.emb_dim = self.res_out_dim + self.action_projdim
        else:
            print("Not learning action embeddings")
            self.emb_dim = self.res_out_dim
        
        self.pred_hidden = kwargs["pred_hidden"]
        if self.learn_act_emb and self.act_cond:
            if self.pred_hidden <= 0:
                self.pred_hidden = self.res_out_dim
            self.predictor = nn.Sequential(
                nn.Linear(self.emb_dim+self.action_projdim, self.pred_hidden,bias=False),
                nn.BatchNorm1d(self.pred_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.pred_hidden, self.res_out_dim),)
        else:
            self.emb_dim = self.res_out_dim
            if self.pred_hidden <= 0:
                self.pred_hidden = self.res_out_dim
            self.predictor = nn.Sequential(
                nn.Linear(self.emb_dim, self.pred_hidden, bias=False),
                nn.BatchNorm1d(self.pred_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.pred_hidden, self.res_out_dim),)
                
        ### Transformer Encoder
        self.agg_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
        self.transformer_encoder = TransformerEncoder_(self.emb_dim, self.num_heads,
                                                       self.num_enc_layers, mlp_ratio=4, post_norm=True)
        

    def _update_target_network(self):
        for online_params, target_params in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.ema_decay * target_params.data + (1 - self.ema_decay) * online_params.data

    def update_moving_average(self):
        """Update the target network using EMA."""
        self._update_target_network()
    
    def add_probes(self):
        for param in self.parameters():
            param.requires_grad = False
        self.pos_regressor = nn.Linear(self.res_out_dim*2, 2)
        self.agg_classifier = nn.Linear(self.emb_dim, self.num_classes)
        self.res_classifier = nn.Linear(self.res_out_dim, self.num_classes)
    
    def forward(self, fov_x_obs, fov_x_last, action_latents):
        ### pred encodings
        num_saccades = fov_x_obs.shape[1] + 1
        fov_x_obs = fov_x_obs.reshape(-1, self.n_channels, self.fovea_size, self.fovea_size)
        fov_x_last = fov_x_last.reshape(-1, self.n_channels, self.fovea_size, self.fovea_size)
        if self.ema:
            fov_encs_last = self.target_encoder(fov_x_last)
            fov_encs_last_detached = fov_encs_last.detach()
        else:
            fov_encs_last = self.encoder(fov_x_last)
            fov_encs_last_detached = fov_encs_last.detach() 
        ### obs encodings
        fov_encs_obs = self.encoder(fov_x_obs)
        fov_encs_obs = fov_encs_obs.reshape((-1, num_saccades-1, self.res_out_dim))
        ### action conditioning
        if self.act_cond:
            act_enc_obs = action_latents[:,:-1,:].reshape(-1, num_saccades-1, self.action_latentdim)
            act_enc_last = action_latents[:,-1,:].reshape(-1, self.action_latentdim)
            relative_act_enc_obs = torch.zeros_like(act_enc_obs)
            relative_act_enc_obs[:,:-1] = act_enc_obs[:,1:] - act_enc_obs[:,:-1]
            act_enc_last = act_enc_last - act_enc_obs[:,-1]
            relative_act_enc_obs = relative_act_enc_obs.reshape(-1, self.action_latentdim)
            relative_act_enc_obs = self.action_proj(relative_act_enc_obs)
            relative_act_enc_obs = relative_act_enc_obs.reshape(-1, num_saccades-1, self.action_projdim)
            relative_act_enc_obs[:,-1,...] = 0.
            fov_encs_obs = torch.cat((fov_encs_obs, relative_act_enc_obs), dim=-1)
            act_enc_last = self.action_proj(act_enc_last)

        fov_encs_reshape = fov_encs_obs.reshape((-1, num_saccades-1, self.emb_dim))
        B, N, _ = fov_encs_reshape.shape
        agg_tokens = self.agg_token.expand(B, -1, -1)

        x = torch.cat((agg_tokens, fov_encs_reshape), dim=1)
        

        x, _ = self.transformer_encoder(x)
            
        agg_out = x[:, 0]
        if self.act_cond:
            agg_out_conditioned = torch.cat((agg_out, act_enc_last), dim=-1)
        else:
            agg_out_conditioned = agg_out
        pred_out = self.predictor(agg_out_conditioned)
        
        loss = 1-self.criterion(pred_out, fov_encs_last_detached).mean()
        z1 = fov_encs_obs[:,0,:self.res_out_dim]
        if num_saccades == 2:
            z2 = fov_encs_last_detached
        else:
            z2 = fov_encs_obs[:,1,:self.res_out_dim]

        return loss, agg_out, z1, z2



####### Seq-JEPA Transforms #######
class SeqJEPA_Transforms(nn.Module):
    def __init__(self, img_size, ema, ema_decay=0.996, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.ema = ema
        self.ema_decay = ema_decay
        self.n_channels = kwargs["n_channels"]
        self.num_classes = kwargs["num_classes"]
        self.num_heads = kwargs["num_heads"]
        self.num_enc_layers = kwargs["num_enc_layers"]
        self.learn_act_emb = kwargs["learn_act_emb"]
        self.act_cond = kwargs["act_cond"]
        self.cifar_resnet = kwargs["cifar_resnet"]
        
        self.criterion = nn.CosineSimilarity(dim=1)

        self.encoder = torchvision.models.resnet18(pretrained=False, zero_init_residual=True)
        self.res_out_dim = self.encoder.fc.in_features
        self.encoder.fc = torch.nn.Identity()
        if self.cifar_resnet:
            self.encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = torch.nn.Identity()
                                                
        if self.ema:
            self.target_encoder = copy.deepcopy(self.encoder)
            for param in self.target_encoder.parameters():
                param.requires_grad = False
        else:
            print("Not using EMA")
        
        if self.act_cond:
            self.action_latentdim = kwargs["act_latentdim"]
            if self.learn_act_emb:
                self.action_projdim = kwargs["act_projdim"]
                self.action_proj = nn.Sequential(nn.Linear(self.action_latentdim, self.action_projdim, bias=False), torch.nn.BatchNorm1d(self.action_projdim, affine=False))                
                self.emb_dim = self.res_out_dim + self.action_projdim
                self.pred_indim = self.emb_dim + self.action_projdim
            else:
                self.emb_dim = self.res_out_dim + self.action_latentdim
                # Ensure emb_dim is divisible by num_heads
                if self.emb_dim % self.num_heads != 0:
                    padding = self.num_heads - (self.emb_dim % self.num_heads)
                    self.emb_dim += padding  # Adjust emb_dim to make it divisible by num_heads
                    self.pred_indim = self.emb_dim + self.action_latentdim + padding
                    print("Padded emb_dim: ", self.emb_dim, "Padded pred_indim: ", self.pred_indim)
                else:
                    self.pred_indim = self.emb_dim + self.action_latentdim
        else:
            self.emb_dim = self.res_out_dim
            self.pred_indim = self.emb_dim
        
        self.pred_hidden = kwargs["pred_hidden"]
        if self.act_cond:
            if self.learn_act_emb:
                pred_in = self.pred_indim
            else:
                pred_in = self.pred_indim
        else:
            pred_in = self.emb_dim
        if self.pred_hidden > 0:
            self.predictor = nn.Sequential(
                nn.Linear(pred_in, self.pred_hidden,bias=False),
                nn.BatchNorm1d(self.pred_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.pred_hidden, self.res_out_dim),)
        else:
            self.predictor = nn.Sequential(
                nn.Linear(pred_in, self.res_out_dim, bias=False), torch.nn.BatchNorm1d(self.res_out_dim))
                
        ### Transformer Encoder
        self.agg_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
        self.transformer_encoder = TransformerEncoder_(self.emb_dim, self.num_heads,
                                                    self.num_enc_layers, mlp_ratio=4, post_norm=True)
        
    
    def _update_target_network(self):
        for online_params, target_params in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.ema_decay * target_params.data + (1 - self.ema_decay) * online_params.data
    def update_moving_average(self):
        """Update the target network using EMA."""
        self._update_target_network()
    
    def add_probes(self, latent_type, val_len=3):
        for param in self.parameters():
            param.requires_grad = False
        dim = self.res_out_dim
        if latent_type == "aug":
            self.blur_regressor = nn.Linear(dim, 1)
            self.crop_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.jitter_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
        elif latent_type == "rotcolor":
            self.rot_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.color_regressor = nn.Linear(dim, 2)
        self.agg_classifier = nn.Linear(self.emb_dim, self.num_classes)
        self.res_classifier = nn.Linear(self.res_out_dim, self.num_classes)
    
    def forward(self, x_obs, x_pred, act_lat_obs, act_lat_pred, rel_latents=None):
        ### encodings
        seq_len = x_obs.shape[1]
        x_obs = x_obs.reshape(-1, self.n_channels, self.img_size, self.img_size)
        enc_obs = self.encoder(x_obs)
        enc_obs = enc_obs.reshape(-1, seq_len, self.res_out_dim)
        x_pred = x_pred.reshape(-1, self.n_channels, self.img_size, self.img_size)
        if self.ema:
            enc_pred = self.target_encoder(x_pred)
            enc_pred_detached = enc_pred.detach() 
            enc_pred_detached = enc_pred_detached.reshape(-1, self.res_out_dim)
        else:
            enc_pred = self.encoder(x_pred)
            enc_pred_detached = enc_pred.detach() 
            enc_pred_detached = enc_pred_detached.reshape(-1, self.res_out_dim)
                
        ### action conditioning
        if self.act_cond:
            act_lat_obs = act_lat_obs.reshape(-1, self.action_latentdim)
            if self.learn_act_emb:
                act_enc_obs = act_lat_obs
                act_enc_pred = act_lat_pred
                act_enc_obs = act_enc_obs.reshape(-1, seq_len, self.action_latentdim)
                relative_act_enc_obs = torch.zeros_like(act_enc_obs)
                if rel_latents is not None:
                    relative_act_enc_obs = rel_latents.reshape(-1, seq_len, self.action_latentdim)
                    act_enc_pred = rel_latents[:,-1]
                else:
                    relative_act_enc_obs[:,:-1] = act_enc_obs[:,1:] - act_enc_obs[:,:-1]
                    # Last action 
                    act_enc_pred = act_enc_pred - act_enc_obs[:,-1]
                act_enc_pred = self.action_proj(act_enc_pred)
                relative_act_enc_obs = relative_act_enc_obs.reshape(-1, self.action_latentdim)
                relative_act_enc_obs = self.action_proj(relative_act_enc_obs)
                relative_act_enc_obs = relative_act_enc_obs.reshape(-1, seq_len, self.action_projdim)
                relative_act_enc_obs[:,-1,...] = 0.
                enc_obs_concat = torch.cat((enc_obs, relative_act_enc_obs), dim=-1)
            else:
                if rel_latents is not None:
                    act_enc_obs = rel_latents.reshape(-1, seq_len, self.action_latentdim)
                    act_enc_obs[:,-1,...] = 0.
                    act_enc_pred = rel_latents[:,-1]                    
                    ### If the action_latentdim is not divisible by num_heads, pad the last dimension
                    if self.action_latentdim % self.num_heads != 0:
                        padding = self.num_heads - (self.action_latentdim % self.num_heads)
                        act_enc_obs = F.pad(act_enc_obs, (0, padding))  # Pad the last dimension of act_enc_obs
                        act_enc_pred = F.pad(act_enc_pred, (0, padding))  # Pad the last dimension of act_enc_pred
                else:
                    act_enc_obs = act_lat_obs.reshape(-1, seq_len, self.action_latentdim)
                    act_enc_pred = act_lat_pred
                    # If the action_latentdim is not divisible by num_heads, pad the last dimension
                    if self.action_latentdim % self.num_heads != 0:
                        padding = self.num_heads - (self.action_latentdim % self.num_heads)
                        act_enc_obs = F.pad(act_enc_obs, (0, padding))  # Pad the last dimension of act_enc_obs
                        act_enc_pred = F.pad(act_enc_pred, (0, padding))  # Pad the last dimension of act_enc_pred
                enc_obs_concat = torch.cat((enc_obs, act_enc_obs), dim=-1)
        else:
            enc_obs_concat = enc_obs
        
        ### agg token
        B, _, _ = enc_obs.shape
        agg_tokens = self.agg_token.expand(B, -1, -1)
        
        ### transformer encoder
        x = torch.cat((agg_tokens, enc_obs_concat), dim=1)
        x, _ = self.transformer_encoder(x)
        
        agg_out = x[:, 0]
        ### predictor
        if self.act_cond:
            agg_out_conditioned = torch.cat((agg_out, act_enc_pred), dim=-1)
            pred_out = self.predictor(agg_out_conditioned)
        else:
            pred_out = self.predictor(agg_out)
        loss = 1-self.criterion(pred_out, enc_pred_detached).mean()
        z1 = enc_obs[:,0]
        if seq_len == 1:
            z2 = enc_pred_detached
        else:
            z2 = enc_obs[:,1]
        return loss, agg_out, z1, z2


########### Equivariant Methods ###########

###### Conv-I-JEPA ######
class Conv_IJEPA(nn.Module):
    def __init__(self, fovea_size, img_size, ema, ema_decay=0.996, **kwargs):
        super().__init__()

        self.fovea_size = fovea_size
        self.img_size = img_size
        self.ema = ema
        self.ema_decay = ema_decay
        self.n_channels = kwargs["n_channels"]
        self.num_classes = kwargs["num_classes"]
        self.num_heads = kwargs["num_heads"]
        self.num_enc_layers = kwargs["num_enc_layers"]
        self.action_projdim = kwargs["act_projdim"]
        self.action_latentdim = kwargs["act_latentdim"]
        self.act_cond = kwargs["act_cond"]
        self.cifar_resnet = kwargs["cifar_resnet"]
        
        self.criterion = nn.CosineSimilarity(dim=1)

        self.encoder = torchvision.models.resnet18(pretrained=False, zero_init_residual=True)
        self.res_out_dim = self.encoder.fc.in_features
        self.emb_dim = self.res_out_dim
        self.encoder.fc = torch.nn.Identity()
        if self.cifar_resnet:
            self.encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = torch.nn.Identity()
        

        self.projector = nn.Sequential(
                nn.Linear(self.res_out_dim, 2048, bias=False),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 128, bias=False),
                nn.BatchNorm1d(128, affine=False))
        self.proj_outdim = 128
        self.predin_dim = 128 + self.action_projdim
        self.predout_dim = self.res_out_dim
                     
        if self.ema:
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_projector = copy.deepcopy(self.projector)
            for param in self.target_encoder.parameters():
                param.requires_grad = False
            for param in self.target_projector.parameters():
                param.requires_grad = False
        
        self.action_proj = nn.Sequential(
            nn.Linear(self.action_latentdim, self.action_projdim, bias=False), torch.nn.BatchNorm1d(self.action_projdim, affine=False))
        self.pred_hidden = kwargs["pred_hidden"]

        self.predictor = nn.Sequential(
            nn.Linear(self.predin_dim, self.pred_hidden,bias=False),
            nn.BatchNorm1d(self.pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.pred_hidden, self.proj_outdim),)
    
    def _update_target_network(self):
        for online_params, target_params in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.ema_decay * target_params.data + (1 - self.ema_decay) * online_params.data

        for online_proj_params, target_proj_params in zip(self.projector.parameters(), self.target_projector.parameters()):
            target_proj_params.data = self.ema_decay * target_proj_params.data + (1 - self.ema_decay) * online_proj_params.data
            
    def update_moving_average(self):
        """Update the target network using EMA."""
        self._update_target_network()
    
    def add_probes(self):
        for param in self.parameters():
            param.requires_grad = False
        self.pos_regressor = nn.Linear(self.res_out_dim*2, 2)
        self.res_classifier = nn.Linear(self.res_out_dim, self.num_classes)
    
    def forward(self, fov_x_obs, fov_x_last, abs_latents):
        ###encodings
        self.num_saccades = fov_x_obs.shape[1] + 1
        fov_x_obs = fov_x_obs.reshape(-1, self.n_channels, self.fovea_size, self.fovea_size)
        fov_x_last = fov_x_last.reshape(-1, self.n_channels, self.fovea_size, self.fovea_size)

        fov_encs_obs = self.encoder(fov_x_obs)
        
        loss = 0.
        fov_encs_projs = self.projector(fov_encs_obs)
        fov_encs_obs = fov_encs_obs.reshape(-1, self.num_saccades-1, self.res_out_dim)
        fov_encs_projs = fov_encs_projs.reshape(-1, self.num_saccades-1, self.proj_outdim)
        
        if self.ema:
            pred_enc = self.target_encoder(fov_x_last)
            pred_proj = self.target_projector(pred_enc)
            pred_proj = pred_proj.detach()
        else:
            pred_enc = self.encoder(fov_x_last)
            pred_proj = self.projector(pred_enc)
        
        lat_obs = abs_latents[:,:-1,:].reshape(-1, self.num_saccades-1, self.action_latentdim)
        lat_pred = abs_latents[:,-1,:].reshape(-1, self.action_latentdim)
        loss = 0.
        for i in range(self.num_saccades-1):
            rel_lat = lat_pred - lat_obs[:,i]
            rel_lat = self.action_proj(rel_lat)
            pred_in = torch.concat((rel_lat, fov_encs_projs[:,i]),dim=1)
            pred_hat = self.predictor(pred_in) # NxC
            loss += 1-self.criterion(pred_hat, pred_proj).mean()
        loss /= (self.num_saccades-1)

        return loss, None, fov_encs_obs[:,0], pred_enc




###### SimCLR-EquiMod ######
## Based on https://github.com/ADevillers/EquiMod/blob/SimCLR%2BEquiMod/src/model.py
class SimCLREquiMod(nn.Module):
    def __init__(self, num_classes, latent_size, learn_action_emb, action_emb_size, equi_factor, temp, cifar_resnet):
        super().__init__()
        
        self.cifar_resnet = cifar_resnet
        ### ResNet encoder
        self.encoder = torchvision.models.resnet18(pretrained=False, zero_init_residual=True)
        self.res_out_dim = self.encoder.fc.in_features
        self.encoder.fc = torch.nn.Identity()
        if self.cifar_resnet:
            self.encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = torch.nn.Identity()
        self.inv_emb_size = 128
        self.equi_emb_size = 128
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.learn_action_emb = learn_action_emb
        if self.learn_action_emb:
            self.action_emb_size = action_emb_size
            self.action_proj = torch.nn.Sequential(torch.nn.Linear(self.latent_size, self.action_emb_size, bias=False), torch.nn.BatchNorm1d(self.action_emb_size, affine=False))
        else:
            self.action_emb_size = latent_size
        self.equi_factor = equi_factor
        self.temp = temp
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.projector_equi  = torch.nn.Sequential(
            torch.nn.Linear(self.res_out_dim, 1024, bias=False),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024, bias=False),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, self.equi_emb_size, bias=False),
            torch.nn.BatchNorm1d(self.equi_emb_size, affine=False)
        )

        self.projector_inv  = torch.nn.Sequential(
            torch.nn.Linear(self.res_out_dim, 1024, bias=False),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024, bias=False),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, self.inv_emb_size, bias=False),
            torch.nn.BatchNorm1d(self.inv_emb_size, affine=False)
        )
        
        self.pred_in_dim = self.equi_emb_size + self.action_emb_size
        self.predictor = torch.nn.Sequential(torch.nn.Linear(self.pred_in_dim, self.equi_emb_size, bias=False), torch.nn.BatchNorm1d(self.equi_emb_size, affine=False))

    def NTXent_loss(self, features, batch_size):
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(labels.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        logits = logits / self.temp
        return logits, labels
    
    def add_probes(self, latent_type):
        for param in self.parameters():
            param.requires_grad = False
        dim = self.res_out_dim
        if latent_type == "aug":
            self.blur_regressor = nn.Linear(dim, 1)
            self.crop_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.jitter_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
        elif latent_type == "rotcolor":
            self.rot_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.color_regressor = nn.Linear(dim, 2)
        self.res_classifier = nn.Linear(self.res_out_dim, self.num_classes)

    def forward(self, x, y, rel_latent):
        x_emb = self.encoder(x)
        y_emb = self.encoder(y)
        
        emb_to_return1 = x_emb
        emb_to_return2 = y_emb
        
        x_inv = self.projector_inv(x_emb)
        y_inv = self.projector_inv(y_emb)

        x_equi = self.projector_equi(x_emb)
        y_equi = self.projector_equi(y_emb)
        
        if self.learn_action_emb:
            rel_latent = self.action_proj(rel_latent)
            
        pred_in = torch.concat((rel_latent,x_equi),dim=1)
        y_equi_pred = self.predictor(pred_in)
        
        features = torch.cat([y_equi_pred,y_equi],axis=0)
        logits, labels = self.NTXent_loss(features,batch_size=x.shape[0])
        loss_equi = self.criterion(logits, labels)
        
        features = torch.cat([x_inv,y_inv],axis=0)
        logits, labels = self.NTXent_loss(features,batch_size=x.shape[0])
        loss_inv = self.criterion(logits, labels)
        
        loss = self.equi_factor*loss_equi + loss_inv

        return loss, emb_to_return1, emb_to_return2
    

#### Conditional BYOL ####
class BYOL_Conditional(nn.Module):
    def __init__(self, n_classes, a_dim, a_projdim, **kwargs):
        super().__init__()
        self.cifar_resnet = kwargs["cifar_resnet"]
        ### ResNet encoder
        self.encoder = torchvision.models.resnet18(pretrained=False, zero_init_residual=True)
        self.res_out_dim = self.encoder.fc.in_features
        self.encoder.fc = torch.nn.Identity()
        if self.cifar_resnet:
            self.encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = torch.nn.Identity()
        self.target_encoder = copy.deepcopy(self.encoder)
        self.ema_decay = kwargs["ema_decay"]
        self.num_classes = n_classes
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.emb_dim = self.res_out_dim
        self.num_heads = kwargs["num_heads"]
        self.num_enc_layers = kwargs["num_enc_layers"]
        self.a_dim = a_dim
        self.a_projdim = a_projdim
        
        self.predin_dim = 128 + self.a_projdim
        self.predout_dim = 128
        
        self.criterion = nn.CosineSimilarity(dim=1)
        self.projector = nn.Sequential(
                nn.Linear(self.res_out_dim, 2048, bias=False),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, self.predout_dim, bias=False),
                nn.BatchNorm1d(self.predout_dim, affine=False))
            
        
        self.action_proj = torch.nn.Sequential(torch.nn.Linear(self.a_dim, self.a_projdim, bias=False), torch.nn.BatchNorm1d(self.a_projdim, affine=False))
        
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_projector.parameters():
            p.requires_grad = False
        
        
        self.predictor = nn.Sequential(
                nn.Linear(self.predin_dim, 2048, bias=False),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, self.predout_dim),)

    def _update_target_network(self):
        for online_params, target_params in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.ema_decay * target_params.data + (1 - self.ema_decay) * online_params.data

        for online_proj_params, target_proj_params in zip(self.projector.parameters(), self.target_projector.parameters()):
            target_proj_params.data = self.ema_decay * target_proj_params.data + (1 - self.ema_decay) * online_proj_params.data

    def update_moving_average(self):
        """Update the target network using EMA."""
        self._update_target_network()
    
    def add_probes(self, latent_type):
        for param in self.parameters():
            param.requires_grad = False
        dim = self.res_out_dim
        if latent_type == "aug":
            self.blur_regressor = nn.Linear(dim, 1)
            self.crop_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.jitter_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
        elif latent_type == "rotcolor":
            self.rot_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.color_regressor = nn.Linear(dim, 2)
        self.res_classifier = nn.Linear(self.res_out_dim, self.num_classes)

    def forward(self, x1, x2, rel_latent):
        z1 = self.encoder(x1) # NxC
        
        y1 = self.projector(z1)
        
        act_enc = self.action_proj(rel_latent)
        pred_in = torch.concat((act_enc, y1),dim=1)
        
        y2_hat = self.predictor(pred_in) # NxC
        
        ### target network
        ##z1_target = self.target_encoder(x1)
        z2_target = self.target_encoder(x2)
        y2_target = self.target_projector(z2_target)
        ##y1_target, y2_target = y1_target.detach(), y2_target.detach()
        y2_target = y2_target.detach()
        
        loss = 2-2*self.criterion(y2_hat, y2_target).mean()
        return loss, z1, z2_target


#### SEN-Contrastive ####
class SEN_Contrastive(nn.Module):
    def __init__(self, num_classes, latent_size, learn_action_emb, action_emb_size, temp, cifar_resnet):
        super().__init__()
        
        self.cifar_resnet = cifar_resnet
        
        self.encoder = torchvision.models.resnet18(pretrained=False, zero_init_residual=True)
        self.res_out_dim = self.encoder.fc.in_features
        self.encoder.fc = torch.nn.Identity()
        if self.cifar_resnet:
            self.encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = torch.nn.Identity()
        self.equi_emb_size = 2048
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.learn_action_emb = learn_action_emb
        if learn_action_emb:
            self.action_emb_size = action_emb_size
            self.action_proj = torch.nn.Sequential(torch.nn.Linear(self.latent_size, self.action_emb_size, bias=False), torch.nn.BatchNorm1d(self.action_emb_size, affine=False))
        else:
            self.action_emb_size = latent_size
        self.temp = temp
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.projector_equi  = nn.Sequential(
                nn.Linear(self.res_out_dim, 2048, bias=False),
                nn.BatchNorm1d(2048),
                nn.ReLU(True),
                nn.Linear(2048, 2048, bias=False),
                nn.BatchNorm1d(2048),
                nn.ReLU(True),
                nn.Linear(2048, self.equi_emb_size, bias=False),
                nn.BatchNorm1d(self.equi_emb_size, affine=False)
            )
        
        self.pred_in_dim = self.equi_emb_size + self.action_emb_size
        self.predictor = torch.nn.Sequential(torch.nn.Linear(self.pred_in_dim, self.equi_emb_size, bias=False), torch.nn.BatchNorm1d(self.equi_emb_size, affine=False))
    
    def add_probes(self, latent_type):
        for param in self.parameters():
            param.requires_grad = False
        dim = self.res_out_dim
        if latent_type == "aug":
            self.blur_regressor = nn.Linear(dim, 1)
            self.crop_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.jitter_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
        elif latent_type == "rotcolor":
            self.rot_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.color_regressor = nn.Linear(dim, 2)
        self.res_classifier = nn.Linear(self.res_out_dim, self.num_classes)

    def NTXent_loss(self, features, batch_size):
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(labels.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        logits = logits / self.temp
        return logits, labels
    

    def forward(self, x, y, rel_latent=None):
        x_emb = self.encoder(x)
        y_emb = self.encoder(y)

        emb_to_return1 = x_emb
        emb_to_return2 = y_emb

        x_equi = self.projector_equi(x_emb)
        y_equi = self.projector_equi(y_emb)
        
        if self.learn_action_emb:
            rel_latent = self.action_proj(rel_latent)
            
        pred_in = torch.concat((rel_latent,x_equi),dim=1)
        y_equi_pred = self.predictor(pred_in)

        
        features = torch.cat([y_equi_pred,y_equi],axis=0)
        logits, labels = self.NTXent_loss(features, x_equi.shape[0])
        loss = self.criterion(logits, labels)

        return loss, emb_to_return1, emb_to_return2
    




##### SIE with Hypernetwork #####
## Based on https://github.com/facebookresearch/SIE/blob/main/src/models.py
class HyperNet(nn.Module):
    def __init__(self, latent_size : int, output_size : int, bias_hypernet=False, hypernetwork="linear"):
        super(HyperNet,self).__init__()
        if bias_hypernet:
            print("Bias in the hypernetwork")
        if hypernetwork == "linear":
            self.net = nn.Sequential(
                nn.Linear(latent_size,output_size,bias=bias_hypernet), # Linear combination for now
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(latent_size,latent_size,bias=bias_hypernet),
                nn.ReLU(),
                nn.Linear(latent_size,latent_size,bias=bias_hypernet),
                nn.ReLU(),
                nn.Linear(latent_size,output_size,bias=bias_hypernet),
            )
        
    def forward(self, x : torch.Tensor):
        out = self.net(x)
        return out

    
class ParametrizedNet(nn.Module):
    def __init__(self,equivariant_size : int, latent_size : int):
        super(ParametrizedNet,self).__init__()
        self.predictor_relu = False

        archi_str = str(equivariant_size) + "-" + str(equivariant_size)

        print("Predictor architecture: ", archi_str)
        self.predictor = [int(x) for x in archi_str.split("-")]
        
        self.num_weights_each = [ self.predictor[i]*self.predictor[i+1] for i in range(len(self.predictor)-1)]

        self.num_params_each = self.num_weights_each
        print(self.num_params_each)
        self.cum_params = [0] + list(np.cumsum(self.num_params_each))        
        self.hypernet = HyperNet(latent_size, self.cum_params[-1])
        self.activation = nn.ReLU() if self.predictor_relu else nn.Identity()
        
    def forward(self, x : torch.Tensor, z : torch.Tensor):
        """
        x must be (batch_size, 1, size)
        
        Since F.linear(x,A,b) = x @ A.T + b (to have A (out_dim,in_dim) and be coherent with nn.linear)
        and  torch.bmm(x,A)_i = x_i @ A_i
        to emulate the same behaviour, we transpose A along the last two axes before bmm
        """
        weights = self.hypernet(z)
        out=x
        for i in range(len(self.predictor)-1):
            w = weights[...,self.cum_params[i]:self.cum_params[i] + self.num_weights_each[i]].view(-1,self.predictor[i+1],self.predictor[i])
            out = torch.bmm(out,torch.transpose(w,-2,-1))
            # if self.args.bias_pred:
            #     b = weights[...,self.cum_params[i+1] - self.num_biases_each[i]:self.cum_params[i+1]].unsqueeze(1)
            #     out = out + b
            if i < len(self.predictor)-2:
                out = self.activation(out)
        
        return out.squeeze()

class SIE(nn.Module):
    def __init__(self, n_classes, inv_coeff, var_coeff, cov_coeff, equi_factor, latent_size, cifar_resnet):
        super().__init__()
        self.cifar_resnet = cifar_resnet
        ### ResNet encoder
        self.encoder = torchvision.models.resnet18(pretrained=False, zero_init_residual=True)
        self.res_out_dim = self.encoder.fc.in_features
        self.encoder.fc = torch.nn.Identity()
        if self.cifar_resnet:
            self.encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = torch.nn.Identity()
        
        self.inv_repr_size = self.res_out_dim // 2
        self.equi_repr_size = self.res_out_dim - self.inv_repr_size
        
        self.inv_emb_size = 1024
        self.equi_emb_size = 1024
        self.latent_size = latent_size
        
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.equi_factor = equi_factor
        
        self.projector_inv =  nn.Sequential(
            nn.Linear(self.inv_repr_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.inv_emb_size, bias=False)
        )
        
        self.projector_equi =  nn.Sequential(
            nn.Linear(self.equi_repr_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.equi_emb_size, bias=False)
        )
        self.predictor = ParametrizedNet(self.equi_emb_size,self.latent_size)
        
        self.num_classes = n_classes
        

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def add_probes(self, latent_type):
        for param in self.parameters():
            param.requires_grad = False
        dim = self.res_out_dim
        if latent_type == "aug":
            self.blur_regressor = nn.Linear(dim, 1)
            self.crop_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.jitter_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
        elif latent_type == "rotcolor":
            self.rot_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.color_regressor = nn.Linear(dim, 2)
        self.res_classifier = nn.Linear(self.res_out_dim, self.num_classes)

    def forward(self, x, y, rel_latent=None):
        batch_size = x.size(0)
        x_emb = self.encoder(x)
        y_emb = self.encoder(y)
        
        emb_to_return1 = x_emb
        emb_to_return2 = y_emb

        x_inv = x_emb[...,:self.inv_repr_size]
        y_inv = y_emb[...,:self.inv_repr_size]
        x_equi = x_emb[...,self.inv_repr_size:]
        y_equi = y_emb[...,self.inv_repr_size:]
        
        x_inv = self.projector_inv(x_inv)
        y_inv = self.projector_inv(y_inv)
        x_equi = self.projector_equi(x_equi)
        y_equi = self.projector_equi(y_equi)

        # Concatenate both parts to apply the regularization on the whole vectors
        # This helps remove information that would be redundant in both parts
        # _________________
        # |        |      |
        # |   Inv  |Common|
        # |________|______|
        # |        |      |
        # | Common |  Eq  |
        # |________|______|
        #
        # Without this concatenation we would not regularize the common parts

        x = torch.cat((x_inv,x_equi),dim=1)
        y = torch.cat((y_inv, y_equi),dim=1)

        #======================================
        #           Inv part
        #======================================
        repr_loss_inv = F.mse_loss(x_inv, y_inv)

        #======================================
        #           Equi part
        #======================================

        # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
        y_equi_pred = self.predictor(x_equi.unsqueeze(1), rel_latent)
        
        repr_loss_equi = F.mse_loss(y_equi_pred, y_equi)
        
        #======================================
        #           Common part
        #======================================

        y_equi_pred = y_equi_pred - y_equi_pred.mean(dim=0)
        std_y_equi_pred = torch.sqrt(y_equi_pred.var(dim=0) + 0.0001)
        pred_std_loss = torch.mean(F.relu(1 - std_y_equi_pred)) / 2

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(cov_x.shape[0]) \
            + off_diagonal(cov_y).pow_(2).sum().div(cov_x.shape[0])

        loss = (
                  self.inv_coeff * repr_loss_inv
                + self.equi_factor*self.inv_coeff * repr_loss_equi
                + self.var_coeff * std_loss
                + self.var_coeff * pred_std_loss
                + self.cov_coeff * cov_loss
                )

        return loss, emb_to_return1, emb_to_return2


########### Invariant Methods ###########



##### SimCLR #####
class SimCLR(nn.Module):
    def __init__(self, n_classes, temp, cifar_resnet):
        super().__init__()
    
        self.cifar_resnet = cifar_resnet
        ### ResNet encoder
        self.encoder = torchvision.models.resnet18(pretrained=False, zero_init_residual=True)
        self.res_out_dim = self.encoder.fc.in_features
        self.encoder.fc = torch.nn.Identity()
        if self.cifar_resnet:
            self.encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = torch.nn.Identity()
        self.projector = nn.Sequential(
                nn.Linear(self.res_out_dim, 2048, bias=False),
                nn.BatchNorm1d(2048),
                nn.ReLU(True),
                nn.Linear(2048, 2048, bias=False),
                nn.BatchNorm1d(2048),
                nn.ReLU(True),
                nn.Linear(2048, 2048, bias=False),
                nn.BatchNorm1d(2048, affine=False)
            )
        self.num_classes = n_classes
        self.temp = temp
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def NTXent_loss(self, features, batch_size):
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(labels.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        logits = logits / self.temp
        return logits, labels
    
    def add_probes(self, latent_type):
        for param in self.parameters():
            param.requires_grad = False
        dim = self.res_out_dim
        if latent_type == "aug":
            self.blur_regressor = nn.Linear(dim, 1)
            self.crop_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.jitter_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
        elif latent_type == "rotcolor":
            self.rot_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.color_regressor = nn.Linear(dim, 2)
        self.res_classifier = nn.Linear(self.res_out_dim, self.num_classes)

    def forward(self, x1, x2, rel_latent=None):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)


        y1 = self.projector(z1)
        y2 = self.projector(z2)

        features = torch.cat([y1,y2],axis=0)
        logits, labels = self.NTXent_loss(features, y1.shape[0])
        loss = self.criterion(logits, labels)
        return loss, z1, z2




# ##### BYOL #####
class BYOL(nn.Module):
    def __init__(self, n_classes, cifar_resnet, **kwargs):
        super().__init__()
        self.cifar_resnet = cifar_resnet
        self.encoder = torchvision.models.resnet18(pretrained=False, zero_init_residual=True)
        self.res_out_dim = self.encoder.fc.in_features
        self.encoder.fc = torch.nn.Identity()
        if self.cifar_resnet:
            self.encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = torch.nn.Identity()
        self.target_encoder = copy.deepcopy(self.encoder)
        self.ema_decay = kwargs["ema_decay"]
        self.num_classes = n_classes
        for p in self.target_encoder.parameters():
            p.requires_grad = False
            
        self.res_out_dim = 512

        self.num_heads = kwargs["num_heads"]
        self.num_enc_layers = kwargs["num_enc_layers"]
        
        self.predin_dim = 128
        self.predout_dim = 128
        
        self.criterion = nn.CosineSimilarity(dim=1)
        self.projector = nn.Sequential(
                nn.Linear(self.res_out_dim, 2048, bias=False),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, self.predin_dim, bias=False),
                nn.BatchNorm1d(self.predin_dim, affine=False))
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_projector.parameters():
            p.requires_grad = False
        
        self.predictor = nn.Sequential(
                nn.Linear(self.predin_dim, 2048, bias=False),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, self.predout_dim),)

        
    
    def _update_target_network(self):
        for online_params, target_params in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.ema_decay * target_params.data + (1 - self.ema_decay) * online_params.data

        for online_proj_params, target_proj_params in zip(self.projector.parameters(), self.target_projector.parameters()):
            target_proj_params.data = self.ema_decay * target_proj_params.data + (1 - self.ema_decay) * online_proj_params.data

    def update_moving_average(self):
        """Update the target network using EMA."""
        self._update_target_network()
        
    def add_probes(self, latent_type):
        for param in self.parameters():
            param.requires_grad = False
        dim = self.res_out_dim
        if latent_type == "aug":
            self.blur_regressor = nn.Linear(dim, 1)
            self.crop_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.jitter_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
        elif latent_type == "rotcolor":
            self.rot_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.color_regressor = nn.Linear(dim, 2)
        self.res_classifier = nn.Linear(self.res_out_dim, self.num_classes)

    def forward(self, x1, x2, rel_latent=None):
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC
        
        y1 = self.projector(z1)
        y2 = self.projector(z2)
        
        p1 = self.predictor(y1) # NxC
        p2 = self.predictor(y2) # NxC
        
        ### target network
        z1_target = self.target_encoder(x1)
        z2_target = self.target_encoder(x2)

        y1_target = self.target_projector(z1_target)
        y2_target = self.target_projector(z2_target)
        y1_target, y2_target = y1_target.detach(), y2_target.detach()
        
        
        loss1 = 2-2*self.criterion(p1, y2_target).mean()
        loss2 = 2-2*self.criterion(p2, y1_target).mean()
        loss = loss1 + loss2
        return loss, z1, z2




##### VICReg #####

class VICReg(nn.Module):
    def __init__(self, n_classes, inv_coeff, var_coeff, cov_coeff, cifar_resnet):
        super().__init__()
        self.cifar_resnet = cifar_resnet
        ### ResNet encoder
        self.encoder = torchvision.models.resnet18(pretrained=False, zero_init_residual=True)
        self.res_out_dim = self.encoder.fc.in_features
        self.encoder.fc = torch.nn.Identity()
        if self.cifar_resnet:
            self.encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = torch.nn.Identity()
        
        self.emb_size = 2048
        self.projector =  nn.Sequential(
                nn.Linear(self.res_out_dim, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, self.emb_size, bias=False)
            )
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.num_classes = n_classes
       
    def add_probes(self, latent_type, non_relative_trans=False):
        for param in self.parameters():
            param.requires_grad = False
        if non_relative_trans:
            dim = self.res_out_dim
        else:
            dim = self.res_out_dim*2
        if latent_type == "aug":
            self.blur_regressor = nn.Linear(dim, 1)
            self.crop_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.jitter_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
        elif latent_type == "rotcolor":
            self.rot_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.color_regressor = nn.Linear(dim, 2)
        self.res_classifier = nn.Linear(self.res_out_dim, self.num_classes)

    def forward(self, x1, x2, rel_latent=None):
        batch_size = x1.size(0)
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        y1 = self.projector(z1)
        y2 = self.projector(z2)
            
        inv_loss = F.mse_loss(y1, y2)
        
        y1 = y1 - y1.mean(dim=0)
        y2 = y2 - y2.mean(dim=0)
        
        std_x = torch.sqrt(y1.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y2.var(dim=0) + 0.0001)
        
        var_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (y1.T @ y1) / (batch_size - 1)
        cov_y = (y2.T @ y2) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.emb_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.emb_size)

        loss = (
            self.inv_coeff * inv_loss
            + self.var_coeff * var_loss
            + self.cov_coeff * cov_loss
        )
        
        return loss, z1, z2


#################################### VICReg_Traj ########################################################
class VICReg_Traj(nn.Module):
    def __init__(self, n_classes, inv_coeff, var_coeff, cov_coeff, alpha, distributed, cifar_resnet):
        super().__init__()
        self.cifar_resnet = cifar_resnet
        self.encoder = torchvision.models.resnet18(pretrained=False, zero_init_residual=True)
        self.res_out_dim = self.encoder.fc.in_features
        self.encoder.fc = torch.nn.Identity()
        if self.cifar_resnet:
            self.encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = torch.nn.Identity()
        
        self.emb_size = 2048
        self.distributed = distributed
        self.projector =  nn.Sequential(
                nn.Linear(self.res_out_dim, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, self.emb_size, bias=False)
            )
            print("Using MLP projector")
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.alpha = alpha
        self.num_classes = n_classes
       
    def add_probes(self, latent_type):
        for param in self.parameters():
            param.requires_grad = False
        dim = self.res_out_dim
        if latent_type == "aug":
            self.blur_regressor = nn.Linear(dim, 1)
            self.crop_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.jitter_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
        elif latent_type == "rotcolor":
            self.rot_regressor = nn.Sequential(
                nn.Linear(dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.rot_abs_regressor = nn.Sequential(
                nn.Linear(self.res_out_dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
            self.lin_rot_regressor = nn.Linear(dim, 4)
            self.lin_rot_abs_regressor = nn.Linear(self.res_out_dim, 4)
            self.color_regressor = nn.Linear(dim, 2)
        self.res_classifier = nn.Linear(self.res_out_dim, self.num_classes)
        
    def forward(self, x, rel_latent=None):
        assert len(x.shape) == 5 and x.shape[1] == 3 and x.shape[2] == 3
        
        batch_size = x.size(0)
        x_reshaped = x.reshape(batch_size*3, 3, x.shape[3], x.shape[4])
        x_reshaped = self.encoder(x_reshaped)
        y = self.projector(x_reshaped)
        x_reshaped = x_reshaped.reshape(batch_size, 3, self.res_out_dim)
        z_l, z_c, z_r = x_reshaped[:, 0], x_reshaped[:, 1], x_reshaped[:, 2]

        y = y.reshape(batch_size, 3, self.emb_size)
        y_l, y_c, y_r = y[:, 0], y[:, 1], y[:, 2]
        
        v1 = y_c - y_l
        v2 = y_r - y_c
        u1 = v1 - (v1 * y_c).sum(dim=1, keepdim=True) * y_c
        u2 = v2 - (v2 * y_c).sum(dim=1, keepdim=True) * y_c
        
        traj_loss = -nn.CosineSimilarity(dim=1)(u1, u2).mean()
            
        inv_loss = F.mse_loss(y_l, y_c)
        
        y1 = y_l - y_l.mean(dim=0)
        y2 = y_c - y_c.mean(dim=0)
        
        std_x = torch.sqrt(y1.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y2.var(dim=0) + 0.0001)
        
        var_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (y1.T @ y1) / (batch_size - 1)
        cov_y = (y2.T @ y2) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.emb_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.emb_size)


        loss = (
            self.inv_coeff * inv_loss
            + self.var_coeff * var_loss
            + self.cov_coeff * cov_loss
            + self.alpha * traj_loss
        )
        
        return loss, z_l, z_c, z_r

