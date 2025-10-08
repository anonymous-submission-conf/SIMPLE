import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers import ViTModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import *
from cfg import CFG

class SIMPLE(torch.nn.Module):
    def __init__(self, tkr, device):
        super(SIMPLE, self).__init__()
        self.tkr = tkr
        self.device = device
        self.bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.img_encoder = ViTModel.from_pretrained("google/vit-base-patch32-224-in21k")

        #Params - Shared Fusion
        self.sigmoid = nn.Sigmoid()
        self.attention_i = nn.MultiheadAttention(embed_dim=768, num_heads=4)
        self.attention_t = nn.MultiheadAttention(embed_dim=768, num_heads=4)
        self.W_gated_i = nn.Linear(768, 768)
        self.W_gated_t = nn.Linear(768, 768)
        self.alpha_1 = nn.Parameter(torch.tensor(1.0).to(self.device))
        self.alpha_2 = nn.Parameter(torch.tensor(1.0).to(self.device))
        self.beta_1 = nn.Parameter(torch.tensor(1.0).to(self.device))
        self.beta_2 = nn.Parameter(torch.tensor(1.0).to(self.device))
        self.proj = nn.Linear(768, 768)
#         self.img_proj = nn.Linear(50, CFG.max_len)
    
    def get_contrastive_loss(self, F_mat, is_sarcastic, temperature=0.07):
        F_pooled = F_mat.mean(dim=1) #(B, D)
        z = F.normalize(F_pooled, dim=1)
        sim = torch.matmul(z, z.T) / temperature  # (B, B)

        B = z.size(0)
        mask_self = torch.eye(B, device=z.device).to(self.device)
        sim = sim - 1e9 * mask_self  # effectively remove diagonal

        labels = is_sarcastic.view(-1, 1)
        pos_mask = (labels == labels.T).float() * (1 - mask_self)

        # log_softmax over each row (dim=1)
        log_prob = F.log_softmax(sim, dim=1)  # (B, B)

        # only keep positive logits
        loss = -(log_prob * pos_mask).sum(dim=1)

        valid = (pos_mask.sum(dim=1) > 0).float()
        loss = loss * valid
        contrastive_loss = loss.sum() / (valid.sum() + 1e-8)

        return contrastive_loss
        
    def shared_fusion(self, text_embeddings, img_embeddings, attention_mask):

        attention_mask = torch.tile(attention_mask.unsqueeze(2), dims=(1,1,text_embeddings.shape[-1])).to(self.device)

        #F_IT
        A_i, _ = self.attention_i(img_embeddings, img_embeddings, img_embeddings)
        F_it = (text_embeddings * A_i) * attention_mask
        # F_it = (text_embeddings * A_i)

        #F_TI
        A_t, _ = self.attention_t(text_embeddings, text_embeddings, text_embeddings)
        A_t *= attention_mask
        F_ti = img_embeddings * A_t

        #G_I
        G_i = self.sigmoid(self.W_gated_i(img_embeddings))

        #G_T
        # G_t = self.sigmoid(self.W_gated_t(text_embeddings))
        G_t = self.sigmoid(self.W_gated_t(text_embeddings)) * attention_mask

        #Computing SF output
        F_1 = (G_i * F_ti) + ((1-G_i) * F_it)
        F_2 = (G_t * F_ti) + ((1-G_t) * F_it)
        F_i = (G_i*img_embeddings) + ((1-G_i) * F_ti)
        F_t = (G_t*text_embeddings) + ((1-G_t) * F_it)
        shared_fusion_output = (self.alpha_1 * F_1) + (self.alpha_2 * F_2) + (self.beta_1 * F_i) + (self.beta_2 * F_t)
        return shared_fusion_output

    def forward(self, input_image, input_ids, attention_mask, target_ids, is_sarcastic=None, mode="train", gen_params=None, **kwargs):
        text_embeds = self.bart.model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        img_embeds = self.img_encoder(input_image).last_hidden_state[:,0,:]
        img_embeds = self.proj(img_embeds).unsqueeze(1).expand(-1, text_embeds.shape[1], -1)

#         concat_embeds = torch.cat((text_embeds, img_embeds), dim=2)
#         contrastive_loss = self.get_contrastive_loss(concat_embeds, is_sarcastic)
        contrastive_loss = 0

        shared_fusion_out = self.shared_fusion(text_embeds, img_embeds, attention_mask)    
        combined_embeds = BaseModelOutput(last_hidden_state=shared_fusion_out)
        
        if (mode == "train" or mode == "val"):
            final_output = self.bart(
                encoder_outputs = combined_embeds,
                attention_mask = attention_mask,
                labels = target_ids,
                return_dict = True
            )
            ce_loss = final_output.loss
            return contrastive_loss, ce_loss
        else:
            gen_result = self.bart.generate(
                encoder_outputs = combined_embeds,
                attention_mask = attention_mask,
                max_new_tokens = 128,
                do_sample=True,
                num_beams=8,
#                 **gen_params,
#                 num_beams=4
            )
            gen_decoded = self.tkr.batch_decode(gen_result, skip_special_tokens=True)
            return gen_result, gen_decoded

class TARGEN(torch.nn.Module):
    def __init__(self, tkr, device):
        super(TARGEN, self).__init__()
        self.tkr = tkr
        self.device = device
        #     self.vision_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.img_encoder = ViTModel.from_pretrained("google/vit-base-patch32-224-in21k")

        #Params - Shared Fusion
        self.sigmoid = nn.Sigmoid()
        self.attention_i = nn.MultiheadAttention(embed_dim=768, num_heads=4)
        self.attention_t = nn.MultiheadAttention(embed_dim=768, num_heads=4)
        self.W_gated_i = nn.Linear(768, 768)
        self.W_gated_t = nn.Linear(768, 768)
        self.alpha_1 = nn.Parameter(torch.tensor(1.0).to(self.device))
        self.alpha_2 = nn.Parameter(torch.tensor(1.0).to(self.device))
        self.beta_1 = nn.Parameter(torch.tensor(1.0).to(self.device))
        self.beta_2 = nn.Parameter(torch.tensor(1.0).to(self.device))
        self.proj = nn.Linear(768, 768)
#         self.img_proj = nn.Linear(50, CFG.max_len)
    
    def get_contrastive_loss(self, F_mat, is_sarcastic, temperature=0.07):
        F_pooled = F_mat.mean(dim=1) #(B, D)
        z = F.normalize(F_pooled, dim=1)
        sim = torch.matmul(z, z.T) / temperature  # (B, B)

        B = z.size(0)
        mask_self = torch.eye(B, device=z.device).to(self.device)
        sim = sim - 1e9 * mask_self  # effectively remove diagonal

        labels = is_sarcastic.view(-1, 1)
        pos_mask = (labels == labels.T).float() * (1 - mask_self)

        # log_softmax over each row (dim=1)
        log_prob = F.log_softmax(sim, dim=1)  # (B, B)

        # only keep positive logits
        loss = -(log_prob * pos_mask).sum(dim=1)

        valid = (pos_mask.sum(dim=1) > 0).float()
        loss = loss * valid
        contrastive_loss = loss.sum() / (valid.sum() + 1e-8)

        return contrastive_loss
        
    def shared_fusion(self, text_embeddings, img_embeddings, attention_mask):

        attention_mask = torch.tile(attention_mask.unsqueeze(2), dims=(1,1,text_embeddings.shape[-1])).to(self.device)

        #F_IT
        A_i, _ = self.attention_i(img_embeddings, img_embeddings, img_embeddings)
        F_it = (text_embeddings * A_i) * attention_mask
        # F_it = (text_embeddings * A_i)

        #F_TI
        A_t, _ = self.attention_t(text_embeddings, text_embeddings, text_embeddings)
        A_t *= attention_mask
        F_ti = img_embeddings * A_t

        #G_I
        G_i = self.sigmoid(self.W_gated_i(img_embeddings))

        #G_T
        # G_t = self.sigmoid(self.W_gated_t(text_embeddings))
        G_t = self.sigmoid(self.W_gated_t(text_embeddings)) * attention_mask

        #Computing SF output
        F_1 = (G_i * F_ti) + ((1-G_i) * F_it)
        F_2 = (G_t * F_ti) + ((1-G_t) * F_it)
        F_i = (G_i*img_embeddings) + ((1-G_i) * F_ti)
        F_t = (G_t*text_embeddings) + ((1-G_t) * F_it)
        shared_fusion_output = (self.alpha_1 * F_1) + (self.alpha_2 * F_2) + (self.beta_1 * F_i) + (self.beta_2 * F_t)
        return shared_fusion_output

    def forward(self, input_image, input_ids, attention_mask, **kwargs):        
        text_embeds = self.bart.model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        img_embeds = self.img_encoder(input_image).last_hidden_state[:,0,:]
        img_embeds = self.proj(img_embeds).unsqueeze(1).expand(-1, text_embeds.shape[1], -1)
        
        shared_fusion_out = self.shared_fusion(text_embeds, img_embeds, attention_mask)    
        
        combined_embeds = BaseModelOutput(last_hidden_state=shared_fusion_out)
        
        gen_result = self.bart.generate(
    #                 input_ids = input_ids,
            encoder_outputs = combined_embeds,
            attention_mask = attention_mask,
            max_new_tokens = 64,
            do_sample=True,
            num_beams=8,
#                 **gen_params,
#                 num_beams=4
        )
        gen_decoded = self.tkr.batch_decode(gen_result, skip_special_tokens=True)
        return gen_result, gen_decoded