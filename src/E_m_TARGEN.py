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

class Model(torch.nn.Module):
    def __init__(self, tkr, device):
        super(Model, self).__init__()
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
    
    def get_contrastive_loss(self, F_mat, is_sarcastic, temperature=0.07):
        F_pooled = F_mat.mean(dim=1) #(B, D)
        z = F.normalize(F_pooled, dim=1)
        sim = torch.matmul(z, z.T) / temperature  # (B, B)

        B = z.size(0)
        mask_self = torch.eye(B, device=z.device).to(self.device)
        sim = sim - 1e9 * mask_self  # effectively remove diagonal

        labels = is_sarcastic.view(-1, 1)
        pos_mask = (labels == labels.T).float() * (1 - mask_self)

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

        contrastive_loss = 0
        shared_fusion_out = self.shared_fusion(text_embeds, img_embeds, attention_mask)    
        
        combined_embeds = BaseModelOutput(last_hidden_state=shared_fusion_out)
        if (mode == "train"):
            final_output = self.bart(
        #                 input_ids = input_ids, 
                encoder_outputs = combined_embeds,
                attention_mask = attention_mask,
                labels = target_ids,
                return_dict = True
            )
            ce_loss = final_output.loss
            return contrastive_loss, ce_loss
        else:
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

# class TARGEN(torch.nn.Module):
#     def __init__(self, tkr, device):
#         super(TARGEN, self).__init__()
#         self.tkr = tkr
#         self.device = device
#         #     self.vision_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
#         self.bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
#         self.img_encoder = ViTModel.from_pretrained("google/vit-base-patch32-224-in21k")

#         # we need to project vision encoding into bart space
#         #     self.vision_to_text_space = torch.nn.Linear(768, 768)

#         #     # now we learn lambda vision and lambda text, parameters to correctly fuse text features and vision features
#         #     self.lambda_vision = torch.nn.Parameter(torch.tensor(1.0))
#         #     self.lambda_text = torch.nn.Parameter(torch.tensor(1.0))

#         #Params - Shared Fusion
#         self.sigmoid = nn.Sigmoid()
#         self.attention_i = nn.MultiheadAttention(embed_dim=768, num_heads=4)
#         self.attention_t = nn.MultiheadAttention(embed_dim=768, num_heads=4)
#         self.W_gated_i = nn.Linear(768, 768)
#         self.W_gated_t = nn.Linear(768, 768)
#         self.alpha_1 = nn.Parameter(torch.tensor(1.0).to(self.device))
#         self.alpha_2 = nn.Parameter(torch.tensor(1.0).to(self.device))
#         self.beta_1 = nn.Parameter(torch.tensor(1.0).to(self.device))
#         self.beta_2 = nn.Parameter(torch.tensor(1.0).to(self.device))
#         self.proj = nn.Linear(768, 768)
# #         self.img_proj = nn.Linear(50, CFG.max_len)
        
        
#     def shared_fusion(self, text_embeddings, img_embeddings, attention_mask):

#         attention_mask = torch.tile(attention_mask.unsqueeze(2), dims=(1,1,text_embeddings.shape[-1])).to(self.device)

#         #F_IT
#         A_i, _ = self.attention_i(img_embeddings, img_embeddings, img_embeddings)
#         F_it = (text_embeddings * A_i) * attention_mask
#         # F_it = (text_embeddings * A_i)

#         #F_TI
#         A_t, _ = self.attention_t(text_embeddings, text_embeddings, text_embeddings)
#         A_t *= attention_mask
#         F_ti = img_embeddings * A_t

#         #G_I
#         G_i = self.sigmoid(self.W_gated_i(img_embeddings))

#         #G_T
#         # G_t = self.sigmoid(self.W_gated_t(text_embeddings))
#         G_t = self.sigmoid(self.W_gated_t(text_embeddings)) * attention_mask

#         #Computing SF output
#         F_1 = (G_i * F_ti) + ((1-G_i) * F_it)
#         F_2 = (G_t * F_ti) + ((1-G_t) * F_it)
#         F_i = (G_i*img_embeddings) + ((1-G_i) * F_ti)
#         F_t = (G_t*text_embeddings) + ((1-G_t) * F_it)
#         shared_fusion_output = (self.alpha_1 * F_1) + (self.alpha_2 * F_2) + (self.beta_1 * F_i) + (self.beta_2 * F_t)
#         return shared_fusion_output

#     def forward(self, input_image, input_ids, attention_mask, target_ids, mode="train", gen_params=None, **kwargs):
#         # do vision encoding and get cls token
#         #     vision_encoding = self.vision_encoder(input_image).last_hidden_state[:, 0, :]

#         #     # now we project vision encoding into BART space
#         #     vision_encoding = self.vision_to_text_space(vision_encoding)
        
#         # now we do text encoding
#         text_embeds = self.bart.model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

#         with torch.no_grad():
#             img_embeds = self.img_encoder(input_image).last_hidden_state[:,0,:]
#             img_embeds = self.proj(img_embeds).unsqueeze(1).expand(-1, text_embeds.shape[1], -1)
# #         img_embeds = self.img_proj(img_embeddings.transpose(1,2)).transpose(1,2)

#         shared_fusion_out = self.shared_fusion(text_embeds, img_embeds, attention_mask)

#         #     sequence_length = text_encoding.size(1)
#         #     vision_encoding = vision_encoding.unsqueeze(1).expand(-1, sequence_length, -1)

#         #     # now we fuse the two encodings
#         #     fused_encoding = self.lambda_vision * vision_encoding + self.lambda_text * text_encoding

#         combined_embeds = BaseModelOutput(last_hidden_state=shared_fusion_out)
#         if (mode == "train"):
#             final_output = self.bart(
#         #                 input_ids = input_ids, 
#                 encoder_outputs = combined_embeds,
#                 attention_mask = attention_mask,
#                 labels = target_ids,
#                 return_dict = True
#             )
#             return final_output.loss
#         else:
#             gen_result = self.bart.generate(
#         #                 input_ids = input_ids,
#                 encoder_outputs = combined_embeds,
#                 attention_mask = attention_mask,
#                 max_new_tokens = 32,
# #                 do_sample=True,
# #                 num_beams=8,
# #                 **gen_params,
#                 num_beams=4
#             )
#             gen_decoded = self.tkr.batch_decode(gen_result, skip_special_tokens=True)
#             return gen_result, gen_decoded

# class JointModel(nn.Module):
#     def __init__(self, tkr, device):
#         super(JointModel, self).__init__()
#         self.tkr = tkr 
#         self.device = device
#         self.TURBO = Model(self.tkr)
#         self.TARGEN = TARGEN(self.tkr)

#     def target_incorporation(self, caption, target_of_sarcasm):
#         concat = [i + ' </s> ' + j for i, j in zip(caption, target_of_sarcasm)]

#         encoded_dict = self.tkr(
#             concat,
#             max_length=CFG.max_len,
#             padding="max_length",
#             truncation=True,
#             return_tensors='pt',
#             add_prefix_space=True
#         )

#         concat_ids = encoded_dict['input_ids'].to(self.device)
#         concat_attn_mask = encoded_dict['attention_mask'].to(self.device)

#         return concat_ids, concat_attn_mask

#     def forward(self, 
#                 img, caption,
#                 caption_ids, exp_ids, ts_ids,
#                 caption_attn_mask, exp_attn_mask, ts_attn_mask,
#                 graph, 
#                 mode="train",
#                 **kwargs):

#         #  print(caption)
        
#         if (mode == "train" or mode == "val"):
#         #Stage 1: Target prediction
#             loss_ts, logits_ts = self.TARGEN(
#                 in_ids = caption_ids,
#                 in_attn_mask = caption_attn_mask, 
#                 out_ids = ts_ids, 
#                 out_attn_mask = ts_attn_mask,
#                 img = img,
#                 graph = graph
#             )
#             #  print(loss_ts.item())
#             #  print(logits_ts, logits_ts.shape)

#             _, target_of_sarcasm = self.TARGEN(
#                     in_ids = caption_ids,
#                     in_attn_mask = caption_attn_mask, 
#                     out_ids = exp_ids, 
#                     out_attn_mask = exp_attn_mask,
#                     img = img,
#                     graph = graph,
#                     mode="gen"
#             )
#             #  print(target_of_sarcasm)

#             #Stage 2: Incorporation of Predicted Target
#             caption_ids, caption_attn_mask = self.target_incorporation(caption, target_of_sarcasm)
#             #  print(self.tkr.batch_decode(caption_ids, skip_special_tokens=True))
#             #  print(caption_ids, caption_ids.shape)

#             #Stage 3: Prediction of Sarcasm Explanation
#             loss_exp, logits_exp = self.TURBO(
#                 in_ids = caption_ids,
#                 in_attn_mask = caption_attn_mask, 
#                 out_ids = exp_ids, 
#                 out_attn_mask = exp_attn_mask,
#                 img = img,
#                 graph = graph
#             )

#             return loss_ts, logits_ts, loss_exp, logits_exp

#         else:
#             #Stage 1: Generate Target of Sarcasm
#             _, ts_decoded = self.TARGEN(
#                     in_ids = caption_ids,
#                     in_attn_mask = caption_attn_mask, 
#                     out_ids = exp_ids, 
#                     out_attn_mask = exp_attn_mask,
#                     img = img,
#                     graph = graph,
#                     mode="gen"
#             )

#             #Stage 2: Incorporate Predicted Target
#             caption_ids, caption_attn_mask = self.target_incorporation(caption, ts_decoded)

#             #Stage 3: Prediction of Sarcasm Explanation
#             _, exp_decoded = self.TURBO(
#                 in_ids = caption_ids,
#                 in_attn_mask = caption_attn_mask, 
#                 out_ids = exp_ids, 
#                 out_attn_mask = exp_attn_mask,
#                 img = img,
#                 graph = graph,
#                 mode = "gen"
#             )

#             return ts_decoded, exp_decoded