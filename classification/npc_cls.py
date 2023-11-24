from transformers import BertTokenizer
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
import torch
from torch import nn
import contextlib


class npc_cls(nn.Module):
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
        
    @staticmethod
    def init_Qformer(num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @staticmethod
    def init_tokenizer(truncation_side="left"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    
    
    def __init__(self, totallen, device="cpu"):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        self.Qformer, self.query_tokens = self.init_Qformer(32, 768)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.ln_vision = LayerNorm(768)
        self.fc_1 = nn.Linear(768, 2)
        self.at1 = nn.ReLU()
        # self.fc_3 = nn.Linear(196 * 768, 196)
        # self.fc_4 = nn.Linear(196 ,1)
        # self.at2 = nn.ReLU()
        self.device = device
        for name, param in self.Qformer.named_parameters():
            param.requires_grad=False
    
    def forward(self, img, input_ids, attention_mask):
        bs = img.shape[0]
        # x0 = self.fc_3(img.reshape(bs, -1))
        # x0 = self.at2(x0)
        # x0 = self.fc_4(x0)
        with self.maybe_autocast():
            img_embeds = self.ln_vision(img)
        img_atts = torch.ones(img_embeds.shape[:-1], dtype=torch.long).to(img.device)
        query_tokens = self.query_tokens.expand(img_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.shape[:-1], dtype=torch.long).to(img.device)
        Qformer_atts = torch.cat([query_atts, attention_mask], dim=1)
        query_output = self.Qformer.bert(
            input_ids, 
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=img_embeds,
            encoder_attention_mask=img_atts,
            return_dict=True,
        )
        lhs = query_output.last_hidden_state[:,:query_tokens.size(1),:].reshape(bs, -1)
        lhs = self.fc_1(lhs)
        lhs = lhs.mean(dim=1)
        # return torch.sigmoid(torch.cat([x0, lhs], axis=1))
        return torch.sigmoid(lhs)
        
        

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
        
