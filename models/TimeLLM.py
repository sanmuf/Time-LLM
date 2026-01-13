from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import linregress

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
    #    self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
     #   x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=4, stride=4):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = patch_len
        self.stride = stride

        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained("/media/oscar6/6F682A90B86D8F9F1/ss/hf_cache/models")
            #self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                     "/media/oscar6/6F682A90B86D8F9F1/ss/hf_cache/models",
                    #'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                     "/media/oscar6/6F682A90B86D8F9F1/ss/hf_cache/models",
                    #'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                     "/media/oscar6/6F682A90B86D8F9F1/ss/hf_cache/models/tokenizer.model",
                    #'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                     "/media/oscar6/6F682A90B86D8F9F1/ss/hf_cache/models/tokenizer.model",
                    #'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'Please use the binary classification prediction software to build the failure based on the time series.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
       # self.reprogramming_layer = ReprogrammingLayerForCls( configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 1)
        self.head_nf = self.d_ff * (self.patch_nums+2)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        elif self.task_name == 'classification':
        #    self.output_projection = nn.Sequential(
        #        nn.Linear(self.d_ff, 256),
        #        nn.ReLU(),
        #        nn.Dropout(configs.dropout),
        #        nn.Linear(256, 128),
        #        nn.ReLU(),
        #        nn.Dropout(configs.dropout),
        #        nn.Linear(128, configs.c_out)
        #    )
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        self.attn_proj = nn.Linear(self.d_ff, 1)
        self.dllm_proj=nn.Linear(self.d_llm, 1)
        self.last_proj=nn.Linear(self.d_llm,self.d_ff)
        self.feat_proj=nn.Linear(33,self.d_llm)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,x_text=None,x_feat=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        elif self.task_name == 'classification':
            logits = self.classify(x_enc, x_mark_enc,x_text,x_feat)
            return logits
        return None
    
    def classify(self, x_enc, x_mark_enc,x_text=None,x_feat=None):
    #    x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.size()

        x_flat = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_flat, dim=1)[0]
        max_values = torch.max(x_flat, dim=1)[0]
        medians = torch.median(x_flat, dim=1).values
        lags = self.calcute_lags(x_flat)
        trends = x_flat.diff(dim=1).sum(dim=1) 

       # prompt = []
       # for b in range(x_flat.shape[0]):
        #    trend_dir = 'upward' if trends[b, 0].item() > 0 else 'downward'  
         #   prompt_ = (
          #      f"<|start_prompt|>Task: classify build failure;"
           #     f" min {min_values[b].tolist()[0]}, max {max_values[b].tolist()[0]}, median {medians[b].tolist()[0]}, "
            #    f"trend {trend_dir}, top 5 lags {lags[b].tolist()}<|end_prompt|>"
            #)
            #prompt.append(prompt_)
        prompt = []
        history_len_5 = 5
        history_len_10 = 10
        for b in range(x_flat.shape[0]):
            last_fail = int(x_flat[b, -1, 0].item())

            recent_5 = x_flat[b, -history_len_5:, 0].cpu().numpy()
            min_5 = recent_5.min()
            max_5 = recent_5.max()
            median_5 = np.median(recent_5)
            mean_5 = recent_5.mean()

            recent_10 = x_flat[b, -history_len_10:, 0].cpu().numpy()
            min_10 = recent_10.min()
            max_10 = recent_10.max()
            median_10 = np.median(recent_10)
            mean_10 = recent_10.mean()

            x_idx = np.arange(len(recent_10))
            slope, _, _, _, _ = linregress(x_idx, recent_10)
            trend_dir = 'increasing failure trend' if slope > 0 else 'decreasing failure trend'

            lag_v = lags[b].tolist()
            # prompt_ = (
            #     f"<|start_prompt|>"
            #     f"Task: Predict if the current software build will fail (1). "
            #  #   f"Dataset description: {self.description} "
            #     f"Last build failed = {last_fail}, "
            #     f"recent 5 builds failure rate: median={median_5:.2f}, mean={mean_5:.2f}; "
            #     f"recent 10 builds failure rate: median={median_10:.2f}, mean={mean_10:.2f}; "
            #     f"trend={trend_dir}, "
            #     f"top 5 lag correlations = {lag_v}. "
            #     f"The following are the time series data of the build results and the compressed text features"
            #     f"Please only answer 1 if the build failed or 0 if the build succeeded.<|end_prompt|>"
            # )
            prompt_ = (
                "<|start_prompt|>\n"
                "Role: You are an experienced software build outcome prediction expert.\n"
                "Task: Predict Build Result (0=Success, 1=Fail).\n"
                "Model: [History(H)] + [Text_Summary(T)] + [Stats(S)] -> Result\n"
                "Traits for SUCCESS:\n"
                "1) Stability in H (many recent 1s)\n"
                "2) Low-risk text (docs/format/style)\n"
                "3) Low churn in S\n"
                "Traits for FAILURE:\n"
                "1) Instability in H (many recent 0s)\n"
                "2) High-risk text (refactor/deps)\n"
                "3) High churn in S\n"
                "Examples:\n"
                "1) H:[1,1,1] T:Update readme S:Lines<5 -> 0\n"
                "2) H:[0,1,1] T:Fix typo in comment S:Lines<10 -> 0\n"
                "3) H:[0,0,1] T:Minor variable rename S:Lines<20 -> 0\n"
                "4) H:[0,0,0] T:Refactor core logic S:Lines>500 -> 1\n"
                "5) H:[1,1,1] T:Upgrade major dependency S:Files>50 -> 1\n"
                "6) H:[0,1,0] T:Merge experimental branch S:Complex -> 1\n"
                f"Current stats: last={last_fail}, "
                f"recent5(min={min_5:.2f}, max={max_5:.2f}, median={median_5:.2f}, mean={mean_5:.2f}), "
                f"recent10(min={min_10:.2f}, max={max_10:.2f}, median={median_10:.2f}, mean={mean_10:.2f}), "
                f"trend={trend_dir}, top_lags={lag_v}.\n"
                "Target: output only the result digit (0 or 1), no explanation.\n"
                "H:history_data T:text_summary S:stats_data ->\n"
                "<|end_prompt|>"
            )
            prompt.append(prompt_)

        x_enc = x_flat.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids.to(x_enc.device)) 
     #   if(x_text is not None and x_text.nelement()>0):
     #      token_embeddings=self.llm_model.get_input_embeddings()(x_text.to(x_enc.device))
        token_embeddings= self.encode_with_chunks(self.tokenizer,self.llm_model,x_text,chunk_size=256,device=x_enc.device) if(x_text is not None ) else torch.tensor([]).to(x_enc.device)
        token_embeddings.to(x_enc.device)
        token_embeddings=token_embeddings.unsqueeze(1)
        feat_embeddings=self.feat_proj(x_feat.to(torch.bfloat16)).unsqueeze(1) if(x_feat is not None and x_feat.nelement()>0) else torch.tensor([]).to(x_enc.device)
        feat_embeddings.to(x_enc.device)
        
        # keywords = ["failure", "success", "error", "bug", "pass", "succeed"]
        # keyword_ids = sum([self.tokenizer.encode(w, add_special_tokens=False) for w in keywords], [])
        # source_embeddings = self.word_embeddings[keyword_ids] 


        keywords = ["failure", "success"]
        keyword_ids = sum([self.tokenizer.encode(w, add_special_tokens=False) for w in keywords], [])
        keyword_emb = self.word_embeddings[keyword_ids]  # [k, hidden_size]

        # 归一化
        all_emb = F.normalize(self.word_embeddings, dim=1)  # [vocab_size, hidden_size]
        keyword_emb = F.normalize(keyword_emb, dim=1)

        # 计算相似度
        sims = torch.matmul(all_emb, keyword_emb.T)  # [vocab_size, k]
        topk_vals, topk_ids = torch.topk(sims.max(dim=1).values, k=50)  # 选前 50 个相关 token

        source_embeddings = self.word_embeddings[topk_ids]


        #source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_pe = x_enc.permute(0, 2, 1).contiguous()  # (B,N,T)
        enc_out, n_vars = self.patch_embedding(x_pe.to(torch.bfloat16)) 
        # corss attention
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        enc_out[:,-1,:]=0  # zero out the last patch embedding to avoid information leakage

        # token_embeddings=self.reprogramming_layer(token_embeddings, source_embeddings, source_embeddings) if(x_text is not None ) else token_embeddings
        # feat_embeddings=self.reprogramming_layer(feat_embeddings, source_embeddings, source_embeddings) if(x_feat is not None and x_feat.nelement()>0) else feat_embeddings
        
        if(x_text is not None ):
            llama_enc_out = torch.cat([prompt_embeddings, enc_out,token_embeddings,feat_embeddings], dim=1) 
        else:
            llama_enc_out = torch.cat([prompt_embeddings, enc_out,feat_embeddings], dim=1)
        llm_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = self.last_proj(llm_out)

        #pooled = dec_out.mean(dim=1) 
        #attn_weights = torch.softmax(self.attn_proj(dec_out), dim=1)  # (B,L,1)
        #pooled = torch.sum(attn_weights * dec_out, dim=1)

        #pooled = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)              
        #logits = self.output_projection(pooled)     

        #logits = logits.view(B, N, -1).mean(dim=1)   
        #logits = logits.unsqueeze(1)    
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        print("dec_out shape:",dec_out.shape)
        print("patch nums:",self.patch_nums)
        print(dec_out[:, :, :, -(self.patch_nums+2):].shape)
        dec_out = self.output_projection(dec_out[:, :, :, -(self.patch_nums+2):])
        dec_out = dec_out.permute(0, 2, 1).contiguous()             

        return dec_out
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags
    
    def encode_with_chunks(self,tokenizer, encoder, batch_text, chunk_size=256, device="cuda"):
      #  encoder.eval()
      #  for p in encoder.parameters():
      #      p.requires_grad_(False)
        batch_embeddings = []  

        for text in batch_text:
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=False,   
                add_special_tokens=False
            )["input_ids"].squeeze(0)

            chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]

            chunk_embeddings = []
            for chunk in chunks:
                inputs = {"input_ids": chunk.unsqueeze(0).to(device)}
                with torch.no_grad():
                    outputs = encoder(**inputs, use_cache=False)
                emb = outputs.last_hidden_state[:, 0, :].detach().cpu() 
                del outputs, inputs
                chunk_embeddings.append(emb)

            chunk_embeddings = torch.cat(chunk_embeddings, dim=0).to(device)  
            attn_scores = self.dllm_proj(chunk_embeddings)     
            attn_weights = torch.softmax(attn_scores, dim=0)  
            doc_embed = torch.sum(attn_weights * chunk_embeddings, dim=0) 

            batch_embeddings.append(doc_embed.unsqueeze(0).cpu())
            del chunk_embeddings, attn_scores, attn_weights, doc_embed

        return torch.cat(batch_embeddings, dim=0).to(device)  
      


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.query_projection2 = nn.Linear(d_llm, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads
        
        if target_embedding.shape[-1]==self.out_projection.out_features:
            target_embedding = self.query_projection2(target_embedding).view(B, L, H, -1)
        else:
            target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class ReprogrammingLayerForCls(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayerForCls, self).__init__()

        self.n_heads = n_heads
        d_keys = d_keys or (d_model // n_heads)
        self.d_keys = d_keys
        self.d_llm = d_llm

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)

        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.dropout = nn.Dropout(attention_dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))


    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        target_embedding = torch.cat([cls_tokens, target_embedding], dim=1) 

        target_proj = self.query_projection(target_embedding).view(B, L+1, H, -1)
        source_proj = self.key_projection(source_embedding).view(S, H, -1)
        value_proj = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_proj, source_proj, value_proj)  

        B, L1, H, E = out.shape
        out = out.reshape(B, L1, H*E)  
        out = self.out_projection(out) 

        return out

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        return reprogramming_embedding
