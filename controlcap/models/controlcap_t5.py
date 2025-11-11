# controlcap/models/controlcap_t5.py
import math
import copy
import random
import os
import gc
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision
from torchvision.models.vision_transformer import MLPBlock

from textblob import TextBlob
from peft import LoraConfig, get_peft_model

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_t5 import Blip2T5

from controlcap.models.tagging_heads.bert import BertConfig, BertModel
from controlcap.models.tagging_heads.asymmetric_loss import AsymmetricLoss

# HF helper for gentle logits nudging
from transformers.generation.logits_process import LogitsProcessor


# ------------------------------
# Cross-attention residual block
# ------------------------------
class CrossAttnBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, mlp_dim, dropout=0, attention_dropout=0):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.ln_g = norm_layer(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        self.ln_r = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, query_embeds, source_embeds):
        source_embeds = self.ln_g(source_embeds)
        x, attn = self.cross_attention(query_embeds, source_embeds, source_embeds)
        x = self.dropout(x)
        x = x + query_embeds
        y = self.ln_r(x)
        y = self.mlp(y)
        return x + y, attn


@registry.register_model("controlcap_t5")
class ControlCapT5(Blip2T5):
    """
    ControlCap-T5
      - BLIP2 vision (ViT) + Q-Former ➜ T5 seq2seq
      - CVEM/EBM modules to fuse vision & controls
      - Tag head to steer control words
      - Topic modeling: derive topics from image, refine with tag nouns,
        prefix into encoder and optionally bias decoding towards topic tokens.
    """

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.mem_log = kwargs.get("mem_log", False) or os.environ.get("RUN_MEM_LOG", "0") == "1"

        # Extract kwargs meant for Blip2T5
        base_kwargs = copy.deepcopy(kwargs)
        base_kwargs_keys = [
            "vit_model", "img_size", "drop_path_rate", "use_grad_checkpoint", "vit_precision",
            "freeze_vit", "num_query_token", "t5_model", "prompt", "max_txt_len", "apply_lemmatizer"
        ]
        for key in list(kwargs.keys()):
            if key not in base_kwargs_keys:
                base_kwargs.pop(key, None)

        # ---- init BLIP2 (ViT + Q-Former + T5 in FP) ----
        super().__init__(*args, **base_kwargs)

        # AMP for Q-Former + T5 section: {"auto","bf16","fp16","fp32"}
        self.llm_amp_mode = kwargs.get("llm_amp_mode", "auto")

        # optional tag chunking
        self.tag_chunk_size = kwargs.get("tag_chunk_size", None)
        self._tag_chunk_logged = False

        # decoding score length-norm (not used by default)
        self.length_normalize_scores = kwargs.get("length_normalize_scores", False)

        # -------------------------
        # Quantized T5 (4/8-bit)
        # -------------------------
        load_4_bit = kwargs.get("load_in_4bit", kwargs.get("load_4_bit", False))
        load_8_bit = kwargs.get("load_in_8bit", kwargs.get("load_8_bit", False))
        if load_4_bit and load_8_bit:
            raise ValueError("Only one of load_4_bit or load_8_bit can be True.")
        if load_4_bit or load_8_bit:
            try:
                from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
            except ImportError as e:
                raise ImportError("transformers with bitsandbytes support is required for quantization.") from e

            model_id = base_kwargs.get("t5_model", None)
            if model_id is None:
                raise ValueError("t5_model must be specified to use quantized loading.")

            # Avoid auto sharding across multiple GPUs (source of cuda:0/cuda:1 mixups)
            ddp_active = dist.is_available() and dist.is_initialized()
            if ddp_active:
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                torch.cuda.set_device(local_rank)
                device_map = {"": f"cuda:{local_rank}"}
            else:
                if torch.cuda.is_available():
                    device_map = {"": "cuda:0"}
                else:
                    device_map = {"": "cpu"}

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=load_4_bit,
                load_in_8bit=load_8_bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            # Replace FP T5 with quantized T5
            del self.t5_model
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                quantization_config=bnb_cfg,
                device_map=device_map,
            )
            self._is_quantized = True
            if self.mem_log:
                print(f"[INFO] Loaded quantized T5 ({'4-bit' if load_4_bit else '8-bit'}) on {device_map}")
        else:
            self._is_quantized = False

        # -------------------------------------
        # CVEM: build region+context embeddings
        # -------------------------------------
        input_image_size = self.visual_encoder.image_size
        patch_size = self.visual_encoder.patch_embed.patch_size[0]
        self._roi_align = torchvision.ops.RoIAlign(
            output_size=input_image_size // patch_size,
            spatial_scale=1 / patch_size,
            sampling_ratio=2
        )

        self.cvem_mlp = nn.Sequential(
            nn.Linear(self.visual_encoder.embed_dim * 2, self.visual_encoder.embed_dim),
            nn.ReLU(),
            nn.Linear(self.visual_encoder.embed_dim, self.visual_encoder.embed_dim)
        )
        self.cvem_tag_mlp = nn.Sequential(
            nn.Linear(self.visual_encoder.embed_dim * 2, self.visual_encoder.embed_dim),
            nn.ReLU(),
            nn.Linear(self.visual_encoder.embed_dim, self.visual_encoder.embed_dim)
        )

        # ------------------------------------
        # CEM: control embedding into T5 space
        # ------------------------------------
        self.cem_memory = nn.Parameter(torch.zeros(self.t5_model.model_dim))

        # -------------------------------------------------
        # EBM: bridge/control <-> vision via cross-attn MLP
        # -------------------------------------------------
        ebm_dim = 128
        ebm_num_heads = 8
        self.ebm_c2l_mlp = nn.Linear(self.t5_model.model_dim, ebm_dim)
        self.ebm_l2c_mlp = nn.Linear(ebm_dim, self.t5_model.model_dim)
        self.ebm_v2l_mlp = nn.Linear(self.visual_encoder.embed_dim, ebm_dim)
        self.ebm_l2v_mlp = nn.Linear(ebm_dim, self.visual_encoder.embed_dim)
        self.ebm_cl2vl_ca = CrossAttnBlock(num_heads=ebm_num_heads, hidden_dim=ebm_dim, mlp_dim=ebm_dim)
        self.ebm_vl2cl_ca = CrossAttnBlock(num_heads=ebm_num_heads, hidden_dim=ebm_dim, mlp_dim=ebm_dim)

        # -----------------------------------
        # Tagging head (predict region tags)
        # -----------------------------------
        tag_bert_config = BertConfig.from_json_file(
            kwargs.get("tag_bert_config", "controlcap/models/tagging_heads/tag_bert_config.json")
        )
        tag_bert_config.encoder_width = self.Qformer.config.encoder_width
        self.tag_head = BertModel(config=tag_bert_config, add_pooling_layer=False)
        del self.tag_head.embeddings
        for layer in self.tag_head.encoder.layer:
            del layer.attention
        tag_list = kwargs.get("tag_list", "controlcap/common/tagging/ram_tag_list.txt")
        with open(tag_list, "r") as fr:
            self.tag_list = [t.strip() for t in fr.readlines()]
        self.num_tags = len(self.tag_list)
        self.tag_labels = nn.Embedding(self.num_tags * 2, tag_bert_config.hidden_size)  # subj + obj
        self.tag_fc = nn.Linear(tag_bert_config.hidden_size, 1)
        self.tag_weight = 0.005
        self.tag_loss_function = AsymmetricLoss(gamma_neg=7, gamma_pos=0, clip=0.05)

        # --------------------------------------
        # Trainable subset selection + optional LoRA
        # --------------------------------------
        names = ["cvem", "cem", "tag", "ebm", "Qformer", "t5_proj"]
        self.finetune_llm = kwargs.get("finetune_llm", False)
        if self.finetune_llm:
            lora_config = LoraConfig(
                r=64, lora_alpha=128, lora_dropout=0.0,
                target_modules=["embed_tokens", "lm_head", "q", "v"]
            )
            self.t5_model = get_peft_model(self.t5_model, lora_config)
            if not self._is_quantized:
                self.t5_model.to(torch.float32)
            names.append("lora")

        trainable_params = 0
        all_params = 0
        params = [0] * len(names)
        for pname, p in self.named_parameters():
            all_params += p.numel()
            p.requires_grad = False
            for i, n in enumerate(names):
                if n in pname:
                    p.requires_grad = True
                    trainable_params += p.numel()
                    params[i] += p.numel()
                    break
        print(f"[ trainable ratio : {trainable_params / all_params}]")
        for i, n in enumerate(names):
            print(f"[{n} ratio : {params[i] / all_params}]")

        # =====================================================================
        # Topic knobs
        # =====================================================================
        self.topic_gen_k = kwargs.get("topic_gen_k", 3)
        self.topic_bias = kwargs.get("topic_bias", 1.2)
        self.topic_prefix_max_words = kwargs.get("topic_prefix_max_words", 6)

    # --------------------------
    # ROI pooling for region ViT
    # --------------------------
    def roi_align(self, image_embeds, samples):
        spatio_image_embeds = image_embeds[:, 1:]
        cls_image_embeds = image_embeds[:, 0][:, None]
        b, hw, c = spatio_image_embeds.shape
        h, w = int(math.sqrt(hw)), int(math.sqrt(hw))
        spatio_image_embeds = spatio_image_embeds.reshape(b, h, w, c).permute(0, 3, 1, 2)

        # extract roi features
        bboxes = samples["bboxes"]
        ids = samples["batch_idx"].to(torch.int64)
        rois = torch.cat([ids[:, None], bboxes], -1)
        spatio_rois_embeds = self._roi_align(spatio_image_embeds, rois)
        cls_image_embeds = cls_image_embeds[ids]

        # back to sequence
        bv = spatio_rois_embeds.shape[0]
        spatio_rois_embeds = spatio_rois_embeds.permute(0, 2, 3, 1).reshape(bv, -1, c)
        rois_embeds = torch.cat([cls_image_embeds, spatio_rois_embeds], 1)
        return rois_embeds

    # -------------------------------------------
    # CVEM forward: combine ROI + region image
    # -------------------------------------------
    def cvem_forward(self, samples, embeds):
        bz = len(samples["image"])
        image_embeds = embeds[:bz]
        region_embeds = embeds[bz:]
        rois_embeds = self.roi_align(image_embeds, samples)
        visual_embeds = torch.cat([rois_embeds, region_embeds], -1)
        visual_tag_embeds = self.cvem_tag_mlp(visual_embeds)
        visual_embeds = self.cvem_mlp(visual_embeds)
        return visual_embeds, visual_tag_embeds

    # -------------------------------------------
    # Tagging head forward (with optional chunk)
    # -------------------------------------------
    def tag_forward(self, samples, tag_embeds):
        bs = tag_embeds.shape[0]
        device = tag_embeds.device
        chunk = self.tag_chunk_size
        use_chunk = isinstance(chunk, int) and chunk > 0 and chunk < bs
        if use_chunk and not self._tag_chunk_logged:
            print(f"[INFO] Using tag head chunking with chunk size = {chunk}")
            self._tag_chunk_logged = True
        if not use_chunk:
            object_atts = torch.ones(tag_embeds.size()[:-1], dtype=torch.long, device=device)
            label_embed = self.tag_labels.weight.unsqueeze(0).repeat(bs, 1, 1)
            tagging_embed = self.tag_head(
                encoder_embeds=label_embed,
                encoder_hidden_states=tag_embeds,
                encoder_attention_mask=object_atts,
                return_dict=False,
                mode='tagging',
            )
            tag_logits = self.tag_fc(tagging_embed[0]).squeeze(-1)
            return tag_logits

        object_atts_full = torch.ones(tag_embeds.size()[:-1], dtype=torch.long, device=device)
        logits_chunks = []
        for st in range(0, bs, chunk):
            ed = min(st + chunk, bs)
            te = tag_embeds[st:ed]
            oa = object_atts_full[st:ed]
            label_embed = self.tag_labels.weight.unsqueeze(0).expand(ed - st, -1, -1).to(device)
            tagging_embed = self.tag_head(
                encoder_embeds=label_embed,
                encoder_hidden_states=te,
                encoder_attention_mask=oa,
                return_dict=False,
                mode='tagging',
            )
            logits = self.tag_fc(tagging_embed[0]).squeeze(-1)
            logits_chunks.append(logits)
        tag_logits = torch.cat(logits_chunks, dim=0)
        return tag_logits

    # -------------------------------------------------
    # Autocast helper for Q-Former + T5
    # -------------------------------------------------
    def _llm_autocast(self):
        mode = getattr(self, "llm_amp_mode", "auto")
        if mode == "auto":
            mode = "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "fp32"
        if mode == "bf16":
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if mode == "fp16":
            return torch.cuda.amp.autocast(dtype=torch.float16)
        return nullcontext()

    # ---------------------------------------------
    # CEM: Build control token embeddings for T5
    # ---------------------------------------------
    def cem_forward(self, tags, embeds):
        control_tokens = self.t5_tokenizer(
            tags,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        )
        emb_dev = self.t5_model.encoder.embed_tokens.weight.device
        control_ids = control_tokens.input_ids.to(emb_dev)
        control_embeds = self.t5_model.encoder.embed_tokens(control_ids)
        control_embeds = control_embeds + self.cem_memory.to(emb_dev, dtype=control_embeds.dtype)
        return control_embeds, control_tokens

    # ---------------------------------------------
    # EBM: fuse vision/control before Q-Former
    # ---------------------------------------------
    def ebm_forward(self, v_embeds, c_embeds):
        vl = self.ebm_v2l_mlp(v_embeds)
        cl = self.ebm_c2l_mlp(c_embeds)
        vl, _ = self.ebm_cl2vl_ca(vl, cl)
        cl, _ = self.ebm_vl2cl_ca(cl, vl)
        v_embeds = v_embeds + self.ebm_l2v_mlp(vl)
        c_embeds = c_embeds + self.ebm_l2c_mlp(cl)
        return v_embeds, c_embeds

    # ---------------------------------------------
    # Training forward (loss = LLM + tag)
    # ---------------------------------------------
    def forward(self, samples):
        image = torch.cat([samples["image"], samples["region_images"]], 0)

        with self.maybe_autocast(dtype=torch.float16):
            embeds = self.ln_vision(self.visual_encoder(image))
            visual_embeds, visual_tag_embeds = self.cvem_forward(samples, embeds)
            tag_logits = self.tag_forward(samples, visual_tag_embeds)
            control_words = self.prepare_control_words(samples, tag_logits)
            control_embeds, control_tokens = self.cem_forward(control_words, visual_embeds)
            visual_embeds, control_embeds = self.ebm_forward(visual_embeds, control_embeds)

        with self._llm_autocast():
            q_dtype = next(self.Qformer.parameters()).dtype
            visual_embeds = visual_embeds.to(dtype=q_dtype)
            object_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = self.query_tokens.expand(visual_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=visual_embeds,
                encoder_attention_mask=object_atts,
                return_dict=True,
            )
            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

            control_attn = control_tokens.attention_mask.to(inputs_t5.device)
            control_embeds = control_embeds.to(device=inputs_t5.device, dtype=inputs_t5.dtype)
            encoder_atts = torch.cat([atts_t5, control_attn], dim=1)
            inputs_embeds = torch.cat([inputs_t5, control_embeds], dim=1)

            tags = samples["tags"].to(torch.long)
            loss_tag = self.tag_loss_function(tag_logits, tags) * self.tag_weight

            output_tokens = self.t5_tokenizer(
                samples["caps"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(inputs_embeds.device)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss_llm = outputs.loss

            return {"loss": loss_llm + loss_tag, "loss_llm": loss_llm.detach(), "loss_tag": loss_tag.detach()}

    # --------------------------------------------------------
    # Build control word strings from tag head (train/eval)
    # --------------------------------------------------------
    def prepare_control_words(self, samples, tag_logits):
        control_words = []
        full_drop_ratio = self.kwargs.get("full_drop_ratio", 0.5)
        drop_ratio = self.kwargs.get("drop_ratio", 0.5)
        tag_thr = self.kwargs.get("tag_thr", 0.7)

        if self.training:
            for bz_idx, cap in enumerate(samples["caps"]):
                try:
                    s2 = TextBlob(cap).tags
                    tokens = [el[0] for el in s2]
                    infowords = [name for name, value in s2 if ("NN" in value) or ("JJ" in value)]
                    nouns = [name for name, value in s2 if ("NN" in value)]
                    if len(infowords) > 0:
                        words = []
                        for word in infowords:
                            st_idx = tokens.index(word)
                            ed_idx = st_idx + 1
                            while (ed_idx < len(tokens)) and (tokens[ed_idx] in nouns):
                                ed_idx += 1
                            words.append(" ".join(tokens[st_idx:ed_idx]))
                    else:
                        words = [""]
                except Exception:
                    words = [""]
                tag_idxs = samples["tags"]
                stags = [self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][:self.num_tags])]
                otags = [self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][self.num_tags:])]
                tags = list(set(stags + otags + words))
                if np.random.uniform(0, 1) < full_drop_ratio or len(tags) == 0:
                    control_word = ""
                else:
                    sl = torch.from_numpy(np.random.uniform(0, 1, len(tags)) > drop_ratio)
                    control_word = [tags[i] for i in torch.nonzero(sl)]
                    random.shuffle(control_word)
                    control_word = ",".join(control_word)
                control_words.append(control_word + "|")
            return control_words

        # eval
        tag_scores = tag_logits.sigmoid()
        tag_idxs = (tag_scores > tag_thr).to(torch.long)
        stags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][:self.num_tags])]
                 for bz_idx in range(len(tag_idxs))]
        otags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][self.num_tags:])]
                 for bz_idx in range(len(tag_idxs))]
        tags = [list(set(stag + otag)) for stag, otag in zip(stags, otags)]

        first_word_control = self.kwargs.get("first_word_control", False)
        if first_word_control:
            first_words = []
            for bz_idx, cap in enumerate(samples["caps"]):
                try:
                    s2 = TextBlob(cap).tags
                    tokens = [el[0] for el in s2]
                    infowords = [name for name, value in s2 if ("NN" in value) or ("JJ" in value)]
                    nouns = [name for name, value in s2 if ("NN" in value)]
                    words = []
                    if len(infowords) > 0:
                        for word in infowords:
                            st_idx = tokens.index(word)
                            ed_idx = st_idx + 1
                            while (ed_idx < len(tokens)) and (tokens[ed_idx] in nouns):
                                ed_idx += 1
                            words.append(" ".join(tokens[st_idx:ed_idx]))
                except Exception:
                    words = []
                first_words.append([words[0]] if len(words) > 0 else [])
            tags = [fword + tag for fword, tag in zip(first_words, tags)]

        controls = samples.get("controls", None)
        if controls is not None:
            tags = [control + tag for control, tag in zip(controls, tags)]

        for control_tag in tags:
            control_tag = list(dict.fromkeys(control_tag))  # dedupe preserve order
            control_words.append(",".join(control_tag) + "|")
        return control_words, stags, otags

    # ---------------------------------------------------------
    # Inference: region captions + scores + tag sets
    # ---------------------------------------------------------
    def predict_answers(self, samples, *args, **kwargs):
        image = torch.cat([samples["image"], samples["region_images"]], 0)

        with self.maybe_autocast(dtype=torch.float16):
            embeds = self.ln_vision(self.visual_encoder(image))
            visual_embeds, visual_tag_embeds = self.cvem_forward(samples, embeds)
            tag_logits = self.tag_forward(samples, visual_tag_embeds)
            control_words, stags, otags = self.prepare_control_words(samples, tag_logits)
            control_embeds, control_tokens = self.cem_forward(control_words, visual_embeds)
            visual_embeds, control_embeds = self.ebm_forward(visual_embeds, control_embeds)

        with self._llm_autocast():
            q_dtype = next(self.Qformer.parameters()).dtype
            visual_embeds = visual_embeds.to(dtype=q_dtype)
            object_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = self.query_tokens.expand(visual_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=visual_embeds,
                encoder_attention_mask=object_atts,
                return_dict=True,
            )
            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

            control_attn = control_tokens.attention_mask.to(inputs_t5.device)
            control_embeds = control_embeds.to(device=inputs_t5.device, dtype=inputs_t5.dtype)
            encoder_atts = torch.cat([atts_t5, control_attn], dim=1)
            inputs_embeds = torch.cat([inputs_t5, control_embeds], dim=1)

            llm_kwargs = {
                "do_sample": False,
                "num_beams": self.kwargs.get("num_beams", 5),
                "max_new_tokens": self.kwargs.get("max_new_tokens", 10),
                "min_length": self.kwargs.get("min_length", 1),
                "length_penalty": self.kwargs.get("length_penalty", -1),
                "repetition_penalty": self.kwargs.get("repetition_penalty", None),
                "num_return_sequences": self.kwargs.get("num_return_sequences", 1),
                "top_p": self.kwargs.get("top_p", None),
                "temperature": self.kwargs.get("temperature", None),
            }
            for k in [k for k, v in list(llm_kwargs.items()) if v is None]:
                llm_kwargs.pop(k)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                output_scores=True,
                return_dict_in_generate=True,
                **llm_kwargs
            )

            sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs["sequences"]
            if (hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None) or ("sequences_scores" in outputs):
                seq_scores = outputs.sequences_scores if hasattr(outputs, "sequences_scores") else outputs["sequences_scores"]
                scores = torch.exp(seq_scores)
            else:
                scores = torch.ones(sequences.size(0), device=sequences.device)

            l = sequences.shape[1]
            sequences = sequences.reshape(-1, l)
            scores = scores.reshape(-1).detach().cpu().numpy().tolist()
            captions = self.t5_tokenizer.batch_decode(sequences, skip_special_tokens=True)

        if self._apply_lemmatizer:
            captions = self._lemmatize(captions)

        output = []
        for id, caption, score, stag, otag in zip(samples["ids"], captions, scores, stags, otags):
            output.append({"id": id, "caption": caption, "score": score, "tag_set1": stag, "tag_set2": otag})
        return output

    # ---------------------------------------------------------
    # Config builder
    # ---------------------------------------------------------
    @classmethod
    def from_config(cls, cfg):
        model = cls(**cfg)
        if cfg.pretrained is not None:
            model.load_checkpoint(url_or_filename=cfg.pretrained)
        return model

    # =====================================================================
    # ==================  [Topic Modeling Helpers]  ========================
    # =====================================================================

    def _clean_sentencepiece_tokens(self, text):
        import re
        seg = text.split("scene_topics:", 1)[-1]
        raw = re.split(r"[,\|\n;]+", seg)
        w = []
        for t in raw:
            t = t.strip().lower()
            t = re.sub(r"[^a-z\s]+", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            if 1 <= len(t) <= 20:
                w.append(t)
        seen, out = set(), []
        for s in w:
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _build_topic_prefix(self, topic_info, max_words=None):
        max_words = max_words or self.topic_prefix_max_words
        parts = []
        for st in topic_info.get("subtopics", []):
            parts.append(", ".join(st["keywords"][:max_words]))
        if not parts:
            return ""
        return f"scene_topics: {' | '.join(parts)}."

    class TopicBiasProcessor(LogitsProcessor):
        def __init__(self, token_ids, bias=1.2):
            super().__init__()
            self.ids = list(set(int(x) for x in token_ids))
            self.bias = float(bias)

        def __call__(self, input_ids, scores):
            if not self.ids:
                return scores
            scores[:, self.ids] += self.bias
            return scores

    def _topic_logits_processor_from_keywords(self, keywords, bias=None):
        if not keywords:
            return None
        tok_ids = []
        for w in keywords:
            ids = self.t5_tokenizer(w, add_special_tokens=False).input_ids
            if isinstance(ids, list) and ids:
                if isinstance(ids[0], list):
                    ids = ids[0]
                tok_ids.append(ids[0])
        if not tok_ids:
            return None
        return [self.TopicBiasProcessor(tok_ids, bias=bias or self.topic_bias)]

    # -------------------------------
    # Visually-conditioned topic gen
    # -------------------------------
    def _global_visual_tokens_for_t5(self, image_tensor):
        device = next(self.parameters()).device
        with self.maybe_autocast(dtype=torch.float16):
            vis_tokens = self.ln_vision(self.visual_encoder(image_tensor))  # [B, N, Dv]
            object_atts = torch.ones(vis_tokens.size()[:-1], dtype=torch.long, device=device)
            query_tokens = self.query_tokens.expand(vis_tokens.shape[0], -1, -1)
            q_out = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=vis_tokens,
                encoder_attention_mask=object_atts,
                return_dict=True,
            )
            inputs_t5 = self.t5_proj(q_out.last_hidden_state)                   # [B, nq, Ht5]
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long, device=device)
        return inputs_t5, atts_t5

    def derive_topics_from_image(self, image_tensor, k=None, max_new_tokens=16):
        k = k or self.topic_gen_k
        device = next(self.parameters()).device

        with torch.no_grad():
            inputs_t5, atts_t5 = self._global_visual_tokens_for_t5(image_tensor)

        instr = f"scene_topics: list {k} short keywords about the whole scene, comma-separated."
        tok = self.t5_tokenizer(instr, return_tensors="pt").to(device)
        instr_ids = tok.input_ids
        instr_emb = self.t5_model.encoder.embed_tokens(instr_ids)  # [1, L, H]
        instr_att = tok.attention_mask

        enc_embeds = torch.cat([inputs_t5, instr_emb], dim=1)
        enc_atts = torch.cat([atts_t5, instr_att], dim=1)

        with self._llm_autocast():
            out_ids = self.t5_model.generate(
                inputs_embeds=enc_embeds,
                attention_mask=enc_atts,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.t5_tokenizer.eos_token_id,
                return_dict_in_generate=False
            )

        text = self.t5_tokenizer.decode(out_ids[0], skip_special_tokens=True)
        keywords = self._clean_sentencepiece_tokens(text)
        keywords = keywords[:k] if k and k > 0 else keywords
        if not keywords:
            keywords = ["scene"]

        return {"main_topic_keywords": keywords, "subtopics": [{"keywords": keywords, "score": 1.0}]}

    # ----------------------------
    # Tag-noun filter (bounds-safe)
    # ----------------------------
    def _keep_top_tag_nouns(self, tag_logits, topk=8, thr=0.7):
        """
        Select high-confidence SUBJECT nouns from tag head.
        Shape: tag_logits [B, 2*num_tags]; we only read [:, :num_tags].
        """
        with torch.no_grad():
            subj = tag_logits[..., : self.num_tags]      # [B, num_tags]
            probs = subj.sigmoid()
            mask = probs > thr
            out = []
            for b in range(probs.size(0)):
                p = probs[b]
                m = mask[b]
                cand_idx = torch.nonzero(m).flatten()
                if cand_idx.numel() == 0:
                    k = min(topk, p.numel())
                    cand_idx = torch.topk(p, k=k).indices
                else:
                    _, order = torch.sort(p[cand_idx], descending=True)
                    cand_idx = cand_idx[order[:topk]]

                words = []
                for idx in cand_idx.tolist():
                    if 0 <= idx < self.num_tags:
                        w = self.tag_list[idx]
                        if w and any(ch.isalpha() for ch in w):
                            words.append(w)
                # dedupe preserving order
                seen, uniq = set(), []
                for w in words:
                    if w not in seen:
                        seen.add(w)
                        uniq.append(w)
                out.append(uniq)
            return out

    # ----------------------------
    # Refine topics with tag nouns
    # ----------------------------
    def _refine_topics_with_tags(self, topic_info, tag_logits, k=None):
        """
        Merge image-derived topics with top tag nouns to anchor topics to detected objects.
        """
        k = k or self.topic_gen_k
        tag_nouns = self._keep_top_tag_nouns(tag_logits, topk=8, thr=0.7)  # list[str] per sample
        # we only need a single global refinement per batch (topics are per-image)
        nouns = tag_nouns[0] if len(tag_nouns) > 0 else []
        main = topic_info.get("main_topic_keywords", [])
        merged = []
        for w in main + nouns:
            if w not in merged and isinstance(w, str) and len(w) > 0:
                merged.append(w)
        merged = merged[: max(k, len(main))] if k else merged

        return {
            "main_topic_keywords": merged if merged else main,
            "subtopics": [{"keywords": merged if merged else main, "score": 1.0}],
        }

    # ---------------------------------------------------------
    # Topic-guided inference (derive ➜ refine ➜ bias)
    # ---------------------------------------------------------
    def predict_answers_with_topics(self, samples, *args, **kwargs):
        k = kwargs.get("topic_gen_k", self.topic_gen_k)
        base_topic = self.derive_topics_from_image(samples["image"], k=k)
        # refine topics using tag nouns (computed below)
        image = torch.cat([samples["image"], samples["region_images"]], 0)

        with self.maybe_autocast(dtype=torch.float16):
            embeds = self.ln_vision(self.visual_encoder(image))
            visual_embeds, visual_tag_embeds = self.cvem_forward(samples, embeds)
            tag_logits = self.tag_forward(samples, visual_tag_embeds)

            topic_info = self._refine_topics_with_tags(base_topic, tag_logits, k=k)
            topic_prefix = self._build_topic_prefix(topic_info)

            control_words, stags, otags = self.prepare_control_words(samples, tag_logits)
            cw = control_words[0] if isinstance(control_words, tuple) else control_words
            if topic_prefix:
                cw = [f"{topic_prefix} {x}" if x else f"{topic_prefix} " for x in cw]

            control_embeds, control_tokens = self.cem_forward(cw, visual_embeds)
            visual_embeds, control_embeds = self.ebm_forward(visual_embeds, control_embeds)

        with self._llm_autocast():
            q_dtype = next(self.Qformer.parameters()).dtype
            visual_embeds = visual_embeds.to(dtype=q_dtype)
            object_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = self.query_tokens.expand(visual_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=visual_embeds,
                encoder_attention_mask=object_atts,
                return_dict=True,
            )
            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

            control_attn = control_tokens.attention_mask.to(inputs_t5.device)
            control_embeds = control_embeds.to(device=inputs_t5.device, dtype=inputs_t5.dtype)
            encoder_atts = torch.cat([atts_t5, control_attn], dim=1)
            inputs_embeds = torch.cat([inputs_t5, control_embeds], dim=1)

            bias = kwargs.get("topic_bias", self.topic_bias)
            lp = self._topic_logits_processor_from_keywords(topic_info["main_topic_keywords"], bias=bias)

            llm_kwargs = {
                "do_sample": False,
                "num_beams": self.kwargs.get("num_beams", 5),
                "max_new_tokens": self.kwargs.get("max_new_tokens", 10),
                "min_length": self.kwargs.get("min_length", 1),
                "length_penalty": self.kwargs.get("length_penalty", -1),
                "repetition_penalty": self.kwargs.get("repetition_penalty", None),
                "num_return_sequences": self.kwargs.get("num_return_sequences", 1),
                "top_p": self.kwargs.get("top_p", None),
                "temperature": self.kwargs.get("temperature", None),
                "logits_processor": lp
            }
            for k_ in [k for k, v in list(llm_kwargs.items()) if v is None]:
                llm_kwargs.pop(k_)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                output_scores=True,
                return_dict_in_generate=True,
                **llm_kwargs
            )

            sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs["sequences"]
            if (hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None) or ("sequences_scores" in outputs):
                seq_scores = outputs.sequences_scores if hasattr(outputs, "sequences_scores") else outputs["sequences_scores"]
                scores = torch.exp(seq_scores)
            else:
                scores = torch.ones(sequences.size(0), device=sequences.device)

            l = sequences.shape[1]
            sequences = sequences.reshape(-1, l)
            scores = scores.reshape(-1).detach().cpu().numpy().tolist()
            captions = self.t5_tokenizer.batch_decode(sequences, skip_special_tokens=True)

        if self._apply_lemmatizer:
            captions = self._lemmatize(captions)

        output = []
        for id, caption, score, stag, otag in zip(samples["ids"], captions, scores, stags, otags):
            output.append(
                {
                    "id": id,
                    "caption": caption,
                    "score": score,
                    "tag_set1": stag,
                    "tag_set2": otag,
                    "topics": topic_info
                }
            )

        _dbg_n = int(os.environ.get("TOPIC_DEBUG_N", "0"))
        if _dbg_n > 0:
            for i, item in enumerate(output[:_dbg_n]):
                t = item.get("topics", {})
                print(f"[TOPICS] id={item['id']} :: {t.get('main_topic_keywords', [])} | sub={t.get('subtopics', [])}")

        return output
