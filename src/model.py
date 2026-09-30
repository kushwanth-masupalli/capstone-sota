"""
model.py - Dual-View Multimodal VLM with QLoRA and Auxiliary Clinical Classifier.

Architecture (Plan 2.1):
  * Dual-view tokens: 144 frontal + 144 lateral with learned view embeddings
  * 2-layer MLP Projector: Linear(D_enc -> D_llm) -> GELU -> Linear(D_llm -> D_llm)
  * Auxiliary head: Linear(2 * D_enc -> 14) for multi-label clinical prediction
  * Base LLM: Qwen/Qwen2.5-3B-Instruct with 4-bit NF4 QLoRA (r=32, alpha=64)
  * Loss: L_CE(report tokens) + 0.2 * L_BCE(14 findings)
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

IMAGE_PAD_TOKEN = "<|image_pad|>"


class MLPProjector(nn.Module):
    """2-layer MLP Projector mapping visual tokens to LLM embedding dimension."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualViewRAGVLM(nn.Module):
    """
    Complete Multimodal Model integrating Dual-View Visual Features,
    MLP Projector, Auxiliary 14-finding Classifier, and Qwen QLoRA LLM.
    """

    def __init__(
        self,
        llm_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        d_enc: int = 768,
        num_tokens_per_view: int = 144,
        num_classes: int = 14,
        cls_loss_weight: float = 0.2,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = True,
        device_map: Optional[str] = "auto",
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.d_enc = d_enc
        self.num_tokens_per_view = num_tokens_per_view
        self.cls_loss_weight = cls_loss_weight

        # 1. Learned View Embeddings
        self.view_embed_frontal = nn.Parameter(torch.zeros(1, num_tokens_per_view, d_enc))
        self.view_embed_lateral = nn.Parameter(torch.zeros(1, num_tokens_per_view, d_enc))
        nn.init.normal_(self.view_embed_frontal, std=0.02)
        nn.init.normal_(self.view_embed_lateral, std=0.02)

        # 2. Auxiliary 14-Finding Classifier Head
        self.classifier_head = nn.Linear(2 * d_enc, num_classes)

        # 3. Load LLM and Tokenizer
        print(f"Loading base LLM: {llm_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, trust_remote_code=True)
        if IMAGE_PAD_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_PAD_TOKEN]})
        self.image_pad_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name_or_path,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        d_llm = self.llm.config.hidden_size
        print(f"LLM hidden dimension: {d_llm}")

        # 4. 2-layer MLP Projector
        self.projector = MLPProjector(d_enc, d_llm)

        # 5. Apply QLoRA to LLM
        if load_in_4bit:
            self.llm = prepare_model_for_kbit_training(self.llm)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()

        # Keep projector, classifier head, and view embeddings in float32 for fp16 stability
        target_device = next(self.llm.parameters()).device
        self.projector.to(device=target_device, dtype=torch.float32)
        self.classifier_head.to(device=target_device, dtype=torch.float32)
        self.view_embed_frontal = nn.Parameter(self.view_embed_frontal.to(device=target_device, dtype=torch.float32))
        self.view_embed_lateral = nn.Parameter(self.view_embed_lateral.to(device=target_device, dtype=torch.float32))

    def set_stage(self, stage: int):
        """
        Stage 1: Warmup Projector + Classifier Head (LLM frozen)
        Stage 2: Full SFT (Projector + Head + LLM LoRA trainable)
        """
        if stage == 1:
            print("==> Configured for Stage 1: Projector + Classifier warmup (LLM frozen)")
            for param in self.llm.parameters():
                param.requires_grad = False
            for param in self.projector.parameters():
                param.requires_grad = True
            for param in self.classifier_head.parameters():
                param.requires_grad = True
            self.view_embed_frontal.requires_grad = True
            self.view_embed_lateral.requires_grad = True
        elif stage == 2:
            print("==> Configured for Stage 2: SFT (Projector + Head + LLM LoRA)")
            for name, param in self.llm.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
            for param in self.projector.parameters():
                param.requires_grad = True
            for param in self.classifier_head.parameters():
                param.requires_grad = True
            self.view_embed_frontal.requires_grad = True
            self.view_embed_lateral.requires_grad = True

    def forward(
        self,
        frontal_feats: torch.Tensor,
        lateral_feats: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        labels_14: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multimodal fusion and auxiliary loss.
        """
        device = next(self.llm.parameters()).device
        llm_dtype = self.llm.get_input_embeddings().weight.dtype

        frontal = frontal_feats.to(device=device, dtype=torch.float32)
        lateral = lateral_feats.to(device=device, dtype=torch.float32)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        if labels_14 is not None:
            labels_14 = labels_14.to(device=device, dtype=torch.float32)

        # 1. Add learned view embeddings in float32
        f_tokens = frontal + self.view_embed_frontal
        l_tokens = lateral + self.view_embed_lateral

        # 2. Auxiliary 14-Finding Classifier Head
        f_global = f_tokens.mean(dim=1)
        l_global = l_tokens.mean(dim=1)
        dual_global = torch.cat([f_global, l_global], dim=-1)  # [B, 2*D_enc]
        cls_logits = self.classifier_head(dual_global)         # [B, 14]

        loss_cls = torch.tensor(0.0, device=device, dtype=torch.float32)
        if labels_14 is not None:
            loss_cls = F.binary_cross_entropy_with_logits(cls_logits, labels_14)

        # 3. Project visual tokens to LLM dimension in float32, then cast to LLM dtype
        vis_tokens = torch.cat([f_tokens, l_tokens], dim=1)    # [B, 288, D_enc]
        vis_embeds = self.projector(vis_tokens).to(llm_dtype)  # [B, 288, D_llm]

        # 4. Replace image pad tokens in LLM input sequence
        # Get base model token embeddings
        embed_tokens = self.llm.get_input_embeddings()
        text_embeds = embed_tokens(input_ids)                  # [B, S, D_llm]

        # Fuse visual embeddings into image pad positions
        B, S, D_llm = text_embeds.shape
        inputs_embeds = text_embeds.clone()

        for b in range(B):
            pad_mask = (input_ids[b] == self.image_pad_token_id)
            pad_count = pad_mask.sum().item()
            if pad_count > 0:
                # Insert the visual tokens
                num_to_insert = min(pad_count, vis_embeds.shape[1])
                inputs_embeds[b, pad_mask][:num_to_insert] = vis_embeds[b, :num_to_insert]

        # 5. LLM forward pass
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        loss_lm = outputs.loss if labels is not None else None
        total_loss = None
        if loss_lm is not None:
            total_loss = loss_lm.float() + self.cls_loss_weight * loss_cls.float()

        return {
            "loss": total_loss,
            "loss_lm": loss_lm,
            "loss_cls": loss_cls,
            "logits": outputs.logits,
            "cls_logits": cls_logits,
        }

    @torch.no_grad()
    def generate_report(
        self,
        frontal_feat: torch.Tensor,
        lateral_feat: torch.Tensor,
        prompt_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        num_beams: int = 4,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 3,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate candidate reports given visual features and prompt tokens.
        """
        self.eval()
        device = next(self.llm.parameters()).device
        llm_dtype = self.llm.get_input_embeddings().weight.dtype

        # Ensure batch dimension
        if frontal_feat.ndim == 2:
            frontal_feat = frontal_feat.unsqueeze(0)
        if lateral_feat.ndim == 2:
            lateral_feat = lateral_feat.unsqueeze(0)
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        prompt_ids = prompt_ids.to(device)
        frontal = frontal_feat.to(device=device, dtype=torch.float32)
        lateral = lateral_feat.to(device=device, dtype=torch.float32)

        f_tokens = frontal + self.view_embed_frontal
        l_tokens = lateral + self.view_embed_lateral
        vis_tokens = torch.cat([f_tokens, l_tokens], dim=1)
        vis_embeds = self.projector(vis_tokens).to(llm_dtype)

        # Build inputs_embeds
        embed_tokens = self.llm.get_input_embeddings()
        text_embeds = embed_tokens(prompt_ids)
        inputs_embeds = text_embeds.clone()

        for b in range(prompt_ids.shape[0]):
            pad_mask = (prompt_ids[b] == self.image_pad_token_id)
            pad_count = pad_mask.sum().item()
            if pad_count > 0:
                num_to_insert = min(pad_count, vis_embeds.shape[1])
                inputs_embeds[b, pad_mask][:num_to_insert] = vis_embeds[b, :num_to_insert]

        if attention_mask is None:
            attention_mask = torch.ones(prompt_ids.shape, dtype=torch.long, device=device)

        gen_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode generated token IDs
        generated_texts = []
        for seq in gen_outputs:
            text = self.tokenizer.decode(seq, skip_special_tokens=True).strip()
            # Clean any trailing special tokens
            for marker in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
                text = text.replace(marker, "").strip()
            generated_texts.append(text)

        return generated_texts
