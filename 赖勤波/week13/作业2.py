"""
正在加载权重: Qwen/Qwen3-0.6B\model.safetensors
权重加载成功！
User: 杭州的天气怎么样？
Assistant: <think>
好的，用户问杭州的天气怎么样。首先，我需要确定用户的需求是什么。可能他们想了解当前的天气情况，或者计划去杭州旅游，需要知道天气。但用户没有具体说明，所以得先确认一下。

接下来，我需要考虑杭州的天气情况。杭州位于中国浙江省，属于亚热带季风气候，四季分明。春秋天气变化较大，夏天可能比较热，冬天则比较冷。不过具体到每个季节，可能需要更详细的信息。

然后，用户可能想知道当前的天气，所以应该提供最新的天气预报。但问题在于，用户没有提到时间，所以可能需要询问。不过根据之前的对话，用户可能已经知道，或者需要进一步的信息。

另外，用户可能对杭州的天气感兴趣，比如是否适合户外活动，或者是否需要带伞。这时候可以给出一些建议，比如夏天带防晒用品，冬天带保暖衣物。

还要注意用户可能的深层需求，比如是否需要知道天气
推理完成。
"""

import os
import re
import torch
import torch.nn as nn
from pathlib import Path
from tokenizers import Tokenizer
from safetensors.torch import load_file

# ==========================================
# ⚙️ 配置区
# ==========================================
# Windows 路径示例 (根据你的报错信息，你是在 Windows 上运行)
LOCAL_MODEL_PATH = "Qwen/Qwen3-0.6B"

CHOOSE_MODEL = "0.6B"

USE_REASONING_MODEL = True
USE_INSTRUCT_MODEL = False
USE_BASE_MODEL = not (USE_REASONING_MODEL or USE_INSTRUCT_MODEL)

if (USE_BASE_MODEL + USE_REASONING_MODEL + USE_INSTRUCT_MODEL) != 1:
    raise AttributeError("Only one of the options above can be True.")


# ==========================================
# 🧠 模型架构定义
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        if self.qwen3_compatible:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale
        if self.shift is not None:
            norm_x = norm_x + self.shift
        return norm_x.to(input_dtype)


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2:]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, head_dim, num_kv_groups, qk_norm, dtype):
        super().__init__()
        self.d_in = d_in
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_kv_groups
        self.qk_norm = qk_norm

        self.W_query = nn.Linear(d_in, num_heads * head_dim, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(num_heads * head_dim, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, qwen3_compatible=True)
            self.k_norm = RMSNorm(head_dim, qwen3_compatible=True)

    def forward(self, x, mask, cos, sin):
        B, T, C = x.shape
        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.num_kv_groups != self.num_heads:
            group_size = self.num_heads // self.num_kv_groups
            K = K.repeat_interleave(group_size, dim=1)
            V = V.repeat_interleave(group_size, dim=1)

        if self.qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(attn, dim=-1)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.out_proj(output)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin):
        x = x + self.att(self.norm1(x), mask, cos, sin)
        x = x + self.ff(self.norm2(x))
        return x


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        head_dim = cfg["head_dim"] if cfg["head_dim"] is not None else cfg["emb_dim"] // cfg["n_heads"]
        cos, sin = compute_rope_params(head_dim=head_dim, theta_base=cfg["rope_base"],
                                       context_length=cfg["context_length"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits


# ==========================================
# 📦 模型配置
# ==========================================
MODEL_CONFIGS = {
    "0.6B": {
        "vocab_size": 151936,
        "context_length": 40960,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 28,
        "hidden_dim": 3072,
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1000000.0,
        "dtype": torch.bfloat16,
    },
}

QWEN3_CONFIG = MODEL_CONFIGS[CHOOSE_MODEL]

# ==========================================
# 📥 权重加载
# ==========================================
torch.manual_seed(123)
model = Qwen3Model(QWEN3_CONFIG)

model_file = os.path.join(LOCAL_MODEL_PATH, "model.safetensors")
if not os.path.exists(model_file):
    raise FileNotFoundError(f"未找到权重文件: {model_file}")

print(f"正在加载权重: {model_file}")
weights_dict = load_file(model_file)


def load_weights_into_qwen(model, param_config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"形状不匹配: {tensor_name}. 左侧 {left.shape} vs 右侧 {right.shape}")
        with torch.no_grad():
            left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

    assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]
        att = block.att

        assign(att.W_query.weight, params[f"model.layers.{l}.self_attn.q_proj.weight"],
               f"model.layers.{l}.self_attn.q_proj.weight")
        assign(att.W_key.weight, params[f"model.layers.{l}.self_attn.k_proj.weight"],
               f"model.layers.{l}.self_attn.k_proj.weight")
        assign(att.W_value.weight, params[f"model.layers.{l}.self_attn.v_proj.weight"],
               f"model.layers.{l}.self_attn.v_proj.weight")
        assign(att.out_proj.weight, params[f"model.layers.{l}.self_attn.o_proj.weight"],
               f"model.layers.{l}.self_attn.o_proj.weight")

        if hasattr(att, "q_norm") and f"model.layers.{l}.self_attn.q_norm.weight" in params:
            assign(att.q_norm.scale, params[f"model.layers.{l}.self_attn.q_norm.weight"],
                   f"model.layers.{l}.self_attn.q_norm.weight")
        if hasattr(att, "k_norm") and f"model.layers.{l}.self_attn.k_norm.weight" in params:
            assign(att.k_norm.scale, params[f"model.layers.{l}.self_attn.k_norm.weight"],
                   f"model.layers.{l}.self_attn.k_norm.weight")

        assign(block.norm1.scale, params[f"model.layers.{l}.input_layernorm.weight"],
               f"model.layers.{l}.input_layernorm.weight")
        assign(block.norm2.scale, params[f"model.layers.{l}.post_attention_layernorm.weight"],
               f"model.layers.{l}.post_attention_layernorm.weight")

        assign(block.ff.fc1.weight, params[f"model.layers.{l}.mlp.gate_proj.weight"],
               f"model.layers.{l}.mlp.gate_proj.weight")
        assign(block.ff.fc2.weight, params[f"model.layers.{l}.mlp.up_proj.weight"],
               f"model.layers.{l}.mlp.up_proj.weight")
        assign(block.ff.fc3.weight, params[f"model.layers.{l}.mlp.down_proj.weight"],
               f"model.layers.{l}.mlp.down_proj.weight")

    assign(model.final_norm.scale, params["model.norm.weight"], "model.norm.weight")
    if "lm_head.weight" in params:
        assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = model.tok_emb.weight

    print("权重加载成功！")


load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)


# ==========================================
# ✍️ Tokenizer (修复版)
# ==========================================

class Qwen3Tokenizer:
    # 移除硬编码的空字符串，改为动态获取
    def __init__(self, tokenizer_file_path, apply_chat_template=True, add_generation_prompt=False):
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt

        if not os.path.exists(tokenizer_file_path):
            raise FileNotFoundError(f"未找到 tokenizer.json: {tokenizer_file_path}")

        self._tok = Tokenizer.from_file(tokenizer_file_path)

        # Qwen 模型通常使用 <|endoftext|> 作为 pad/eos
        self.pad_token_id = self._tok.token_to_id("<|endoftext|>")
        self.eos_token_id = self._tok.token_to_id("<|endoftext|>")

        # 如果找不到 <|endoftext|>，尝试 <|im_end|>
        if self.pad_token_id is None:
            self.pad_token_id = self._tok.token_to_id("<|im_end|>")
            self.eos_token_id = self._tok.token_to_id("<|im_end|>")

        # 如果还是找不到，回退到 0 (虽然这不安全，但防止报错)
        if self.pad_token_id is None:
            self.pad_token_id = 0
            self.eos_token_id = 0

    def encode(self, text, chat_wrapped=None):
        # 简单处理，如果是 Instruct 模型，包装对话
        if self.apply_chat_template and chat_wrapped is None:
            text = self._wrap_chat(text)

        ids = self._tok.encode(text).ids
        return ids

    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        # Qwen2.5/Qwen3 标准对话模板
        # 注意：这里使用了简单的模板，实际可能更复杂，但足以跑通
        return f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"


# --- 初始化 Tokenizer ---
tokenizer_file_path = os.path.join(LOCAL_MODEL_PATH, "tokenizer.json")
tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tokenizer_file_path,
    apply_chat_template=True,
    add_generation_prompt=True
)

# ==========================================
# ▶️ 运行推理
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)[:, -1]
            next_token = torch.argmax(out, dim=-1, keepdim=True)
            # 检查是否生成结束符
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            yield next_token
            token_ids = torch.cat([token_ids, next_token], dim=1)


# 测试
prompt = "杭州的天气怎么样？"
print(f"User: {prompt}")
input_token_ids = tokenizer.encode(prompt)
input_token_ids_tensor = torch.tensor([input_token_ids], device=device)

print("Assistant: ", end="", flush=True)
for token in generate_text_basic_stream(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=200,
        eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0).tolist()
    text = tokenizer.decode(token_id)
    print(text, end="", flush=True)
print("\n推理完成。")
