"""
Transformer Language Model Implementation

本模块实现了完整的 Transformer 语言模型，包括：
- 基础层：Linear, Embedding, RMSNorm
- 激活函数：SiLU
- 位置编码：RoPE (Rotary Position Embedding)
- 注意力机制：Scaled Dot-Product Attention, Multi-Head Attention
- 前馈网络：SwiGLU
- 完整模型：TransformerBlock, TransformerLM

Author: PSX
Date: 2025-11-01
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import math


# ============================================================
# 第一部分：基础层 - Linear
# ============================================================
class Linear(nn.Module):

    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        # 初始化权重矩阵 y = x @ W.T + b
        self.weight = nn.Parameter(torch.empty(d_out, d_in))

        # Kaiming 初始化
        bound = math.sqrt(6.0 / d_in)
        nn.init.uniform_(self.weight, -bound, bound)

        # 创建偏置
        if bias:
            self.bias = nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x: Tensor) -> Tensor:
        out = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            out = out + self.bias
        
        return out
    

# ============================================================
# 第二部分：Embedding 层
# ============================================================
class Embedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
    
        # 创建嵌入矩阵
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        # pytorch默认std=1.0,这里采用bert工程实践经验std=0.02
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, tokens_id: Tensor) -> Tensor:
        """
        前向传播：查表操作
        
        参数:
            token_ids: token ID 张量 (batch_size, seq_len)
        
        返回:
            嵌入向量 (batch_size, seq_len, d_model)
        """
        return self.weight[tokens_id]



# ============================================================
# 第三部分：RMSNorm 层
# ============================================================
class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量 (..., d_model)
        
        返回:
            归一化后的张量 (..., d_model)
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        x_normed = x / rms
        x_normed = x_normed * self.weight
        return x_normed


# 激活函数
def silu(x: Tensor) -> Tensor:
    """
    SiLU 激活函数
    """
    exp_minus_x = torch.exp(-x)
    sigmoid_x = 1.0 / (1.0 + exp_minus_x)
    return x * sigmoid_x



# ============================================================
# 第五部分：RoPE 位置编码
# ============================================================
class RoPE(nn.Module):
    """
    旋转位置编码 (Rotary Position Embedding)
    
    RoPE 通过旋转操作将位置信息编码到 Q 和 K 中。
    相比绝对位置编码，RoPE 具有更好的外推能力。
    
    核心思想:
        将 d_head 维向量看作 d_head/2 个复数
        对每个复数应用旋转变换: z' = z * e^(i*m*θ)
        其中 m 是位置，θ 是频率
    
    参数:
        d_head: 每个注意力头的维度 (必须是偶数)
        max_seq_len: 最大序列长度
        theta: 频率基数，默认 10000.0
    
    形状:
        输入: (batch, seq_len, num_heads, d_head)
        输出: (batch, seq_len, num_heads, d_head)
    
    参考:
        RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://arxiv.org/abs/2104.09864
    
    示例:
        >>> rope = RoPE(d_head=64, max_seq_len=2048)
        >>> x = torch.randn(2, 10, 8, 64)  # (batch, seq, heads, d_head)
        >>> pos = torch.arange(10).unsqueeze(0).expand(2, -1)  # (batch, seq)
        >>> x_rotated = rope(x, pos)
    """
    def __init__(self, d_head: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert d_head % 2 == 0, "d_head must be even for RoPE"

        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 预计算频率和旋转角度
        inv_freq = self._compute_inv_frequencies()
        self.register_buffer("inv_freq", inv_freq)  # 注册为模型参数，避免梯度计算

        cos, sin = self._precompute_cos_sin(inv_freq, max_seq_len)
        self.register_buffer("cos", cos)  # 注册为模型参数，避免梯度计算
        self.register_buffer("sin", sin)  # 注册为模型参数，避免梯度计算
    
    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        应用 RoPE
        
        参数:
            x: 输入，支持两种形状：
               - 3D: (..., seq_len, d_model)
               - 4D: (batch, seq_len, num_heads, d_head)
            token_positions: (..., seq_len)
        
        返回:
            旋转后的张量，形状与输入相同
        """

        # 首先判断输入的x是什么形状
        needs_unsqueeze = (x.ndim == 3)
        if needs_unsqueeze:
            x = x.unsqueeze(-2)
        
        # 获取预处理的cos/sin值
        cos, sin = self._get_cos_sin_for_positions(token_positions)
        # 拆分实部虚部
        x_real, x_imag = self._reshape_for_rotation(x)
        # 应用旋转
        x_rotated = self._apply_rotation(x_real, x_imag, cos, sin)

        if needs_unsqueeze:
            x_rotated = x_rotated.squeeze(-2)
        
        return x_rotated

    def _compute_inv_frequencies(self) -> Tensor:
        """
        计算逆频率向量
        
        公式: inv_freq[i] = 1 / (theta^(2i/d_head))
        其中 i = 0, 1, ..., d_head/2 - 1
        
        返回:
            inv_freq: 形状 (d_head // 2,)
        
        示例:
            >>> rope = RoPE(d_head=4, max_seq_len=10, theta=10000.0)
            >>> rope.inv_freq
            tensor([1.0000, 0.0100])  # [1/10000^0, 1/10000^(2/4)]
        """
        indices = torch.arange(0, self.d_head, 2, dtype=torch.float32)
        exponents = indices / self.d_head
        inv_freq = 1.0 / (self.theta ** exponents)
        return inv_freq
    
    def _precompute_cos_sin(self, inv_freq: Tensor, max_seq_len: int) -> Tuple[Tensor, Tensor]:
        """
        预计算所有位置的 cos 和 sin 值
        
        参数:
            inv_freq: 逆频率向量 (d_head // 2,)
            max_seq_len: 最大序列长度
        
        返回:
            cos: 余弦值 (max_seq_len, d_head // 2)
            sin: 正弦值 (max_seq_len, d_head // 2)
        
        计算过程:
            1. 生成位置索引: [0, 1, 2, ..., max_seq_len-1]
            2. 计算角度矩阵: positions ⊗ inv_freq (外积)
            3. 计算 cos 和 sin
        """
        # 生成位置索引
        positions = torch.arange(max_seq_len, dtype=torch.float32)  # (max_seq_len,)
        # 计算角度矩阵
        freqs = torch.outer(positions, inv_freq)

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        return cos, sin
    
    def _reshape_for_rotation(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        将输入重排为复数形式 (实部, 虚部)
        
        参数:
            x: 输入张量 (..., d_head)
        
        返回:
            x_real: 实部 (..., d_head // 2)
            x_imag: 虚部 (..., d_head // 2)
        
        转换示例:
            输入: [x0, x1, x2, x3, x4, x5]
            实部: [x0, x2, x4]  # 偶数索引
            虚部: [x1, x3, x5]  # 奇数索引
        """
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x_real = x_reshaped[..., 0]  # 实部
        x_imag = x_reshaped[..., 1]  # 虚部
        return x_real, x_imag
    
    def _apply_rotation(self, 
                        x_real: Tensor,
                        x_imag: Tensor,
                        cos: Tensor,
                        sin: Tensor) -> Tensor:
        """
        应用复数旋转变换
        
        复数乘法公式:
            (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
        
        参数:
            x_real: 实部 (..., d_head // 2)
            x_imag: 虚部 (..., d_head // 2)
            cos: 余弦值 (..., d_head // 2)
            sin: 正弦值 (..., d_head // 2)
        
        返回:
            旋转后的张量 (..., d_head)
        """
        cos = cos.to(x_real.dtype)
        sin = sin.to(x_real.dtype)
        x_rotated_real = x_real * cos - x_imag * sin
        x_rotated_imag = x_real * sin + x_imag * cos

        x_rotated = torch.stack([x_rotated_real, x_rotated_imag], dim=-1)
        x_rotated = x_rotated.reshape(*x_real.shape[:-1], -1)  # 展平
        return x_rotated
    
    def _get_cos_sin_for_positions(self, token_positions: Tensor) -> Tuple[Tensor, Tensor]:
        """
        根据位置索引获取对应的 cos 和 sin 值
        
        参数:
            token_positions: 位置索引 (batch, seq_len)
        
        返回:
            cos: 对应位置的余弦值 (batch, seq_len, 1, d_head // 2)
            sin: 对应位置的正弦值 (batch, seq_len, 1, d_head // 2)
        """
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

        return cos, sin
    

# ============================================================
# 第六部分：缩放点积注意力
# ============================================================
def scaled_dot_product_attention(
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Optional[Tensor] = None,
) -> Tensor:
    """
    缩放点积注意力
    
    公式:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    步骤:
        1. 计算注意力分数: scores = QK^T / sqrt(d_k)
        2. 应用掩码 (可选)
        3. Softmax 归一化
        4. 加权求和: output = attention_weights @ V
    
    参数:
        Q: Query 张量 (..., seq_len_q, d_k)
        K: Key 张量 (..., seq_len_k, d_k)
        V: Value 张量 (..., seq_len_k, d_v)
        mask: 注意力掩码 (..., seq_len_q, seq_len_k)
              mask=True 的位置会被 mask 掉 (设为 -inf)
    
    返回:
        注意力输出 (..., seq_len_q, d_v)
    
    示例:
        >>> Q = torch.randn(2, 4, 10, 64)  # (batch, heads, seq_q, d_k)
        >>> K = torch.randn(2, 4, 10, 64)  # (batch, heads, seq_k, d_k)
        >>> V = torch.randn(2, 4, 10, 64)  # (batch, heads, seq_k, d_v)
        >>> output = scaled_dot_product_attention(Q, K, V)
        >>> print(output.shape)  # (2, 4, 10, 64)
    """

    # 获取d_k, Q,K向量的维度
    d_k = Q.size(-1)
    # 计算注意力分数 scores = QK^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # 应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask != True, float("-inf"))
    # Softmax 归一化
    attention_weights = torch.softmax(scores, dim=-1)
    # 加权求和
    output = torch.matmul(attention_weights, V)

    return output


        
# ============================================================
# 第七部分：多头自注意力
# ============================================================
class MultiHeadAttention(nn.Module):
    """
    多头自注意力层
    
    多头注意力允许模型同时关注不同位置的不同表示子空间。
    
    步骤:
        1. 线性投影: Q, K, V = xW_q, xW_k, xW_v
        2. 分头: 将 d_model 拆分为 num_heads 个 d_head
        3. 并行计算多个注意力头
        4. 合并多头输出
        5. 输出投影: output = concat(heads)W_o
    
    参数:
        d_model: 模型维度
        num_heads: 注意力头数量
        use_rope: 是否使用 RoPE
        max_seq_len: 最大序列长度 (仅当 use_rope=True 时需要)
        theta: RoPE 频率基数
    
    形状:
        输入: (batch, seq_len, d_model)
        输出: (batch, seq_len, d_model)
    """
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 use_rope: bool = False,
                 max_seq_len: int = 2048,
                 theta: float = 10000.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.use_rope = use_rope
        
        # 高效创建QKV投影层， 通过大线性层，num_heads: 注意力头数量通过reshape切割成各专家
        self.q_proj = Linear(d_model, d_model, bias=False)
        self.k_proj = Linear(d_model, d_model, bias=False)
        self.v_proj = Linear(d_model, d_model, bias=False)

        # 输出投影层
        self.out_proj = Linear(d_model, d_model, bias=False)

        # RoPE 位置编码
        if self.use_rope:
            self.rope = RoPE(d_head=self.d_head, max_seq_len=max_seq_len, theta=theta)
        else:
            self.rope = None
        
    # 前向传播
    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入 (batch, seq_len, d_model)
            token_positions: 位置索引 (batch, seq_len)，仅当 use_rope=True 时需要
        
        返回:
            输出 (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        # 线性投影
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 分头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_head)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_head)

        # 应用 RoPE 位置编码
        if self.use_rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        # 转置
        Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len, d_head)
        K = K.transpose(1, 2)  # (batch, num_heads, seq_len, d_head)
        V = V.transpose(1, 2)  # (batch, num_heads, seq_len, d_head)

        # 创建因果编码
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=0
        )

        # 计算注意力
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # 转置还原
        attn_output = attn_output.transpose(1, 2)  # (batch, seq_len, num_heads, d_head)
 
        # 合并多头
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # 输出投影  # (batch, seq_len, d_model)
        output = self.out_proj(attn_output)

        return output



# ==============================
# 第八部分：前馈网络
# ==============================
class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络
    
    SwiGLU 是一种改进的前馈网络，使用门控机制和 SiLU 激活函数。
    相比传统的 FFN，SwiGLU 有更好的表达能力。
    
    公式:
        SwiGLU(x) = (W1·x ⊙ SiLU(W3·x)) · W2
    
    参数:
        d_model: 输入/输出维度
        d_ff: 中间层维度（通常是 d_model 的 4 倍）
    
    形状:
        输入: (..., d_model)
        输出: (..., d_model)
    
    参考:
        GLU Variants Improve Transformer
        https://arxiv.org/abs/2002.05202
    
    示例:
        >>> swiglu = SwiGLU(d_model=512, d_ff=2048)
        >>> x = torch.randn(2, 10, 512)
        >>> output = swiglu(x)
        >>> print(output.shape)  # (2, 10, 512)
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # 上投影矩阵
        self.w1 = Linear(d_model, d_ff, bias=False)
        # 下投影矩阵
        self.w2 = Linear(d_ff, d_model, bias=False)
        # 门控上投影矩阵
        self.w3 = Linear(d_model, d_ff, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:

        # 上投影
        x1 = self.w1(x)  # (..., d_ff)

        # 门控
        x3 = self.w3(x)  # (..., d_ff)
        gate = x3 * silu(x1)  # (..., d_ff)

        # 下投影
        output = self.w2(gate)  # (..., d_model)

        return output


# ============================================================
# 第六部分：缩放点积注意力
# ============================================================
def scaled_dot_product_attention(
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Optional[Tensor] = None,
) -> Tensor:
    """
    缩放点积注意力
    
    公式:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    步骤:
        1. 计算注意力分数: scores = QK^T / sqrt(d_k)
        2. 应用掩码 (可选)
        3. Softmax 归一化
        4. 加权求和: output = attention_weights @ V
    
    参数:
        Q: Query 张量 (..., seq_len_q, d_k)
        K: Key 张量 (..., seq_len_k, d_k)
        V: Value 张量 (..., seq_len_k, d_v)
        mask: 注意力掩码 (..., seq_len_q, seq_len_k)
              mask=True 的位置会被 mask 掉 (设为 -inf)
    
    返回:
        注意力输出 (..., seq_len_q, d_v)
    
    示例:
        >>> Q = torch.randn(2, 4, 10, 64)  # (batch, heads, seq_q, d_k)
        >>> K = torch.randn(2, 4, 10, 64)  # (batch, heads, seq_k, d_k)
        >>> V = torch.randn(2, 4, 10, 64)  # (batch, heads, seq_k, d_v)
        >>> output = scaled_dot_product_attention(Q, K, V)
        >>> print(output.shape)  # (2, 4, 10, 64)
    """

    # 获取d_k, Q,K向量的维度
    d_k = Q.size(-1)
    # 计算注意力分数 scores = QK^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # 应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask != True, float("-inf"))
    # Softmax 归一化
    attention_weights = torch.softmax(scores, dim=-1)
    # 加权求和
    output = torch.matmul(attention_weights, V)

    return output


        
# ============================================================
# 第七部分：多头自注意力
# ============================================================
class MultiHeadAttention(nn.Module):
    """
    多头自注意力层
    
    多头注意力允许模型同时关注不同位置的不同表示子空间。
    
    步骤:
        1. 线性投影: Q, K, V = xW_q, xW_k, xW_v
        2. 分头: 将 d_model 拆分为 num_heads 个 d_head
        3. 并行计算多个注意力头
        4. 合并多头输出
        5. 输出投影: output = concat(heads)W_o
    
    参数:
        d_model: 模型维度
        num_heads: 注意力头数量
        use_rope: 是否使用 RoPE
        max_seq_len: 最大序列长度 (仅当 use_rope=True 时需要)
        theta: RoPE 频率基数
    
    形状:
        输入: (batch, seq_len, d_model)
        输出: (batch, seq_len, d_model)
    """
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 use_rope: bool = False,
                 max_seq_len: int = 2048,
                 theta: float = 10000.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.use_rope = use_rope
        
        # 高效创建QKV投影层， 通过大线性层，num_heads: 注意力头数量通过reshape切割成各专家
        self.q_proj = Linear(d_model, d_model, bias=False)
        self.k_proj = Linear(d_model, d_model, bias=False)
        self.v_proj = Linear(d_model, d_model, bias=False)

        # 输出投影层
        self.out_proj = Linear(d_model, d_model, bias=False)

        # RoPE 位置编码
        if self.use_rope:
            self.rope = RoPE(d_head=self.d_head, max_seq_len=max_seq_len, theta=theta)
        else:
            self.rope = None
        
    # 前向传播
    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入 (batch, seq_len, d_model)
            token_positions: 位置索引 (batch, seq_len)，仅当 use_rope=True 时需要
        
        返回:
            输出 (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        # 线性投影
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 分头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_head)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_head)

        # 应用 RoPE 位置编码
        if self.use_rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        # 转置
        Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len, d_head)
        K = K.transpose(1, 2)  # (batch, num_heads, seq_len, d_head)
        V = V.transpose(1, 2)  # (batch, num_heads, seq_len, d_head)

        # 创建因果编码
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=0
        )

        # 计算注意力
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # 转置还原
        attn_output = attn_output.transpose(1, 2)  # (batch, seq_len, num_heads, d_head)
 
        # 合并多头
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # 输出投影  # (batch, seq_len, d_model)
        output = self.out_proj(attn_output)

        return output



# ===================================
# 实现transformerblock块，整合上面全部功能
# ===================================
class TransformerBlock(nn.Module):
    """
    Transformer 块
    
    一个标准的 Transformer 块包含：
    1. Pre-Norm + Multi-Head Attention + 残差
    2. Pre-Norm + Feed-Forward Network + 残差
    
    架构 (Pre-Norm):
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    
    参数:
        d_model: 模型维度
        num_heads: 注意力头数量
        d_ff: FFN 中间层维度
        use_rope: 是否使用 RoPE
        max_seq_len: 最大序列长度
        theta: RoPE 频率基数
    
    形状:
        输入: (batch, seq_len, d_model)
        输出: (batch, seq_len, d_model)
    
    示例:
        >>> block = TransformerBlock(d_model=512, num_heads=8, d_ff=2048)
        >>> x = torch.randn(2, 10, 512)
        >>> output = block(x)
        >>> print(output.shape)  # (2, 10, 512)
    """
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 use_rope: bool = False,
                 max_seq_len: int = 2048,
                 theta: float = 10000.0
                 ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 注意力归一化层
        self.attn_norm = RMSNorm(d_model)
        # 多头注意力
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            theta=theta
        )
        # 前馈归一化层
        self.ffn_norm = RMSNorm(d_model)
        # 前馈网络
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
    
    def forward(self,
                 x: Tensor, 
                 token_positions: Optional[Tensor] = None
                ) -> Tensor:
        """
    前向传播
    
    架构:
        x ──→ RMSNorm ──→ MultiHeadAttention ──→ + ──→ x
        └──────────────────────────────────────────┘
        
        x ──→ RMSNorm ──→ SwiGLU ──→ + ──→ output
        └────────────────────────────┘
        """

        #==================
        # 注意力机制
        #==================
        # 1. pre_norm
        normed = self.attn_norm(x)
        # 2. 多头注意力
        attn_output = self.attn(normed, token_positions)
        # 3. 残差
        x = x + attn_output

        # ===================
        # 前馈网络
        # ===================
        # 1. pre_norm
        ffn_normed = self.ffn_norm(x)
        # 2. 前馈网络
        ffn_output = self.ffn(ffn_normed)
        # 3. 残差
        output = x + ffn_output

        return output



class TransformerLM(nn.Module):
    """
    完整的 Transformer 语言模型
    
    架构:
        token_ids 
        → Embedding 
        → [TransformerBlock] × num_layers 
        → RMSNorm 
        → Linear (LM Head) 
        → logits
    
    参数:
        vocab_size: 词表大小
        context_length: 最大上下文长度
        d_model: 模型维度
        num_layers: Transformer 层数
        num_heads: 注意力头数
        d_ff: FFN 中间层维度
        rope_theta: RoPE 频率基数
        eps: RMSNorm 的 epsilon
    
    形状:
        输入: (batch, seq_len) - token IDs
        输出: (batch, seq_len, vocab_size) - 下一个 token 的 logits
    
    示例:
        >>> model = TransformerLM(
        ...     vocab_size=50000,
        ...     context_length=2048,
        ...     d_model=512,
        ...     num_layers=6,
        ...     num_heads=8,
        ...     d_ff=2048
        ... )
        >>> token_ids = torch.randint(0, 50000, (4, 100))  # (batch=4, seq_len=100)
        >>> logits = model(token_ids)
        >>> print(logits.shape)  # (4, 100, 50000)
    """

    def __init__(
            self, 
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float = 10000.0,
            eps: float = 1e-5):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.eps = eps

        # 词嵌入层
        self.embedding = Embedding(vocab_size, d_model)
        # transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                use_rope=True,
                max_seq_len=context_length,
                theta=rope_theta
            ) for _ in range(num_layers)
        ])
        # 归一化层
        self.norm = RMSNorm(d_model, eps=eps)
        # 输出层
        self.output = Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens_id: Tensor) -> Tensor:
        """
        前向传播

        架构:
            token_ids
            → Embedding
            → [TransformerBlock] × num_layers
            → RMSNorm
            → Linear (LM Head)
            → logits
        """
        # 1. 词嵌入
        x = self.embedding(tokens_id)


        # 制作token_positions
        batch_size, seq_len = tokens_id.shape
        token_positions = torch.arange(seq_len, device=tokens_id.device)  # (seq_len,)
        token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)  # (batch, seq_len)

        # 2. transformer层
        for layer in self.layers:
            x = layer(x, token_positions)

        # 3. 归一化
        x = self.norm(x)

        # 4. 输出层
        logits = self.output(x)  # (batch, seq_len, vocab_size)

        return logits
