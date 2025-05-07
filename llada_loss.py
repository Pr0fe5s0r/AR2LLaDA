import torch
import torch.nn.functional as F
import os

# 从环境变量获取MASK_TOKEN_ID，如果不存在则使用默认值
MASK_TOKEN_ID = int(os.environ.get("LLADA_MASK_TOKEN_ID", "126336"))

def forward_process(input_ids, mask_token_id=None, eps=1e-3):
    """
    对输入的tokens进行掩码处理，返回掩码后的batch、掩码的位置和掩码概率。
    
    Args:
        input_ids (torch.Tensor): 输入的token ids，形状为 (batch_size, seq_len)
        mask_token_id (int, optional): 用于掩码的token id，如果为None则使用全局MASK_TOKEN_ID
        eps (float, optional): 掩码概率的最小值。默认为1e-3
        
    Returns:
        tuple:
            - noisy_batch (torch.Tensor): 掩码后的token ids
            - masked_indices (torch.Tensor): 掩码的位置，布尔型张量
            - p_mask (torch.Tensor): 掩码概率
    """
    if mask_token_id is None:
        mask_token_id = MASK_TOKEN_ID
        
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, mask_token_id, input_ids)
    return noisy_batch, masked_indices, p_mask

def compute_pretrain_loss(model, batch, mask_token_id=None):
    """
    计算预训练损失
    
    Args:
        model: LLaDA模型
        batch (dict): 包含input_ids的批处理数据
        mask_token_id (int, optional): 掩码token的id，如果为None则使用全局MASK_TOKEN_ID
        
    Returns:
        torch.Tensor: 预训练损失值
    """
    if mask_token_id is None:
        mask_token_id = MASK_TOKEN_ID
        
    input_ids = batch["input_ids"]
    
    # 有1%的概率随机选择长度
    if torch.rand(1).item() < 0.01:
        random_length = torch.randint(1, input_ids.shape[1] + 1, (1,)).item()
        input_ids = input_ids[:, :random_length]

    # 前向处理：添加噪声（掩码）
    noisy_batch, masked_indices, p_mask = forward_process(input_ids, mask_token_id)
    
    # 通过模型获取logits
    outputs = model(input_ids=noisy_batch)
    logits = outputs.logits
    
    # 计算损失 - 将view替换为reshape
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1))[masked_indices.reshape(-1)], 
        input_ids.reshape(-1)[masked_indices.reshape(-1)], 
        reduction='none'
    ) / p_mask.reshape(-1)[masked_indices.reshape(-1)]
    
    # 归一化损失
    loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
    
    return loss

def compute_sft_loss(model, batch, mask_token_id=None):
    """
    计算SFT（监督微调）损失
    
    Args:
        model: LLaDA模型
        batch (dict): 包含input_ids和prompt_lengths的批处理数据
        mask_token_id (int, optional): 掩码token的id，如果为None则使用全局MASK_TOKEN_ID
        
    Returns:
        torch.Tensor: SFT损失值
    """
    if mask_token_id is None:
        mask_token_id = MASK_TOKEN_ID
        
    input_ids, prompt_lengths = batch["input_ids"], batch["prompt_lengths"]
    
    # 前向处理：添加噪声（掩码）
    noisy_batch, _, p_mask = forward_process(input_ids, mask_token_id)
    
    # 不对prompt部分添加噪声
    batch_size, seq_len = noisy_batch.shape
    token_positions = torch.arange(seq_len, device=noisy_batch.device).expand(batch_size, seq_len)
    prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    
    # 计算答案长度（包括填充的<EOS>标记）
    prompt_mask = prompt_mask.to(torch.int64)
    answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
    answer_lengths = answer_lengths.repeat(1, seq_len)
    
    # 获取掩码位置
    masked_indices = (noisy_batch == mask_token_id)
    
    # 通过模型获取logits
    outputs = model(input_ids=noisy_batch)
    logits = outputs.logits
    
    # 计算损失 - 将view替换为reshape
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1))[masked_indices.reshape(-1)], 
        input_ids.reshape(-1)[masked_indices.reshape(-1)], 
        reduction='none'
    ) / p_mask.reshape(-1)[masked_indices.reshape(-1)]
    
    # 归一化损失（按照答案长度）
    ce_loss = torch.sum(token_loss / answer_lengths.reshape(-1)[masked_indices.reshape(-1)]) / input_ids.shape[0]
    
    return ce_loss

def llada_loss(model, batch, sft_mode=False, mask_token_id=None):
    """
    根據LLaDA公式計算loss。
    
    Args:
        model: LLaDA模型
        batch (dict): 包含input_ids的批處理數據，若為SFT還需prompt_lengths
        sft_mode (bool): 是否為SFT訓練（不對prompt部分加噪聲）
        mask_token_id (int, optional): 掩码token的id，如果为None则使用全局MASK_TOKEN_ID
        
    Returns:
        torch.Tensor: 損失值
    """
    if mask_token_id is None:
        mask_token_id = MASK_TOKEN_ID
        
    input_ids = batch["input_ids"]
    if sft_mode:
        prompt_lengths = batch["prompt_lengths"]
        noisy_batch, _, p_mask = forward_process(input_ids, mask_token_id)
        token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
        prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
        noisy_batch[prompt_mask] = input_ids[prompt_mask]
        prompt_mask = prompt_mask.to(torch.int64)
        answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])
        masked_indices = (noisy_batch == mask_token_id)
        logits = model(input_ids=noisy_batch).logits
        token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
        ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
        return ce_loss
    else:
        # 預訓練loss
        if torch.rand(1) < 0.01:
            random_length = torch.randint(1, input_ids.shape[1] + 1, (1,))
            input_ids = input_ids[:, :random_length]
        noisy_batch, masked_indices, p_mask = forward_process(input_ids, mask_token_id)
        logits = model(input_ids=noisy_batch).logits
        token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
        loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
        return loss