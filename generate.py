import torch
import numpy as np
import torch.nn.functional as F
import os

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.bfloat16)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def get_mask_token_id(model):
    """
    從模型配置或環境變量獲取mask_token_id
    
    Args:
        model: LLaDA模型
        
    Returns:
        int: mask_token_id
    """
    # 首先嘗試從模型配置獲取
    if hasattr(model, 'config') and hasattr(model.config, 'mask_token_id'):
        return model.config.mask_token_id
    
    # 然後嘗試從環境變量獲取
    if 'LLADA_MASK_TOKEN_ID' in os.environ:
        return int(os.environ['LLADA_MASK_TOKEN_ID'])
    
    # 最後使用默認值
    return 126336


@ torch.no_grad()
def generate(model, prompt, steps=256, gen_length=256, block_length=256, temperature=0.7,
             cfg_scale=3, remasking='random', mask_token_id=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_token_id: The token id of [MASK]. If None, will be determined automatically.
    '''
    if mask_token_id is None:
        mask_token_id = get_mask_token_id(model)
        
    x = torch.full((1, prompt.shape[1] + gen_length), mask_token_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_token_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_token_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_token_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_token_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('F:/llada/output/checkpoint-50000', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('F:/llada/output/checkpoint-50000', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # 檢測模型類型（基礎模型或指導模型）
    is_instruct_model = False
    
    # 從模型配置或文件名判斷是否為指導模型
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        is_instruct_model = "instruct" in model.config.model_type.lower()
    
    if not is_instruct_model and hasattr(model, 'name_or_path'):
        is_instruct_model = "instruct" in model.name_or_path.lower()
    
    print(f"檢測到模型類型: {'指導模型' if is_instruct_model else '基礎模型'}")
    
    # 根據模型類型選擇是否使用聊天模板
    if is_instruct_model:
        try:
            # 嘗試使用聊天模板（對於指導模型）
            m = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            print("使用聊天模板格式化提示")
        except (AttributeError, ValueError) as e:
            # 如果聊天模板不可用，退回到手動格式化
            formatted_prompt = f"User: {prompt}\nAssistant: "
            print(f"聊天模板不可用，使用手動格式化: {str(e)}")
    else:
        # 基礎模型直接使用提示
        formatted_prompt = prompt
        print("基礎模型：直接使用提示，不應用聊天模板")
    
    # 編碼格式化後的提示
    input_ids = tokenizer(formatted_prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # 自動獲取mask_token_id
    mask_token_id = get_mask_token_id(model)
    print(f"使用mask_token_id: {mask_token_id}")

    out = generate(model, input_ids, steps=1024, gen_length=1024, block_length=512, 
                  temperature=0.7, cfg_scale=3, remasking='random', 
                  mask_token_id=mask_token_id)
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
