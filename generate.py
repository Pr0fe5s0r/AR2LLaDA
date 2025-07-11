import torch
import numpy as np
import torch.nn.functional as F
import os

from transformers import AutoTokenizer, AutoModel
from modeling_llada import LLaDAModelLM

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
    Get mask_token_id from model config or environment variable
    
    Args:
        model: LLaDA model
        
    Returns:
        int: mask_token_id
    """
    # First try to get from model config
    if hasattr(model, 'config') and hasattr(model.config, 'mask_token_id'):
        return model.config.mask_token_id
    
    # Then try to get from environment variable
    if 'LLADA_MASK_TOKEN_ID' in os.environ:
        return int(os.environ['LLADA_MASK_TOKEN_ID'])
    
    # Finally use default value
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

    model = LLaDAModelLM.from_pretrained('F:/llada/output/checkpoint-50000', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('F:/llada/output/checkpoint-50000', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Detect model type (base model or instruct model)
    is_instruct_model = False
    
    # Determine if it is an instruct model from model config or file name
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        is_instruct_model = "instruct" in model.config.model_type.lower()
    
    if not is_instruct_model and hasattr(model, 'name_or_path'):
        is_instruct_model = "instruct" in model.name_or_path.lower()
    
    print(f"Detected model type: {'Instruct model' if is_instruct_model else 'Base model'}")
    
    # Choose whether to use chat template based on model type
    if is_instruct_model:
        try:
            # Try to use chat template (for instruct models)
            m = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            print("Using chat template to format prompt")
        except (AttributeError, ValueError) as e:
            # If chat template is not available, fallback to manual formatting
            formatted_prompt = f"User: {prompt}\nAssistant: "
            print(f"Chat template not available, using manual formatting: {str(e)}")
    else:
        # Base model uses prompt directly
        formatted_prompt = prompt
        print("Base model: using prompt directly, not applying chat template")
    
    # Encode the formatted prompt
    input_ids = tokenizer(formatted_prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # Automatically get mask_token_id
    mask_token_id = get_mask_token_id(model)
    print(f"Using mask_token_id: {mask_token_id}")

    out = generate(model, input_ids, steps=1024, gen_length=1024, block_length=512, 
                  temperature=0.7, cfg_scale=3, remasking='random', 
                  mask_token_id=mask_token_id)
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
