import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from configuration_llada import LLaDAConfig, ModelConfig
from modeling_llada import LLaDAModelLM, LLaDAModel


def convert_mistral_to_llada(mistral_model_or_path, llada_config=None):
    """
    Convert a Mistral model to an LLaDA model.
    
    Args:
        mistral_model_or_path: Pretrained Mistral model or model path
        llada_config: LLaDA config, if None will be created from Mistral config
        
    Returns:
        Converted LLaDA model
    """
    # First try to get config
    if isinstance(mistral_model_or_path, str):
        # Load config first
        mistral_config = AutoConfig.from_pretrained(mistral_model_or_path)
        print(f"Loaded Mistral config")
        # Then load model
        mistral_model = AutoModelForCausalLM.from_pretrained(mistral_model_or_path)
    else:
        mistral_model = mistral_model_or_path
        mistral_config = mistral_model.config
    
    # Check the MLP dimensions of the first layer to ensure we get the correct dimensions
    first_layer = mistral_model.model.layers[0]
    gate_proj_shape = first_layer.mlp.gate_proj.weight.shape
    up_proj_shape = first_layer.mlp.up_proj.weight.shape
    down_proj_shape = first_layer.mlp.down_proj.weight.shape
    
    print(f"Mistral MLP dimension check:")
    print(f"  gate_proj: {gate_proj_shape}")
    print(f"  up_proj: {up_proj_shape}")
    print(f"  down_proj: {down_proj_shape}")
    
    # Create LLaDA config
    if llada_config is None:
        # Get correct hidden_size and intermediate_size
        hidden_size = mistral_config.hidden_size
        
        # Analyze Mistral model implementation
        # In Mistral's MistralMLP.forward:
        # down_proj(act_fn(gate_proj(x)) * up_proj(x))
        # In LLaDA's LlamaBlock.forward:
        # x, x_up = ff_proj(x), up_proj(x)
        # x = act(x)  
        # x = x * x_up
        # x = ff_out(x)
        
        # Create a ModelConfig object that directly corresponds to Mistral
        model_config = ModelConfig(
            d_model=hidden_size,
            n_heads=mistral_config.num_attention_heads,
            n_kv_heads=mistral_config.num_key_value_heads,
            n_layers=mistral_config.num_hidden_layers,
            vocab_size=mistral_config.vocab_size,
            max_sequence_length=mistral_config.max_position_embeddings,
            rope=True,  # Mistral uses RoPE
            rope_theta=mistral_config.rope_theta,
            attention_dropout=mistral_config.attention_dropout,
            block_type="llama",  # Use LLaMA block type, compatible with Mistral
            activation_type="silu",  # Mistral uses SwiGLU activation
            layer_norm_type="rms",  # Mistral uses RMSNorm
            rms_norm_eps=mistral_config.rms_norm_eps,
            include_bias=False,  # Mistral usually does not use bias
            scale_logits=False,
            embedding_size=mistral_config.vocab_size,
            # Key: mlp_hidden_size must be set to gate_proj output dim
            # So that LLaMA processing is correct
            mlp_hidden_size=gate_proj_shape[0],
            weight_tying=True,  # Use weight tying, so no separate lm_head needed
            pad_token_id=mistral_config.pad_token_id,
            eos_token_id=mistral_config.eos_token_id,
            mask_token_id=mistral_config.pad_token_id,  # Use pad_token as mask_token
        )
        
        print(f"Mistral model config: hidden_size={hidden_size}, intermediate_size={mistral_config.intermediate_size}")
        print(f"Used mlp_hidden_size={model_config.mlp_hidden_size}")
        
        # Create LLaDAConfig based on ModelConfig
        llada_config = LLaDAConfig(use_cache=False, **model_config.__dict__)
    
    # Create LLaDA model architecture (initialized, not loaded with weights)
    llada_model_base = LLaDAModel(model_config, init_params=True)
    
    # Create LLaDA model
    llada_model = LLaDAModelLM(llada_config, model=llada_model_base)
    
    # Check LLaDA model's MLP dimensions to ensure they match Mistral
    llada_first_layer = llada_model.model.transformer.blocks[0] if hasattr(llada_model.model.transformer, 'blocks') else llada_model.model.transformer.block_groups[0][0]
    print(f"LLaDA MLP dimension check:")
    print(f"  ff_proj: {llada_first_layer.ff_proj.weight.shape}")
    print(f"  up_proj: {llada_first_layer.up_proj.weight.shape}")
    print(f"  ff_out: {llada_first_layer.ff_out.weight.shape}")
    
    # Fix each layer's dimensions to ensure input/output dims are correct
    for i in range(model_config.n_layers):
        llada_layer = llada_model.model.transformer.blocks[i] if hasattr(llada_model.model.transformer, 'blocks') else llada_model.model.transformer.block_groups[0][i]
        
        # Recreate ff_proj layer to match Mistral's gate_proj shape
        if llada_layer.ff_proj.weight.shape != gate_proj_shape:
            new_ff_proj = nn.Linear(
                llada_layer.ff_proj.in_features,
                gate_proj_shape[0],
                bias=model_config.include_bias,
                device=llada_layer.ff_proj.weight.device
            )
            llada_layer.ff_proj = new_ff_proj
        
        # Recreate up_proj layer to match Mistral's up_proj shape
        if llada_layer.up_proj.weight.shape != up_proj_shape:
            new_up_proj = nn.Linear(
                llada_layer.up_proj.in_features,
                up_proj_shape[0],
                bias=model_config.include_bias,
                device=llada_layer.up_proj.weight.device
            )
            llada_layer.up_proj = new_up_proj
        
        # Recreate ff_out layer to match Mistral's down_proj shape
        if llada_layer.ff_out.weight.shape != (down_proj_shape[0], down_proj_shape[1]):
            new_ff_out = nn.Linear(
                down_proj_shape[1],
                down_proj_shape[0],
                bias=model_config.include_bias,
                device=llada_layer.ff_out.weight.device
            )
            llada_layer.ff_out = new_ff_out
    
    # Check LLaDA model's MLP dimensions again to ensure changes took effect
    print(f"LLaDA MLP dimension check after modification:")
    print(f"  ff_proj: {llada_first_layer.ff_proj.weight.shape}")
    print(f"  up_proj: {llada_first_layer.up_proj.weight.shape}")
    print(f"  ff_out: {llada_first_layer.ff_out.weight.shape}")
    
    # Copy Mistral model weights to LLaDA model
    try:
        # 1. Copy word embeddings
        llada_model.model.transformer.wte.weight.data.copy_(mistral_model.model.embed_tokens.weight.data)
        
        # 2. Copy layer normalization
        llada_model.model.transformer.ln_f.weight.data.copy_(mistral_model.model.norm.weight.data)
        
        # 3. If using weight tying, no need to copy output layer (using embed_tokens weights)
        # Otherwise, copy output layer
        if not llada_config.weight_tying and hasattr(llada_model.model.transformer, 'ff_out'):
            llada_model.model.transformer.ff_out.weight.data.copy_(mistral_model.lm_head.weight.data)
        
        # 4. Copy each layer's weights
        for i in range(mistral_config.num_hidden_layers):
            mistral_layer = mistral_model.model.layers[i]
            llada_layer = llada_model.model.transformer.blocks[i] if hasattr(llada_model.model.transformer, 'blocks') else llada_model.model.transformer.block_groups[0][i]
            
            # Copy input layer normalization
            llada_layer.attn_norm.weight.data.copy_(mistral_layer.input_layernorm.weight.data)
            
            # Copy post-attention layer normalization
            llada_layer.ff_norm.weight.data.copy_(mistral_layer.post_attention_layernorm.weight.data)
            
            # Copy attention layer weights
            llada_layer.q_proj.weight.data.copy_(mistral_layer.self_attn.q_proj.weight.data)
            llada_layer.k_proj.weight.data.copy_(mistral_layer.self_attn.k_proj.weight.data)
            llada_layer.v_proj.weight.data.copy_(mistral_layer.self_attn.v_proj.weight.data)
            llada_layer.attn_out.weight.data.copy_(mistral_layer.self_attn.o_proj.weight.data)
            
            # Copy MLP layer weights - ensure shapes match
            print(f"Layer {i} MLP weight shapes:")
            print(f"  gate_proj: {mistral_layer.mlp.gate_proj.weight.shape} -> ff_proj: {llada_layer.ff_proj.weight.shape}")
            print(f"  up_proj: {mistral_layer.mlp.up_proj.weight.shape} -> up_proj: {llada_layer.up_proj.weight.shape}")
            print(f"  down_proj: {mistral_layer.mlp.down_proj.weight.shape} -> ff_out: {llada_layer.ff_out.weight.shape}")
            
            # Copy weights - direct mapping
            # Mistral: gate_proj -> act_fn -> * up_proj -> down_proj
            # LLaDA:   ff_proj -> act (splits input) -> * up_proj -> ff_out
            llada_layer.ff_proj.weight.data.copy_(mistral_layer.mlp.gate_proj.weight.data)
            llada_layer.up_proj.weight.data.copy_(mistral_layer.mlp.up_proj.weight.data)
            llada_layer.ff_out.weight.data.copy_(mistral_layer.mlp.down_proj.weight.data)
        
        print("Successfully copied all weights!")
    except RuntimeError as e:
        print(f"Error converting model weights: {e}")
        print("This may be due to model config mismatch, please ensure the config is read correctly.")
        # Further diagnostics
        print("\nDetailed diagnostics:")
        print(f"1. Mistral hidden_size: {mistral_config.hidden_size}")
        print(f"2. Mistral intermediate_size: {mistral_config.intermediate_size}")
        print(f"3. Mistral gate_proj shape: {gate_proj_shape}")
        print(f"4. Mistral up_proj shape: {up_proj_shape}")
        print(f"5. Mistral down_proj shape: {down_proj_shape}")
        print(f"6. LLaDA d_model: {llada_config.d_model}")
        print(f"7. LLaDA mlp_hidden_size: {llada_config.mlp_hidden_size}")
        print(f"8. LLaDA ff_proj shape: {llada_first_layer.ff_proj.weight.shape}")
        print(f"9. LLaDA up_proj shape: {llada_first_layer.up_proj.weight.shape}")
        print(f"10. LLaDA ff_out shape: {llada_first_layer.ff_out.weight.shape}")
        
        raise
    
    return llada_model


def freeze_mlp_parameters(model):
    """
    Freeze the MLP parameters of the model, allowing only attention layer parameters to be updated
    
    Args:
        model: Model to partially freeze parameters
        
    Returns:
        model
    """
    for name, param in model.named_parameters():
        # Freeze MLP-related parameters
        if any(x in name for x in ["mlp", "ff_proj", "up_proj", "ff_out"]):
            param.requires_grad = False
    
    return model 
