import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from configuration_llada import LLaDAConfig, ModelConfig
from modeling_llada import LLaDAModelLM, LLaDAModel


def convert_mistral_to_llada(mistral_model_or_path, llada_config=None):
    """
    将Mistral模型转换为LLaDA模型。
    
    Args:
        mistral_model_or_path: 预训练的Mistral模型或模型路径
        llada_config: LLaDA配置，如果为None则根据Mistral配置创建
        
    Returns:
        转换后的LLaDA模型
    """
    # 首先尝试获取配置
    if isinstance(mistral_model_or_path, str):
        # 先加载配置
        mistral_config = AutoConfig.from_pretrained(mistral_model_or_path)
        print(f"已加载Mistral配置")
        # 然后加载模型
        mistral_model = AutoModelForCausalLM.from_pretrained(mistral_model_or_path)
    else:
        mistral_model = mistral_model_or_path
        mistral_config = mistral_model.config
    
    # 检查一层的MLP维度，确保我们获取正确的维度
    first_layer = mistral_model.model.layers[0]
    gate_proj_shape = first_layer.mlp.gate_proj.weight.shape
    up_proj_shape = first_layer.mlp.up_proj.weight.shape
    down_proj_shape = first_layer.mlp.down_proj.weight.shape
    
    print(f"Mistral MLP维度检查:")
    print(f"  gate_proj: {gate_proj_shape}")
    print(f"  up_proj: {up_proj_shape}")
    print(f"  down_proj: {down_proj_shape}")
    
    # 创建LLaDA配置
    if llada_config is None:
        # 获取正确的hidden_size和intermediate_size
        hidden_size = mistral_config.hidden_size
        
        # 分析Mistral模型实现
        # 在Mistral的MistralMLP.forward中:
        # down_proj(act_fn(gate_proj(x)) * up_proj(x))
        # 而LLaDA的LlamaBlock.forward中:
        # x, x_up = ff_proj(x), up_proj(x)
        # x = act(x)  
        # x = x * x_up
        # x = ff_out(x)
        
        # 创建一个直接对应Mistral的ModelConfig对象
        model_config = ModelConfig(
            d_model=hidden_size,
            n_heads=mistral_config.num_attention_heads,
            n_kv_heads=mistral_config.num_key_value_heads,
            n_layers=mistral_config.num_hidden_layers,
            vocab_size=mistral_config.vocab_size,
            max_sequence_length=mistral_config.max_position_embeddings,
            rope=True,  # Mistral使用RoPE
            rope_theta=mistral_config.rope_theta,
            attention_dropout=mistral_config.attention_dropout,
            block_type="llama",  # 使用LLaMA块类型，与Mistral兼容
            activation_type="silu",  # Mistral使用SwiGLU激活函数
            layer_norm_type="rms",  # Mistral使用RMSNorm
            rms_norm_eps=mistral_config.rms_norm_eps,
            include_bias=False,  # Mistral通常不使用偏置
            scale_logits=False,
            embedding_size=mistral_config.vocab_size,
            # 关键是mlp_hidden_size必须设置为gate_proj的输出维度
            # 这样在LLaMA的处理方式中才会正确
            mlp_hidden_size=gate_proj_shape[0],
            weight_tying=True,  # 使用权重绑定，这样就不需要单独的lm_head
            pad_token_id=mistral_config.pad_token_id,
            eos_token_id=mistral_config.eos_token_id,
            mask_token_id=mistral_config.pad_token_id,  # 使用pad_token作为mask_token
        )
        
        print(f"Mistral模型配置：hidden_size={hidden_size}, intermediate_size={mistral_config.intermediate_size}")
        print(f"使用的mlp_hidden_size={model_config.mlp_hidden_size}")
        
        # 创建LLaDAConfig，基于ModelConfig
        llada_config = LLaDAConfig(use_cache=False, **model_config.__dict__)
    
    # 创建LLaDA模型架构（初始化，不加载权重）
    llada_model_base = LLaDAModel(model_config, init_params=True)
    
    # 创建LLaDA模型
    llada_model = LLaDAModelLM(llada_config, model=llada_model_base)
    
    # 检查LLaDA模型的MLP维度，确保与Mistral匹配
    llada_first_layer = llada_model.model.transformer.blocks[0] if hasattr(llada_model.model.transformer, 'blocks') else llada_model.model.transformer.block_groups[0][0]
    print(f"LLaDA MLP维度检查:")
    print(f"  ff_proj: {llada_first_layer.ff_proj.weight.shape}")
    print(f"  up_proj: {llada_first_layer.up_proj.weight.shape}")
    print(f"  ff_out: {llada_first_layer.ff_out.weight.shape}")
    
    # 修复各层的维度，确保输入输出维度正确
    for i in range(model_config.n_layers):
        llada_layer = llada_model.model.transformer.blocks[i] if hasattr(llada_model.model.transformer, 'blocks') else llada_model.model.transformer.block_groups[0][i]
        
        # 重新创建ff_proj层使其形状与Mistral的gate_proj匹配
        if llada_layer.ff_proj.weight.shape != gate_proj_shape:
            new_ff_proj = nn.Linear(
                llada_layer.ff_proj.in_features,
                gate_proj_shape[0],
                bias=model_config.include_bias,
                device=llada_layer.ff_proj.weight.device
            )
            llada_layer.ff_proj = new_ff_proj
        
        # 重新创建up_proj层使其形状与Mistral的up_proj匹配
        if llada_layer.up_proj.weight.shape != up_proj_shape:
            new_up_proj = nn.Linear(
                llada_layer.up_proj.in_features,
                up_proj_shape[0],
                bias=model_config.include_bias,
                device=llada_layer.up_proj.weight.device
            )
            llada_layer.up_proj = new_up_proj
        
        # 重新创建ff_out层使其形状与Mistral的down_proj匹配
        if llada_layer.ff_out.weight.shape != (down_proj_shape[0], down_proj_shape[1]):
            new_ff_out = nn.Linear(
                down_proj_shape[1],
                down_proj_shape[0],
                bias=model_config.include_bias,
                device=llada_layer.ff_out.weight.device
            )
            llada_layer.ff_out = new_ff_out
    
    # 再次检查LLaDA模型的MLP维度，确保修改生效
    print(f"修改后的LLaDA MLP维度检查:")
    print(f"  ff_proj: {llada_first_layer.ff_proj.weight.shape}")
    print(f"  up_proj: {llada_first_layer.up_proj.weight.shape}")
    print(f"  ff_out: {llada_first_layer.ff_out.weight.shape}")
    
    # 复制Mistral模型的权重到LLaDA模型
    try:
        # 1. 复制词嵌入
        llada_model.model.transformer.wte.weight.data.copy_(mistral_model.model.embed_tokens.weight.data)
        
        # 2. 复制层归一化
        llada_model.model.transformer.ln_f.weight.data.copy_(mistral_model.model.norm.weight.data)
        
        # 3. 如果使用权重绑定，则不需要复制输出层（因为使用的是embed_tokens的权重）
        # 否则，我们需要复制输出层
        if not llada_config.weight_tying and hasattr(llada_model.model.transformer, 'ff_out'):
            llada_model.model.transformer.ff_out.weight.data.copy_(mistral_model.lm_head.weight.data)
        
        # 4. 复制每一层的权重
        for i in range(mistral_config.num_hidden_layers):
            mistral_layer = mistral_model.model.layers[i]
            llada_layer = llada_model.model.transformer.blocks[i] if hasattr(llada_model.model.transformer, 'blocks') else llada_model.model.transformer.block_groups[0][i]
            
            # 复制输入层归一化
            llada_layer.attn_norm.weight.data.copy_(mistral_layer.input_layernorm.weight.data)
            
            # 复制后处理层归一化
            llada_layer.ff_norm.weight.data.copy_(mistral_layer.post_attention_layernorm.weight.data)
            
            # 复制注意力层的权重
            llada_layer.q_proj.weight.data.copy_(mistral_layer.self_attn.q_proj.weight.data)
            llada_layer.k_proj.weight.data.copy_(mistral_layer.self_attn.k_proj.weight.data)
            llada_layer.v_proj.weight.data.copy_(mistral_layer.self_attn.v_proj.weight.data)
            llada_layer.attn_out.weight.data.copy_(mistral_layer.self_attn.o_proj.weight.data)
            
            # 复制MLP层的权重 - 确保形状匹配
            print(f"层 {i} MLP权重形状:")
            print(f"  gate_proj: {mistral_layer.mlp.gate_proj.weight.shape} -> ff_proj: {llada_layer.ff_proj.weight.shape}")
            print(f"  up_proj: {mistral_layer.mlp.up_proj.weight.shape} -> up_proj: {llada_layer.up_proj.weight.shape}")
            print(f"  down_proj: {mistral_layer.mlp.down_proj.weight.shape} -> ff_out: {llada_layer.ff_out.weight.shape}")
            
            # 复制权重 - 直接对应
            # Mistral: gate_proj -> act_fn -> * up_proj -> down_proj
            # LLaDA:   ff_proj -> act(这会将输入切分) -> * up_proj -> ff_out
            llada_layer.ff_proj.weight.data.copy_(mistral_layer.mlp.gate_proj.weight.data)
            llada_layer.up_proj.weight.data.copy_(mistral_layer.mlp.up_proj.weight.data)
            llada_layer.ff_out.weight.data.copy_(mistral_layer.mlp.down_proj.weight.data)
            
        print("成功复制所有权重！")
    except RuntimeError as e:
        print(f"转换模型权重时出错: {e}")
        print("这可能是由于模型配置不匹配导致的，请确保正确读取了模型的配置。")
        # 进一步诊断
        print("\n详细诊断:")
        print(f"1. Mistral hidden_size: {mistral_config.hidden_size}")
        print(f"2. Mistral intermediate_size: {mistral_config.intermediate_size}")
        print(f"3. Mistral gate_proj形状: {gate_proj_shape}")
        print(f"4. Mistral up_proj形状: {up_proj_shape}")
        print(f"5. Mistral down_proj形状: {down_proj_shape}")
        print(f"6. LLaDA d_model: {llada_config.d_model}")
        print(f"7. LLaDA mlp_hidden_size: {llada_config.mlp_hidden_size}")
        print(f"8. LLaDA ff_proj形状: {llada_first_layer.ff_proj.weight.shape}")
        print(f"9. LLaDA up_proj形状: {llada_first_layer.up_proj.weight.shape}")
        print(f"10. LLaDA ff_out形状: {llada_first_layer.ff_out.weight.shape}")
        
        raise
    
    return llada_model


def freeze_mlp_parameters(model):
    """
    冻结模型的MLP参数，仅允许注意力层的参数进行更新
    
    Args:
        model: 需要冻结部分参数的模型
        
    Returns:
        模型
    """
    for name, param in model.named_parameters():
        # 冻结MLP相关参数
        if any(x in name for x in ["mlp", "ff_proj", "up_proj", "ff_out"]):
            param.requires_grad = False
    
    return model 