import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from configuration_llada import LLaDAConfig, ModelConfig
from modeling_llada import LLaDAModelLM, LLaDAModel


def convert_mistral_to_llada(mistral_model_or_path, llada_config=None):
    """
    將Mistral模型轉換為LLaDA模型。
    
    Args:
        mistral_model_or_path: 預訓練的Mistral模型或模型路徑
        llada_config: LLaDA配置，如果為None則根據Mistral配置創建
        
    Returns:
        轉換後的LLaDA模型
    """
    # 首先嘗試獲取配置
    if isinstance(mistral_model_or_path, str):
        # 先加載配置
        mistral_config = AutoConfig.from_pretrained(mistral_model_or_path)
        print(f"已加載Mistral配置")
        # 然後加載模型
        mistral_model = AutoModelForCausalLM.from_pretrained(mistral_model_or_path)
    else:
        mistral_model = mistral_model_or_path
        mistral_config = mistral_model.config
    
    # 檢查一層的MLP維度，確保我們獲取正確的維度
    first_layer = mistral_model.model.layers[0]
    gate_proj_shape = first_layer.mlp.gate_proj.weight.shape
    up_proj_shape = first_layer.mlp.up_proj.weight.shape
    down_proj_shape = first_layer.mlp.down_proj.weight.shape
    
    print(f"Mistral MLP維度檢查:")
    print(f"  gate_proj: {gate_proj_shape}")
    print(f"  up_proj: {up_proj_shape}")
    print(f"  down_proj: {down_proj_shape}")
    
    # 創建LLaDA配置
    if llada_config is None:
        # 獲取正確的hidden_size和intermediate_size
        hidden_size = mistral_config.hidden_size
        
        # 分析Mistral模型實現
        # 在Mistral的MistralMLP.forward中:
        # down_proj(act_fn(gate_proj(x)) * up_proj(x))
        # 而LLaDA的LlamaBlock.forward中:
        # x, x_up = ff_proj(x), up_proj(x)
        # x = act(x)  
        # x = x * x_up
        # x = ff_out(x)
        
        # 創建一個直接對應Mistral的ModelConfig對象
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
            block_type="llama",  # 使用LLaMA塊類型，與Mistral兼容
            activation_type="silu",  # Mistral使用SwiGLU激活函數
            layer_norm_type="rms",  # Mistral使用RMSNorm
            rms_norm_eps=mistral_config.rms_norm_eps,
            include_bias=False,  # Mistral通常不使用偏置
            scale_logits=False,
            embedding_size=mistral_config.vocab_size,
            # 關鍵是mlp_hidden_size必須設置為gate_proj的輸出維度
            # 這樣在LLaMA的處理方式中才會正確
            mlp_hidden_size=gate_proj_shape[0],
            weight_tying=True,  # 使用權重綁定，這樣就不需要單獨的lm_head
            pad_token_id=mistral_config.pad_token_id,
            eos_token_id=mistral_config.eos_token_id,
            mask_token_id=mistral_config.pad_token_id,  # 使用pad_token作為mask_token
        )
        
        print(f"Mistral模型配置：hidden_size={hidden_size}, intermediate_size={mistral_config.intermediate_size}")
        print(f"使用的mlp_hidden_size={model_config.mlp_hidden_size}")
        
        # 創建LLaDAConfig，基於ModelConfig
        llada_config = LLaDAConfig(use_cache=False, **model_config.__dict__)
    
    # 創建LLaDA模型架構（初始化，不加載權重）
    llada_model_base = LLaDAModel(model_config, init_params=True)
    
    # 創建LLaDA模型
    llada_model = LLaDAModelLM(llada_config, model=llada_model_base)
    
    # 檢查LLaDA模型的MLP維度，確保與Mistral匹配
    llada_first_layer = llada_model.model.transformer.blocks[0] if hasattr(llada_model.model.transformer, 'blocks') else llada_model.model.transformer.block_groups[0][0]
    print(f"LLaDA MLP維度檢查:")
    print(f"  ff_proj: {llada_first_layer.ff_proj.weight.shape}")
    print(f"  up_proj: {llada_first_layer.up_proj.weight.shape}")
    print(f"  ff_out: {llada_first_layer.ff_out.weight.shape}")
    
    # 修覆各層的維度，確保輸入輸出維度正確
    for i in range(model_config.n_layers):
        llada_layer = llada_model.model.transformer.blocks[i] if hasattr(llada_model.model.transformer, 'blocks') else llada_model.model.transformer.block_groups[0][i]
        
        # 重新創建ff_proj層使其形狀與Mistral的gate_proj匹配
        if llada_layer.ff_proj.weight.shape != gate_proj_shape:
            new_ff_proj = nn.Linear(
                llada_layer.ff_proj.in_features,
                gate_proj_shape[0],
                bias=model_config.include_bias,
                device=llada_layer.ff_proj.weight.device
            )
            llada_layer.ff_proj = new_ff_proj
        
        # 重新創建up_proj層使其形狀與Mistral的up_proj匹配
        if llada_layer.up_proj.weight.shape != up_proj_shape:
            new_up_proj = nn.Linear(
                llada_layer.up_proj.in_features,
                up_proj_shape[0],
                bias=model_config.include_bias,
                device=llada_layer.up_proj.weight.device
            )
            llada_layer.up_proj = new_up_proj
        
        # 重新創建ff_out層使其形狀與Mistral的down_proj匹配
        if llada_layer.ff_out.weight.shape != (down_proj_shape[0], down_proj_shape[1]):
            new_ff_out = nn.Linear(
                down_proj_shape[1],
                down_proj_shape[0],
                bias=model_config.include_bias,
                device=llada_layer.ff_out.weight.device
            )
            llada_layer.ff_out = new_ff_out
    
    # 再次檢查LLaDA模型的MLP維度，確保修改生效
    print(f"修改後的LLaDA MLP維度檢查:")
    print(f"  ff_proj: {llada_first_layer.ff_proj.weight.shape}")
    print(f"  up_proj: {llada_first_layer.up_proj.weight.shape}")
    print(f"  ff_out: {llada_first_layer.ff_out.weight.shape}")
    
    # 覆制Mistral模型的權重到LLaDA模型
    try:
        # 1. 覆制詞嵌入
        llada_model.model.transformer.wte.weight.data.copy_(mistral_model.model.embed_tokens.weight.data)
        
        # 2. 覆制層歸一化
        llada_model.model.transformer.ln_f.weight.data.copy_(mistral_model.model.norm.weight.data)
        
        # 3. 如果使用權重綁定，則不需要覆制輸出層（因為使用的是embed_tokens的權重）
        # 否則，我們需要覆制輸出層
        if not llada_config.weight_tying and hasattr(llada_model.model.transformer, 'ff_out'):
            llada_model.model.transformer.ff_out.weight.data.copy_(mistral_model.lm_head.weight.data)
        
        # 4. 覆制每一層的權重
        for i in range(mistral_config.num_hidden_layers):
            mistral_layer = mistral_model.model.layers[i]
            llada_layer = llada_model.model.transformer.blocks[i] if hasattr(llada_model.model.transformer, 'blocks') else llada_model.model.transformer.block_groups[0][i]
            
            # 覆制輸入層歸一化
            llada_layer.attn_norm.weight.data.copy_(mistral_layer.input_layernorm.weight.data)
            
            # 覆制後處理層歸一化
            llada_layer.ff_norm.weight.data.copy_(mistral_layer.post_attention_layernorm.weight.data)
            
            # 覆制注意力層的權重
            llada_layer.q_proj.weight.data.copy_(mistral_layer.self_attn.q_proj.weight.data)
            llada_layer.k_proj.weight.data.copy_(mistral_layer.self_attn.k_proj.weight.data)
            llada_layer.v_proj.weight.data.copy_(mistral_layer.self_attn.v_proj.weight.data)
            llada_layer.attn_out.weight.data.copy_(mistral_layer.self_attn.o_proj.weight.data)
            
            # 覆制MLP層的權重 - 確保形狀匹配
            print(f"層 {i} MLP權重形狀:")
            print(f"  gate_proj: {mistral_layer.mlp.gate_proj.weight.shape} -> ff_proj: {llada_layer.ff_proj.weight.shape}")
            print(f"  up_proj: {mistral_layer.mlp.up_proj.weight.shape} -> up_proj: {llada_layer.up_proj.weight.shape}")
            print(f"  down_proj: {mistral_layer.mlp.down_proj.weight.shape} -> ff_out: {llada_layer.ff_out.weight.shape}")
            
            # 覆制權重 - 直接對應
            # Mistral: gate_proj -> act_fn -> * up_proj -> down_proj
            # LLaDA:   ff_proj -> act(這會將輸入切分) -> * up_proj -> ff_out
            llada_layer.ff_proj.weight.data.copy_(mistral_layer.mlp.gate_proj.weight.data)
            llada_layer.up_proj.weight.data.copy_(mistral_layer.mlp.up_proj.weight.data)
            llada_layer.ff_out.weight.data.copy_(mistral_layer.mlp.down_proj.weight.data)
            
        print("成功覆制所有權重！")
    except RuntimeError as e:
        print(f"轉換模型權重時出錯: {e}")
        print("這可能是由於模型配置不匹配導致的，請確保正確讀取了模型的配置。")
        # 進一步診斷
        print("\n詳細診斷:")
        print(f"1. Mistral hidden_size: {mistral_config.hidden_size}")
        print(f"2. Mistral intermediate_size: {mistral_config.intermediate_size}")
        print(f"3. Mistral gate_proj形狀: {gate_proj_shape}")
        print(f"4. Mistral up_proj形狀: {up_proj_shape}")
        print(f"5. Mistral down_proj形狀: {down_proj_shape}")
        print(f"6. LLaDA d_model: {llada_config.d_model}")
        print(f"7. LLaDA mlp_hidden_size: {llada_config.mlp_hidden_size}")
        print(f"8. LLaDA ff_proj形狀: {llada_first_layer.ff_proj.weight.shape}")
        print(f"9. LLaDA up_proj形狀: {llada_first_layer.up_proj.weight.shape}")
        print(f"10. LLaDA ff_out形狀: {llada_first_layer.ff_out.weight.shape}")
        
        raise
    
    return llada_model


def freeze_mlp_parameters(model):
    """
    凍結模型的MLP參數，僅允許注意力層的參數進行更新
    
    Args:
        model: 需要凍結部分參數的模型
        
    Returns:
        模型
    """
    for name, param in model.named_parameters():
        # 凍結MLP相關參數
        if any(x in name for x in ["mlp", "ff_proj", "up_proj", "ff_out"]):
            param.requires_grad = False
    
    return model 
