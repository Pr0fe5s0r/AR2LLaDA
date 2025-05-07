import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
import numpy as np
from configuration_llada import LLaDAConfig, ActivationCheckpointingStrategy
from llada_loss import compute_pretrain_loss, compute_sft_loss
from data_processor import TextDatasetProcessor, NpyDataset, create_dataloader
from mistral_to_llada import convert_mistral_to_llada, freeze_mlp_parameters

# 設置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="訓練LLaDA模型")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="預訓練Mistral模型的路徑或名稱")
    parser.add_argument("--output_dir", type=str, default="./output", help="輸出目錄")
    parser.add_argument("--data_dir", type=str, default="./data", help="數據目錄")
    parser.add_argument("--cache_dir", type=str, default="./processed_data", help="預處理數據緩存目錄")
    parser.add_argument("--logging_steps", type=int, default=10, help="每多少步記錄一次日志")
    parser.add_argument("--save_steps", type=int, default=100, help="每多少步保存一次模型")
    
    # 訓練參數
    parser.add_argument("--mode", type=str, choices=["pretrain", "sft"], default="pretrain", help="訓練模式")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="序列最大長度")
    parser.add_argument("--batch_size", type=int, default=8, help="訓練批次大小")
    parser.add_argument("--epochs", type=int, default=3, help="訓練輪數")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="學習率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="權重衰減")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="預熱步數比例")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累積步數")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    
    # 模型參數
    parser.add_argument("--activation_checkpointing", action="store_true", help="是否使用激活檢查點")
    parser.add_argument("--checkpointing_strategy", type=str, default="one_in_four", 
                        choices=["whole_layer", "one_in_two", "one_in_three", "one_in_four", "fine_grained"], 
                        help="激活檢查點策略")
    
    # 精度參數
    parser.add_argument("--bf16", action="store_true", help="是否使用bf16精度訓練")
    parser.add_argument("--fp16", action="store_true", help="是否使用fp16精度訓練")
    
    # 數據處理參數
    parser.add_argument("--process_data", action="store_true", help="重新處理數據")
    parser.add_argument("--txt_files", type=str, default=None, help="要處理的txt文件通配符路徑")
    parser.add_argument("--hf_dataset", type=str, default=None, help="要處理的Huggingface數據集名稱或本地數據路徑")
    parser.add_argument("--hf_text_column", type=str, default="text", help="Huggingface數據集中的文本列名稱")
    parser.add_argument("--sft_json", type=str, default=None, help="SFT模式下的對話JSON文件路徑")
    
    # 創建示例數據文件的選項
    parser.add_argument("--create_example_data", action="store_true", help="創建示例數據文件用於測試")
    
    args = parser.parse_args()
    
    # 處理路徑，將相對路徑轉換為絕對路徑
    if args.txt_files and not os.path.isabs(args.txt_files):
        args.txt_files = os.path.abspath(args.txt_files)
    
    if args.hf_dataset and not args.hf_dataset.startswith("wikitext/") and not os.path.isabs(args.hf_dataset):
        args.hf_dataset = os.path.abspath(args.hf_dataset)
    
    return args


def train(args):
    # 設置隨機種子
    set_seed(args.seed)
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # 如果需要創建示例數據
    if args.create_example_data:
        example_data_path = os.path.join(args.data_dir, "example_data.txt")
        
        if not os.path.exists(args.data_dir):
            os.makedirs(args.data_dir, exist_ok=True)
            
        with open(example_data_path, "w", encoding="utf-8") as f:
            f.write("這是一個示例數據文件，用於測試LLaDA模型訓練。\n")
            f.write("LLaDA是一種將自回歸語言模型轉換為掩碼預測器的架構。\n")
            f.write("它通過移除Transformer的因果掩碼，使模型能夠雙向注意。\n")
            for i in range(100):
                f.write(f"這是示例數據的第{i+1}行文本，提供更多訓練內容。\n")
        
        logger.info(f"創建了示例數據文件: {example_data_path}")
        
        # 如果沒有指定數據源，使用示例數據
        if not args.txt_files and not args.hf_dataset:
            args.txt_files = example_data_path
            logger.info(f"將使用示例數據進行訓練: {args.txt_files}")
    
    # 加載tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # 確保tokenizer有正確的特殊標記，使用bos_token_id作為mask_token_id
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id:
        logger.info(f"使用bos_token_id作為mask_token_id: {tokenizer.bos_token_id}")
        mask_token_id = tokenizer.bos_token_id
    else:
        logger.warning(f"Tokenizer沒有bos_token_id，將使用eos_token_id代替: {tokenizer.eos_token_id}")
        mask_token_id = tokenizer.eos_token_id
    
    # 將mask_token_id設置為全局常量，方便在其他地方使用
    os.environ["LLADA_MASK_TOKEN_ID"] = str(mask_token_id)
    
    # 數據處理
    processor = TextDatasetProcessor(
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        cache_dir=args.cache_dir
    )
    
    # 根據訓練模式處理數據
    if args.mode == "pretrain":
        if args.process_data or (not args.txt_files and not args.hf_dataset):
            if args.txt_files:
                npy_paths = processor.process_txt_files(args.txt_files, force_reprocess=args.process_data)
            elif args.hf_dataset:
                npy_paths = processor.process_hf_dataset(
                    args.hf_dataset, 
                    text_column=args.hf_text_column,
                    force_reprocess=args.process_data
                )
            else:
                raise ValueError("預訓練模式需要指定txt_files或hf_dataset參數")
        else:
            # 使用已處理的數據
            npy_paths = [os.path.join(args.cache_dir, f) for f in os.listdir(args.cache_dir) 
                         if f.startswith("pretrain") and f.endswith(".npy")]
            
            # 如果沒有找到預處理的數據文件，則提示處理數據
            if not npy_paths:
                logger.warning("未找到預處理的數據文件。請使用--process_data參數進行數據處理，或提供正確的--txt_files或--hf_dataset參數。")
                if args.hf_dataset:
                    logger.info(f"正在處理數據集: {args.hf_dataset}")
                    npy_paths = processor.process_hf_dataset(
                        args.hf_dataset, 
                        text_column=args.hf_text_column,
                        force_reprocess=True
                    )
                elif args.txt_files:
                    logger.info(f"正在處理文本文件: {args.txt_files}")
                    npy_paths = processor.process_txt_files(args.txt_files, force_reprocess=True)
                else:
                    raise ValueError("未找到預處理的數據文件，且未指定txt_files或hf_dataset參數")
        
        # 確保找到了數據文件
        if not npy_paths:
            raise ValueError("無法找到或處理數據文件，請檢查您的參數")
            
        dataset = NpyDataset(npy_paths, is_sft=False)
    else:  # SFT模式
        if not args.sft_json:
            raise ValueError("SFT模式需要指定sft_json參數")
        
        npy_paths = processor.process_conversations(
            args.sft_json, 
            force_reprocess=args.process_data
        )
        dataset = NpyDataset(npy_paths, is_sft=True)
    
    # 創建數據加載器
    dataloader = create_dataloader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加載並轉換模型
    logger.info(f"加載Mistral模型: {args.model_name_or_path}")
    llada_model = convert_mistral_to_llada(args.model_name_or_path)
    
    # 凍結MLP參數
    logger.info("凍結MLP參數，只更新注意力層")
    freeze_mlp_parameters(llada_model)
    
    # 激活檢查點
    if args.activation_checkpointing:
        strategy = getattr(ActivationCheckpointingStrategy, args.checkpointing_strategy)
        logger.info(f"使用激活檢查點策略: {strategy}")
        llada_model.model.set_activation_checkpointing(strategy)
    
    # 將模型移至設備
    llada_model.to(device)
    
    # 準備優化器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in llada_model.named_parameters() 
                      if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in llada_model.named_parameters() 
                      if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # 計算總訓練步數
    total_steps = len(dataloader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # 創建學習率調度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # 使用混合精度訓練
    scaler = None
    if args.bf16 and torch.cuda.is_bf16_supported():
        logger.info("使用 BF16 精度訓練")
        amp_dtype = torch.bfloat16
        autocast_enabled = True
    elif args.fp16:
        logger.info("使用 FP16 精度訓練")
        amp_dtype = torch.float16
        autocast_enabled = True
        scaler = torch.cuda.amp.GradScaler()
    else:
        logger.info("使用 FP32 精度訓練")
        amp_dtype = torch.float32
        autocast_enabled = False
    
    # 訓練循環
    global_step = 0
    llada_model.train()
    
    for epoch in range(args.epochs):
        logger.info(f"開始 Epoch {epoch+1}/{args.epochs}")
        
        epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(epoch_iterator):
            # 將數據移至設備
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=amp_dtype):
                # 根據模式計算損失
                if args.mode == "pretrain":
                    loss = compute_pretrain_loss(llada_model, batch, mask_token_id)
                else:  # SFT模式
                    loss = compute_sft_loss(llada_model, batch, mask_token_id)
                
                # 梯度累積
                loss = loss / args.gradient_accumulation_steps
            
            # 梯度計算與更新
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(llada_model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(llada_model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 日志記錄
                if global_step % args.logging_steps == 0:
                    logger.info(f"步驟 {global_step} - 損失: {loss.item() * args.gradient_accumulation_steps:.4f}")
                
                # 保存模型
                if global_step % args.save_steps == 0:
                    output_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(output_path, exist_ok=True)
                    
                    # 保存模型
                    llada_model.save_pretrained(output_path)
                    tokenizer.save_pretrained(output_path)
                    
                    # 保存mask_token_id
                    config_path = os.path.join(output_path, "config.json")
                    if os.path.exists(config_path):
                        import json
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        config['mask_token_id'] = mask_token_id
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config, f, indent=2)
                    
                    logger.info(f"保存模型檢查點到 {output_path}")
    
    # 保存最終模型
    final_output_path = os.path.join(args.output_dir, "final-model")
    os.makedirs(final_output_path, exist_ok=True)
    
    llada_model.save_pretrained(final_output_path)
    tokenizer.save_pretrained(final_output_path)
    
    # 保存mask_token_id
    config_path = os.path.join(final_output_path, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        config['mask_token_id'] = mask_token_id
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    logger.info(f"訓練完成，保存最終模型到 {final_output_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
