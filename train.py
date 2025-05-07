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

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="训练LLaDA模型")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="预训练Mistral模型的路径或名称")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--cache_dir", type=str, default="./processed_data", help="预处理数据缓存目录")
    parser.add_argument("--logging_steps", type=int, default=10, help="每多少步记录一次日志")
    parser.add_argument("--save_steps", type=int, default=100, help="每多少步保存一次模型")
    
    # 训练参数
    parser.add_argument("--mode", type=str, choices=["pretrain", "sft"], default="pretrain", help="训练模式")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="序列最大长度")
    parser.add_argument("--batch_size", type=int, default=8, help="训练批次大小")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热步数比例")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 模型参数
    parser.add_argument("--activation_checkpointing", action="store_true", help="是否使用激活检查点")
    parser.add_argument("--checkpointing_strategy", type=str, default="one_in_four", 
                        choices=["whole_layer", "one_in_two", "one_in_three", "one_in_four", "fine_grained"], 
                        help="激活检查点策略")
    
    # 精度参数
    parser.add_argument("--bf16", action="store_true", help="是否使用bf16精度训练")
    parser.add_argument("--fp16", action="store_true", help="是否使用fp16精度训练")
    
    # 数据处理参数
    parser.add_argument("--process_data", action="store_true", help="重新处理数据")
    parser.add_argument("--txt_files", type=str, default=None, help="要处理的txt文件通配符路径")
    parser.add_argument("--hf_dataset", type=str, default=None, help="要处理的Huggingface数据集名称或本地数据路径")
    parser.add_argument("--hf_text_column", type=str, default="text", help="Huggingface数据集中的文本列名称")
    parser.add_argument("--sft_json", type=str, default=None, help="SFT模式下的对话JSON文件路径")
    
    # 创建示例数据文件的选项
    parser.add_argument("--create_example_data", action="store_true", help="创建示例数据文件用于测试")
    
    args = parser.parse_args()
    
    # 处理路径，将相对路径转换为绝对路径
    if args.txt_files and not os.path.isabs(args.txt_files):
        args.txt_files = os.path.abspath(args.txt_files)
    
    if args.hf_dataset and not args.hf_dataset.startswith("wikitext/") and not os.path.isabs(args.hf_dataset):
        args.hf_dataset = os.path.abspath(args.hf_dataset)
    
    return args


def train(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # 如果需要创建示例数据
    if args.create_example_data:
        example_data_path = os.path.join(args.data_dir, "example_data.txt")
        
        if not os.path.exists(args.data_dir):
            os.makedirs(args.data_dir, exist_ok=True)
            
        with open(example_data_path, "w", encoding="utf-8") as f:
            f.write("这是一个示例数据文件，用于测试LLaDA模型训练。\n")
            f.write("LLaDA是一种将自回归语言模型转换为掩码预测器的架构。\n")
            f.write("它通过移除Transformer的因果掩码，使模型能够双向注意。\n")
            for i in range(100):
                f.write(f"这是示例数据的第{i+1}行文本，提供更多训练内容。\n")
        
        logger.info(f"创建了示例数据文件: {example_data_path}")
        
        # 如果没有指定数据源，使用示例数据
        if not args.txt_files and not args.hf_dataset:
            args.txt_files = example_data_path
            logger.info(f"将使用示例数据进行训练: {args.txt_files}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # 确保tokenizer有正确的特殊标记，使用bos_token_id作为mask_token_id
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id:
        logger.info(f"使用bos_token_id作为mask_token_id: {tokenizer.bos_token_id}")
        mask_token_id = tokenizer.bos_token_id
    else:
        logger.warning(f"Tokenizer没有bos_token_id，将使用eos_token_id代替: {tokenizer.eos_token_id}")
        mask_token_id = tokenizer.eos_token_id
    
    # 将mask_token_id设置为全局常量，方便在其他地方使用
    os.environ["LLADA_MASK_TOKEN_ID"] = str(mask_token_id)
    
    # 数据处理
    processor = TextDatasetProcessor(
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        cache_dir=args.cache_dir
    )
    
    # 根据训练模式处理数据
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
                raise ValueError("预训练模式需要指定txt_files或hf_dataset参数")
        else:
            # 使用已处理的数据
            npy_paths = [os.path.join(args.cache_dir, f) for f in os.listdir(args.cache_dir) 
                         if f.startswith("pretrain") and f.endswith(".npy")]
            
            # 如果没有找到预处理的数据文件，则提示处理数据
            if not npy_paths:
                logger.warning("未找到预处理的数据文件。请使用--process_data参数进行数据处理，或提供正确的--txt_files或--hf_dataset参数。")
                if args.hf_dataset:
                    logger.info(f"正在处理数据集: {args.hf_dataset}")
                    npy_paths = processor.process_hf_dataset(
                        args.hf_dataset, 
                        text_column=args.hf_text_column,
                        force_reprocess=True
                    )
                elif args.txt_files:
                    logger.info(f"正在处理文本文件: {args.txt_files}")
                    npy_paths = processor.process_txt_files(args.txt_files, force_reprocess=True)
                else:
                    raise ValueError("未找到预处理的数据文件，且未指定txt_files或hf_dataset参数")
        
        # 确保找到了数据文件
        if not npy_paths:
            raise ValueError("无法找到或处理数据文件，请检查您的参数")
            
        dataset = NpyDataset(npy_paths, is_sft=False)
    else:  # SFT模式
        if not args.sft_json:
            raise ValueError("SFT模式需要指定sft_json参数")
        
        npy_paths = processor.process_conversations(
            args.sft_json, 
            force_reprocess=args.process_data
        )
        dataset = NpyDataset(npy_paths, is_sft=True)
    
    # 创建数据加载器
    dataloader = create_dataloader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载并转换模型
    logger.info(f"加载Mistral模型: {args.model_name_or_path}")
    llada_model = convert_mistral_to_llada(args.model_name_or_path)
    
    # 冻结MLP参数
    logger.info("冻结MLP参数，只更新注意力层")
    freeze_mlp_parameters(llada_model)
    
    # 激活检查点
    if args.activation_checkpointing:
        strategy = getattr(ActivationCheckpointingStrategy, args.checkpointing_strategy)
        logger.info(f"使用激活检查点策略: {strategy}")
        llada_model.model.set_activation_checkpointing(strategy)
    
    # 将模型移至设备
    llada_model.to(device)
    
    # 准备优化器
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
    
    # 计算总训练步数
    total_steps = len(dataloader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # 使用混合精度训练
    scaler = None
    if args.bf16 and torch.cuda.is_bf16_supported():
        logger.info("使用 BF16 精度训练")
        amp_dtype = torch.bfloat16
        autocast_enabled = True
    elif args.fp16:
        logger.info("使用 FP16 精度训练")
        amp_dtype = torch.float16
        autocast_enabled = True
        scaler = torch.cuda.amp.GradScaler()
    else:
        logger.info("使用 FP32 精度训练")
        amp_dtype = torch.float32
        autocast_enabled = False
    
    # 训练循环
    global_step = 0
    llada_model.train()
    
    for epoch in range(args.epochs):
        logger.info(f"开始 Epoch {epoch+1}/{args.epochs}")
        
        epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(epoch_iterator):
            # 将数据移至设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=amp_dtype):
                # 根据模式计算损失
                if args.mode == "pretrain":
                    loss = compute_pretrain_loss(llada_model, batch, mask_token_id)
                else:  # SFT模式
                    loss = compute_sft_loss(llada_model, batch, mask_token_id)
                
                # 梯度累积
                loss = loss / args.gradient_accumulation_steps
            
            # 梯度计算与更新
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
                
                # 日志记录
                if global_step % args.logging_steps == 0:
                    logger.info(f"步骤 {global_step} - 损失: {loss.item() * args.gradient_accumulation_steps:.4f}")
                
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
                    
                    logger.info(f"保存模型检查点到 {output_path}")
    
    # 保存最终模型
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
    
    logger.info(f"训练完成，保存最终模型到 {final_output_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)