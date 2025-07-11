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

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLaDA model")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or name of pretrained Mistral model")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--cache_dir", type=str, default="./processed_data", help="Preprocessed data cache directory")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save model every N steps")
    
    # Training parameters
    parser.add_argument("--mode", type=str, choices=["pretrain", "sft"], default="pretrain", help="Training mode")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup steps ratio")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model parameters
    parser.add_argument("--activation_checkpointing", action="store_true", help="Use activation checkpointing")
    parser.add_argument("--checkpointing_strategy", type=str, default="one_in_four", 
                        choices=["whole_layer", "one_in_two", "one_in_three", "one_in_four", "fine_grained"], 
                        help="Activation checkpointing strategy")
    
    # Precision parameters
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision training")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 precision training")
    
    # Data processing parameters
    parser.add_argument("--process_data", action="store_true", help="Reprocess data")
    parser.add_argument("--txt_files", type=str, default=None, help="Glob path to txt files to process")
    parser.add_argument("--hf_dataset", type=str, default=None, help="Huggingface dataset name or local data path to process")
    parser.add_argument("--hf_text_column", type=str, default="text", help="Text column name in Huggingface dataset")
    parser.add_argument("--sft_json", type=str, default=None, help="Path to conversation JSON file for SFT mode")
    
    # Option to create example data file
    parser.add_argument("--create_example_data", action="store_true", help="Create example data file for testing")
    
    args = parser.parse_args()
    
    # Handle paths, convert relative to absolute
    if args.txt_files and not os.path.isabs(args.txt_files):
        args.txt_files = os.path.abspath(args.txt_files)
    
    # if args.hf_dataset and not args.hf_dataset.startswith("wikitext/") and not os.path.isabs(args.hf_dataset):
    #     args.hf_dataset = os.path.abspath(args.hf_dataset)
    
    return args


def train(args):
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # If need to create example data
    if args.create_example_data:
        example_data_path = os.path.join(args.data_dir, "example_data.txt")
        
        if not os.path.exists(args.data_dir):
            os.makedirs(args.data_dir, exist_ok=True)
            
        with open(example_data_path, "w", encoding="utf-8") as f:
            f.write("This is an example data file for testing LLaDA model training.\n")
            f.write("LLaDA is an architecture that converts autoregressive language models into mask predictors.\n")
            f.write("It enables the model to attend bidirectionally by removing the causal mask from the Transformer.\n")
            for i in range(100):
                f.write(f"This is line {i+1} of example data, providing more training content.\n")
        
        logger.info(f"Created example data file: {example_data_path}")
        
        # If no data source specified, use example data
        if not args.txt_files and not args.hf_dataset:
            args.txt_files = example_data_path
            logger.info(f"Using example data for training: {args.txt_files}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Ensure tokenizer has correct special tokens, use bos_token_id as mask_token_id
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id:
        logger.info(f"Using bos_token_id as mask_token_id: {tokenizer.bos_token_id}")
        mask_token_id = tokenizer.bos_token_id
    else:
        logger.warning(f"Tokenizer does not have bos_token_id, using eos_token_id instead: {tokenizer.eos_token_id}")
        mask_token_id = tokenizer.eos_token_id
    
    # Set mask_token_id as global constant for use elsewhere
    os.environ["LLADA_MASK_TOKEN_ID"] = str(mask_token_id)
    
    # Data processing
    processor = TextDatasetProcessor(
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        cache_dir=args.cache_dir
    )

    # Add test_mode argument if present
    test_mode = getattr(args, 'test_mode', False)

    # Process data according to training mode
    if args.mode == "pretrain":
        # Check if processed npy file exists
        npy_filename = f"pretrain_{args.hf_dataset}_{'train' if not hasattr(args, 'split') else args.split}.npy" if args.hf_dataset else None
        npy_path = os.path.join(args.cache_dir, npy_filename) if npy_filename else None
        if npy_path and os.path.exists(npy_path):
            logger.info(f"Found existing processed data file: {npy_path}")
            npy_paths = [npy_path]
        else:
            if args.process_data or (not args.txt_files and not args.hf_dataset):
                if args.txt_files:
                    npy_paths = processor.process_txt_files(args.txt_files, force_reprocess=args.process_data)
                elif args.hf_dataset:
                    npy_paths = processor.process_hf_dataset(
                        args.hf_dataset, 
                        text_column=args.hf_text_column,
                        force_reprocess=args.process_data,
                        test_mode=test_mode
                    )
                else:
                    raise ValueError("Pretrain mode requires txt_files or hf_dataset parameter")
            else:
                # Use already processed data
                npy_paths = [os.path.join(args.cache_dir, f) for f in os.listdir(args.cache_dir) 
                             if f.startswith("pretrain") and f.endswith(".npy")]
                if not npy_paths:
                    logger.warning("No preprocessed data files found. Please use --process_data to process data, or provide correct --txt_files or --hf_dataset parameter.")
                    if args.hf_dataset:
                        logger.info(f"Processing dataset: {args.hf_dataset}")
                        npy_paths = processor.process_hf_dataset(
                            args.hf_dataset, 
                            text_column=args.hf_text_column,
                            force_reprocess=True,
                            test_mode=test_mode
                        )
                    elif args.txt_files:
                        logger.info(f"Processing text files: {args.txt_files}")
                        npy_paths = processor.process_txt_files(args.txt_files, force_reprocess=True)
                    else:
                        raise ValueError("No preprocessed data files found and no txt_files or hf_dataset parameter specified")
        # Ensure data files found
        if not npy_paths:
            raise ValueError("Could not find or process data files, please check your parameters")
        dataset = NpyDataset(npy_paths, is_sft=False)
    else:  # SFT mode
        if not args.sft_json:
            raise ValueError("SFT mode requires sft_json parameter")
        
        npy_paths = processor.process_conversations(
            args.sft_json, 
            force_reprocess=args.process_data
        )
        dataset = NpyDataset(npy_paths, is_sft=True)
    
    # Create data loader
    dataloader = create_dataloader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and convert model
    logger.info(f"Loading Mistral model: {args.model_name_or_path}")
    llada_model = convert_mistral_to_llada(args.model_name_or_path)

    # Freeze parameters according to model type
    if 'qwen' in args.model_name_or_path.lower():
        logger.info("Detected Qwen model: freezing all but attention layers (q_proj, k_proj, v_proj, o_proj)")
        for name, param in llada_model.named_parameters():
            if (
                ".self_attn.q_proj." in name or
                ".self_attn.k_proj." in name or
                ".self_attn.v_proj." in name or
                ".self_attn.o_proj." in name
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        logger.info("Freezing MLP parameters, only updating attention layers")
        freeze_mlp_parameters(llada_model)

    # Print parameter statistics
    total_params = sum(p.numel() for p in llada_model.parameters())
    trainable_params = sum(p.numel() for p in llada_model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params/1e9:.3f}B")
    logger.info(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    # Activation checkpointing
    if args.activation_checkpointing:
        strategy = getattr(ActivationCheckpointingStrategy, args.checkpointing_strategy)
        logger.info(f"Using activation checkpointing strategy: {strategy}")
        llada_model.model.set_activation_checkpointing(strategy)
    
    # Move model to device
    llada_model.to(device)
    
    # Prepare optimizer
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
    
    # Calculate total training steps
    total_steps = len(dataloader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Use mixed precision training
    scaler = None
    if args.bf16 and torch.cuda.is_bf16_supported():
        logger.info("Using BF16 precision training")
        amp_dtype = torch.bfloat16
        autocast_enabled = True
    elif args.fp16:
        logger.info("Using FP16 precision training")
        amp_dtype = torch.float16
        autocast_enabled = True
        scaler = torch.cuda.amp.GradScaler()
    else:
        logger.info("Using FP32 precision training")
        amp_dtype = torch.float32
        autocast_enabled = False
    
    # Training loop
    global_step = 0
    llada_model.train()
    
    for epoch in range(args.epochs):
        logger.info(f"Starting Epoch {epoch+1}/{args.epochs}")
        
        epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(epoch_iterator):
            # Move data to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=amp_dtype):
                # Compute loss according to mode
                if args.mode == "pretrain":
                    loss = compute_pretrain_loss(llada_model, batch, mask_token_id)
                else:  # SFT mode
                    loss = compute_sft_loss(llada_model, batch, mask_token_id)
                
                # Gradient accumulation
                loss = loss / args.gradient_accumulation_steps
            
            # Gradient calculation and update
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
                
                # Logging
                if global_step % args.logging_steps == 0:
                    logger.info(f"Step {global_step} - Loss: {loss.item() * args.gradient_accumulation_steps:.4f}")
                
                # Save model
                if global_step % args.save_steps == 0:
                    output_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(output_path, exist_ok=True)
                    
                    # Save model
                    llada_model.save_pretrained(output_path)
                    tokenizer.save_pretrained(output_path)
                    
                    # Save mask_token_id
                    config_path = os.path.join(output_path, "config.json")
                    if os.path.exists(config_path):
                        import json
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        config['mask_token_id'] = mask_token_id
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config, f, indent=2)
                    
                    logger.info(f"Saved model checkpoint to {output_path}")
    
    # Save final model
    final_output_path = os.path.join(args.output_dir, "final-model")
    os.makedirs(final_output_path, exist_ok=True)
    
    llada_model.save_pretrained(final_output_path)
    tokenizer.save_pretrained(final_output_path)
    
    # Save mask_token_id
    config_path = os.path.join(final_output_path, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        config['mask_token_id'] = mask_token_id
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    logger.info(f"Training complete, final model saved to {final_output_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
