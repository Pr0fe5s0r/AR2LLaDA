import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import logging
from tqdm import tqdm
from datasets import load_dataset
import json

logger = logging.getLogger(__name__)


class TextDatasetProcessor:
    """
    處理文本數據集，將其轉換為npy文件以加速訓練
    """
    def __init__(self, tokenizer, max_length=2048, overlap=0, cache_dir="processed_data"):
        """
        初始化文本數據處理器
        
        Args:
            tokenizer: 用於標記化文本的分詞器
            max_length: 序列的最大長度
            overlap: 在連續片段之間的重疊token數量
            cache_dir: 緩存處理後數據的目錄
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap = overlap
        self.cache_dir = cache_dir
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def process_txt_files(self, txt_files_pattern, output_prefix="pretrain", force_reprocess=False):
        """
        處理多個文本文件並保存為npy文件
        
        Args:
            txt_files_pattern: 文本文件的glob模式，例如 "data/*.txt"
            output_prefix: 輸出文件的前綴
            force_reprocess: 是否強制重新處理，即使已有處理好的文件
            
        Returns:
            已處理數據的路徑列表
        """
        # 檢查是否已存在處理好的文件
        output_path = os.path.join(self.cache_dir, f"{output_prefix}_data.npy")
        if os.path.exists(output_path) and not force_reprocess:
            logger.info(f"找到已處理的數據文件: {output_path}")
            return [output_path]
        
        # 獲取文件列表
        txt_files = glob.glob(txt_files_pattern)
        if not txt_files:
            raise ValueError(f"未找到匹配的文件: {txt_files_pattern}")
        
        all_tokenized_data = []
        
        # 處理每個文件
        for txt_file in tqdm(txt_files, desc="處理文本文件"):
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 標記化文本
            tokenized_data = self.tokenizer.encode(text)
            all_tokenized_data.extend(tokenized_data)
        
        # 切分為固定長度的序列
        sequences = []
        for i in range(0, len(all_tokenized_data) - self.max_length + 1, self.max_length - self.overlap):
            sequences.append(all_tokenized_data[i:i + self.max_length])
        
        # 轉換為numpy數組並保存
        sequences_array = np.array(sequences, dtype=np.int32)
        np.save(output_path, sequences_array)
        
        logger.info(f"已處理並保存數據到: {output_path}")
        return [output_path]
    
    def process_hf_dataset(self, dataset_name, text_column="text", split="train", output_prefix="pretrain", force_reprocess=False):
        """
        處理Huggingface數據集並保存為npy文件
        
        Args:
            dataset_name: Huggingface數據集名稱或本地路徑
            text_column: 文本列的名稱
            split: 要處理的數據分割
            output_prefix: 輸出文件的前綴
            force_reprocess: 是否強制重新處理，即使已有處理好的文件
            
        Returns:
            已處理數據的路徑列表
        """
        # 提取數據集名稱用於文件命名
        dataset_id = dataset_name.replace("/", "_").replace("\\", "_").replace(".", "_")
        if dataset_name.startswith("./") or dataset_name.startswith(".\\"):
            dataset_id = os.path.basename(dataset_name)
        
        # 檢查是否已存在處理好的文件
        output_path = os.path.join(self.cache_dir, f"{output_prefix}_{dataset_id}_{split}.npy")
        if os.path.exists(output_path) and not force_reprocess:
            logger.info(f"找到已處理的數據文件: {output_path}")
            return [output_path]
        
        try:
            # 首先嘗試作為本地路徑加載
            if os.path.exists(dataset_name):
                if os.path.isdir(dataset_name):
                    # 是本地目錄，嘗試作為數據集加載
                    logger.info(f"從本地路徑加載數據集: {dataset_name}")
                    dataset = load_dataset(dataset_name, split=split)
                else:
                    # 是本地文件，直接讀取
                    logger.info(f"讀取本地文件: {dataset_name}")
                    with open(dataset_name, 'r', encoding='utf-8') as f:
                        texts = [line.strip() for line in f if line.strip()]
                    
                    all_tokenized_data = []
                    for text in tqdm(texts, desc=f"處理 {dataset_name}"):
                        tokenized_data = self.tokenizer.encode(text)
                        all_tokenized_data.extend(tokenized_data)
                    
                    # 切分為固定長度的序列
                    sequences = []
                    for i in range(0, len(all_tokenized_data) - self.max_length + 1, self.max_length - self.overlap):
                        sequences.append(all_tokenized_data[i:i + self.max_length])
                    
                    # 轉換為numpy數組並保存
                    sequences_array = np.array(sequences, dtype=np.int32)
                    np.save(output_path, sequences_array)
                    
                    logger.info(f"已處理並保存數據到: {output_path}")
                    return [output_path]
            else:
                # 嘗試從Hugging Face Hub加載
                logger.info(f"從Hugging Face Hub加載數據集: {dataset_name}")
                dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            logger.error(f"加載數據集 {dataset_name} 時出錯: {str(e)}")
            raise ValueError(f"無法加載數據集 {dataset_name}: {str(e)}")
        
        all_tokenized_data = []
        
        # 處理數據集中的每個樣本
        for item in tqdm(dataset, desc=f"處理 {dataset_name}"):
            text = item[text_column]
            
            # 標記化文本
            tokenized_data = self.tokenizer.encode(text)
            all_tokenized_data.extend(tokenized_data)
        
        # 切分為固定長度的序列
        sequences = []
        for i in range(0, len(all_tokenized_data) - self.max_length + 1, self.max_length - self.overlap):
            sequences.append(all_tokenized_data[i:i + self.max_length])
        
        # 轉換為numpy數組並保存
        sequences_array = np.array(sequences, dtype=np.int32)
        np.save(output_path, sequences_array)
        
        logger.info(f"已處理並保存數據到: {output_path}")
        return [output_path]
    
    def process_conversations(self, input_file, output_prefix="sft", force_reprocess=False):
        """
        處理對話數據並保存為SFT格式
        
        Args:
            input_file: 包含對話的JSON文件路徑
            output_prefix: 輸出文件的前綴
            force_reprocess: 是否強制重新處理，即使已有處理好的文件
            
        Returns:
            已處理數據的路徑元組 (input_ids_path, prompt_lengths_path)
        """
        # 檢查是否已存在處理好的文件
        input_ids_path = os.path.join(self.cache_dir, f"{output_prefix}_input_ids.npy")
        prompt_lengths_path = os.path.join(self.cache_dir, f"{output_prefix}_prompt_lengths.npy")
        
        if os.path.exists(input_ids_path) and os.path.exists(prompt_lengths_path) and not force_reprocess:
            logger.info(f"找到已處理的SFT數據文件")
            return input_ids_path, prompt_lengths_path
        
        # 加載對話數據
        with open(input_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        all_input_ids = []
        all_prompt_lengths = []
        
        # 處理每個對話
        for conv in tqdm(conversations, desc="處理對話數據"):
            user_msg = conv["user"]
            assistant_msg = conv["assistant"]
            
            # 將對話格式化為 SFT 格式
            user_tokens = self.tokenizer.encode("<start_id>user<end_id>\n" + user_msg + "<eot_id>")
            assistant_tokens = self.tokenizer.encode("<start_id>assistant<end_id>\n" + assistant_msg)
            
            # 添加結束標記並填充到 max_length
            full_tokens = [self.tokenizer.bos_token_id] + user_tokens + assistant_tokens + [self.tokenizer.eos_token_id]
            prompt_length = len(user_tokens) + 1  # +1 是為了包括 bos_token
            
            # 如果序列太長，則截斷
            if len(full_tokens) > self.max_length:
                full_tokens = full_tokens[:self.max_length]
            
            # 如果序列太短，則填充
            if len(full_tokens) < self.max_length:
                padding = [self.tokenizer.eos_token_id] * (self.max_length - len(full_tokens))
                full_tokens.extend(padding)
            
            all_input_ids.append(full_tokens)
            all_prompt_lengths.append(prompt_length)
        
        # 轉換為numpy數組並保存
        input_ids_array = np.array(all_input_ids, dtype=np.int32)
        prompt_lengths_array = np.array(all_prompt_lengths, dtype=np.int32)
        
        np.save(input_ids_path, input_ids_array)
        np.save(prompt_lengths_path, prompt_lengths_array)
        
        logger.info(f"已處理並保存SFT數據")
        return input_ids_path, prompt_lengths_path


class NpyDataset(Dataset):
    """
    使用預處理的npy文件創建PyTorch數據集
    """
    def __init__(self, npy_files, is_sft=False):
        """
        初始化npy數據集
        
        Args:
            npy_files: npy文件路徑或路徑列表
            is_sft: 是否為SFT格式的數據集
        """
        self.is_sft = is_sft
        
        if is_sft:
            if not isinstance(npy_files, tuple) or len(npy_files) != 2:
                raise ValueError("SFT模式需要提供輸入ID和提示長度的路徑元組")
            
            self.input_ids = np.load(npy_files[0])
            self.prompt_lengths = np.load(npy_files[1])
            assert len(self.input_ids) == len(self.prompt_lengths), "輸入ID和提示長度數量不匹配"
        else:
            if isinstance(npy_files, list):
                # 合並多個npy文件
                self.data = []
                for npy_file in npy_files:
                    self.data.append(np.load(npy_file))
                self.data = np.concatenate(self.data, axis=0)
            else:
                self.data = np.load(npy_files)
    
    def __len__(self):
        if self.is_sft:
            return len(self.input_ids)
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.is_sft:
            return {
                "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
                "prompt_lengths": torch.tensor(self.prompt_lengths[idx], dtype=torch.long)
            }
        return {
            "input_ids": torch.tensor(self.data[idx], dtype=torch.long)
        }


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    """
    創建數據加載器
    
    Args:
        dataset: 數據集
        batch_size: 批次大小
        shuffle: 是否打亂數據
        num_workers: 數據加載的工作進程數
        
    Returns:
        數據加載器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    ) 
