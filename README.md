# Mistral轉LLaDA模型訓練

本項目實現了Mistral模型到LLaDA模型的轉換和微調，參考了LLaDA論文中的方法。

LLaDA（Language Model as a Denoising Autoencoder）是一種將自回歸語言模型轉換為掩碼預測器的架構，主要修改了Transformer架構中的自注意力機制，移除了因果掩碼。

## 項目結構

- `llada_loss.py`: LLaDA的核心損失函數實現
- `mistral_to_llada.py`: Mistral模型到LLaDA模型的轉換工具
- `data_processor.py`: 數據處理工具，支持將文本數據轉換為npy文件
- `train.py`: 訓練腳本，支持預訓練和SFT兩種模式
- `configuration_llada.py`: LLaDA模型配置（從項目中已有）
- `modeling_llada.py`: LLaDA模型定義（從項目中已有）

## 環境要求

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- 其他依賴項可通過`pip install -r requirements.txt`安裝

## 使用方法

### 數據預處理

本項目支持從本地TXT文件或Huggingface數據集加載數據，並將其處理為npy格式以加速訓練：

```bash
# 預處理本地TXT文件
python train.py \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --txt_files "data/*.txt" \
  --process_data \
  --mode pretrain \
  --output_dir ./output

# 預處理Huggingface數據集
python train.py \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --hf_dataset "wikitext/wikitext-2-v1" \
  --hf_text_column "text" \
  --process_data \
  --mode pretrain \
  --output_dir ./output

# 創建並使用示例數據進行快速測試
python train.py \
  --model_name_or_path Locutusque/TinyMistral-248M \
  --create_example_data \
  --process_data \
  --mode pretrain \
  --batch_size 4 \
  --epochs 1 \
  --max_seq_length 512 \
  --output_dir ./output
```

### 預訓練

```bash
python train.py \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --mode pretrain \
  --txt_files "data/*.txt" \
  --batch_size 8 \
  --epochs 3 \
  --learning_rate 5e-5 \
  --max_seq_length 2048 \
  --activation_checkpointing \
  --checkpointing_strategy one_in_four \
  --bf16 \
  --output_dir ./output
```

### SFT（監督微調）

```bash
python train.py \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --mode sft \
  --sft_json "data/conversations.json" \
  --batch_size 8 \
  --epochs 3 \
  --learning_rate 2e-5 \
  --max_seq_length 2048 \
  --activation_checkpointing \
  --checkpointing_strategy one_in_four \
  --bf16 \
  --output_dir ./output
```

## SFT數據格式

SFT數據應為JSON格式，包含用戶和助手對話：

```json
[
  {
    "user": "What is the capital of France?",
    "assistant": "The capital of France is Paris."
  },
  {
    "user": "Explain quantum physics in simple terms.",
    "assistant": "Quantum physics is the study of matter and energy at the most fundamental level..."
  }
]
```

## 參數說明

- `--model_name_or_path`: 預訓練Mistral模型的路徑或名稱
- `--mode`: 訓練模式，可選"pretrain"或"sft"
- `--txt_files`: 本地TXT文件的通配符路徑（預訓練模式）
- `--hf_dataset`: Huggingface數據集名稱（預訓練模式）
- `--sft_json`: SFT模式下的對話JSON文件路徑
- `--batch_size`: 訓練批次大小
- `--epochs`: 訓練輪數
- `--learning_rate`: 學習率
- `--max_seq_length`: 序列最大長度
- `--activation_checkpointing`: 是否使用激活檢查點
- `--checkpointing_strategy`: 激活檢查點策略
- `--bf16`: 是否使用bf16精度訓練
- `--fp16`: 是否使用fp16精度訓練
- `--process_data`: 重新處理數據
- `--output_dir`: 輸出目錄

## 技術細節

1. 本項目通過移除Mistral模型中的因果掩碼，將其從自回歸模型轉換為掩碼預測器。
2. 僅針對注意力層進行微調，凍結了MLP參數，以減少訓練成本。
3. 支持梯度檢查點和混合精度訓練，以優化訓練效率和內存使用。
4. 支持預處理數據並緩存為npy文件，以加速訓練過程。 
