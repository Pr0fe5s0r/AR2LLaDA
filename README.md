# Mistral转LLaDA模型训练

本项目实现了Mistral模型到LLaDA模型的转换和微调，参考了LLaDA论文中的方法。

LLaDA（Language Model as a Denoising Autoencoder）是一种将自回归语言模型转换为掩码预测器的架构，主要修改了Transformer架构中的自注意力机制，移除了因果掩码。

## 项目结构

- `llada_loss.py`: LLaDA的核心损失函数实现
- `mistral_to_llada.py`: Mistral模型到LLaDA模型的转换工具
- `data_processor.py`: 数据处理工具，支持将文本数据转换为npy文件
- `train.py`: 训练脚本，支持预训练和SFT两种模式
- `configuration_llada.py`: LLaDA模型配置（从项目中已有）
- `modeling_llada.py`: LLaDA模型定义（从项目中已有）

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- 其他依赖项可通过`pip install -r requirements.txt`安装

## 使用方法

### 数据预处理

本项目支持从本地TXT文件或Huggingface数据集加载数据，并将其处理为npy格式以加速训练：

```bash
# 预处理本地TXT文件
python train.py \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --txt_files "data/*.txt" \
  --process_data \
  --mode pretrain \
  --output_dir ./output

# 预处理Huggingface数据集
python train.py \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --hf_dataset "wikitext/wikitext-2-v1" \
  --hf_text_column "text" \
  --process_data \
  --mode pretrain \
  --output_dir ./output

# 创建并使用示例数据进行快速测试
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

### 预训练

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

### SFT（监督微调）

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

## SFT数据格式

SFT数据应为JSON格式，包含用户和助手对话：

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

## 参数说明

- `--model_name_or_path`: 预训练Mistral模型的路径或名称
- `--mode`: 训练模式，可选"pretrain"或"sft"
- `--txt_files`: 本地TXT文件的通配符路径（预训练模式）
- `--hf_dataset`: Huggingface数据集名称（预训练模式）
- `--sft_json`: SFT模式下的对话JSON文件路径
- `--batch_size`: 训练批次大小
- `--epochs`: 训练轮数
- `--learning_rate`: 学习率
- `--max_seq_length`: 序列最大长度
- `--activation_checkpointing`: 是否使用激活检查点
- `--checkpointing_strategy`: 激活检查点策略
- `--bf16`: 是否使用bf16精度训练
- `--fp16`: 是否使用fp16精度训练
- `--process_data`: 重新处理数据
- `--output_dir`: 输出目录

## 技术细节

1. 本项目通过移除Mistral模型中的因果掩码，将其从自回归模型转换为掩码预测器。
2. 仅针对注意力层进行微调，冻结了MLP参数，以减少训练成本。
3. 支持梯度检查点和混合精度训练，以优化训练效率和内存使用。
4. 支持预处理数据并缓存为npy文件，以加速训练过程。 