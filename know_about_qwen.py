from transformers import AutoModelForCausalLM; 

def freeze_all_but_attention(model):
    for name, param in model.named_parameters():
        # Only attention layers remain trainable
        if (
            ".self_attn.q_proj." in name or
            ".self_attn.k_proj." in name or
            ".self_attn.v_proj." in name or
            ".self_attn.o_proj." in name
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False


model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B'); 
from pprint import pprint
pprint([name for name, _ in model.named_parameters()])


# Freeze all but attention
freeze_all_but_attention(model)

# Print trainable parameters
print("Trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total/1e9:.3f}B")
print(f"Trainable parameters: {trainable/1e6:.2f}M")
print(f"Total Tokens needed to train Qwen2.5-0.5B: {trainable/1e6*20:.2f}M")
print(f"Percentage trainable: {100 * trainable / total:.2f}%")