"""
Test different prompt formats to see which works
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import torch

# Load model
base = GPT2LMHeadModel.from_pretrained('gpt2')
model = PeftModel.from_pretrained(base, 'models/fine_tuned_gpt2_v2')
tokenizer = GPT2Tokenizer.from_pretrained('models/fine_tuned_gpt2_v2')
tokenizer.pad_token = tokenizer.eos_token
model.eval()

test_question = "How to add interface mapping for Cost Center ENERGY572?"

print("="*60)
print("TESTING DIFFERENT PROMPT FORMATS")
print("="*60)

# Format 1: Question/Answer style (current)
print("\n1. Question/Answer Format:")
prompt1 = f"Question: {test_question}\n\nAnswer:"
inputs1 = tokenizer(prompt1, return_tensors='pt')
with torch.no_grad():
    outputs1 = model.generate(inputs1['input_ids'], max_length=200, temperature=0.5, do_sample=True, pad_token_id=tokenizer.eos_token_id)
result1 = tokenizer.decode(outputs1[0], skip_special_tokens=True)
print(f"Prompt: {prompt1}")
print(f"Output: {result1}")

# Format 2: Direct prompt
print("\n2. Direct Format:")
prompt2 = test_question
inputs2 = tokenizer(prompt2, return_tensors='pt')
with torch.no_grad():
    outputs2 = model.generate(inputs2['input_ids'], max_length=200, temperature=0.5, do_sample=True, pad_token_id=tokenizer.eos_token_id)
result2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
print(f"Prompt: {prompt2}")
print(f"Output: {result2}")

# Format 3: With context marker
print("\n3. Context Format:")
prompt3 = f"{test_question}\n\n"
inputs3 = tokenizer(prompt3, return_tensors='pt')
with torch.no_grad():
    outputs3 = model.generate(inputs3['input_ids'], max_length=200, temperature=0.5, do_sample=True, pad_token_id=tokenizer.eos_token_id)
result3 = tokenizer.decode(outputs3[0], skip_special_tokens=True)
print(f"Prompt: {prompt3}")
print(f"Output: {result3}")

print("\n" + "="*60)
print("Check which format produces best output")
print("="*60)