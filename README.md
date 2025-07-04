# Cell 1: Install and import
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model and tokenizer loaded successfully.")
