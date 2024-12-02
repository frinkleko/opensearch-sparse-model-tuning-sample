import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)   

from scripts.model.bigram_tokenizer import BiGramBertTokenizer
from scripts.model.decomposition_bert import *

# Load pre-trained BERT tokenizer
tokenizer = BiGramBertTokenizer.from_pretrained('bert-base-uncased', bi_gram_file='./bigrams_vocab.txt')

# Tokenize input
text = "This is a sample text 7 km."
encoded_input = tokenizer(text, return_tensors='pt')

# Initialize the custom DecompBertForMaskedLM model
config = DecompBertConfig(
    vocab_size=tokenizer.vocab_size,
    vocab_intermediate_size=256,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=512,
    type_vocab_size=2
)
model = DecompBertForMaskedLM(config)

# Mask a token in the input
masked_input = encoded_input.input_ids.clone()
mask_idx = torch.randint(1, masked_input.size(-1), (1,)).item()
masked_input[0, mask_idx] = tokenizer.mask_token_id

# Forward pass
outputs = model(masked_input)
logits = outputs.logits

# Get the predicted token for the masked position
masked_token_logits = logits[0, mask_idx, :]
predicted_token_id = torch.argmax(masked_token_logits).item()

print(predicted_token_id, tokenizer.convert_ids_to_tokens([predicted_token_id]))

print(predicted_token_id, tokenizer.convert_ids_to_tokens([predicted_token_id]))

predicted_token = tokenizer.convert_ids_to_tokens([predicted_token_id])
print(f"Predicted token for masked position: {predicted_token}")
# print correct token
correct_token = tokenizer.convert_ids_to_tokens([encoded_input.input_ids[0, mask_idx].item()])
print(f"Correct token: {correct_token}")

