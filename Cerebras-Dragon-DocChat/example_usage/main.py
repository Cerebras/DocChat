# Demonstration of DocChat retriever in a multi-turn setting
# The sample documents are from a spec sheet about the Cerebras system & supercomputers
#
# Usage: python ./main.py

from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("cerebras/Dragon-DocChat-Query-Encoder")
query_encoder = AutoModel.from_pretrained("cerebras/Dragon-DocChat-Query-Encoder")
context_encoder = AutoModel.from_pretrained("cerebras/Dragon-DocChat-Context-Encoder")

documents = []
for file in ["sample_document_1.md", "sample_document_2.md"]:
    with open(file, "r") as f:
        documents.append(f.read())


query = [
    {"role": "user", "content": "How many cores does a WSE-3 have?"},
    {"role": "agent", "content": "WSE-3 has 900k cores."},
    {"role": "user", "content": "What is Condor Galaxy?"}
]

formatted_query = "\n".join([turn["role"] + ": " + turn["content"] for turn in query]).strip()

query_input = tokenizer(formatted_query, return_tensors='pt')
ctx_input = tokenizer(documents, padding=True, truncation=True, max_length=512, return_tensors='pt')
query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]
ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]

## Compute similarity scores:
similarities = query_emb.matmul(ctx_emb.transpose(0, 1)) # (1, num_ctx)

## Rank the similarity from highest to lowest
ranked_results = torch.argsort(similarities, dim=-1, descending=True) # (1, num_ctx)

for i, doc_idx in enumerate(ranked_results[0].tolist()):
    print(f"Rank {i}th document:")
    print("-" * 80)
    print(documents[doc_idx])
    print()