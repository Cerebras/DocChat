# Demonstration of DocChat in a multi-turn setting
# The sample document is a spec sheet about the Cerebras system & supercomputers
# The first question is provided as an example. You may enter subsequent
# questions via console
#
# Usage: python ./main.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "cerebras/Llama3-DocChat-1.0-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")


system = "This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
instruction = "Please give a full and complete answer for the question."

with open("sample_document.md", "r") as f:
    document = f.read()

# We supply the first question as an example for the user. Subsequent questions can be inputted in the console
question = "How many total CS systems does Condor Galaxy 1, 2, and 3 have combined, and how many flops does this correspond to?"
# Model will respond with: Condor Galaxy 1, 2, and 3 have a total of 192 CS systems, which corresponds to 16 ExaFLOPs.

user_turn = f"""<context>
{document}
</context>
{instruction} {question}"""

messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": user_turn}
]
print(f"> {messages[-1]['content']}")

while True:
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
    )
    response = outputs[0][input_ids.shape[-1]:]
    decoded_response = tokenizer.decode(response, skip_special_tokens=True)

    print(decoded_response)

    messages.append({"role": "assistant", "content": decoded_response})

    # Allow the user to provide the next question!
    print("> ", end="")
    question = input().strip()
    if question == "" or question == "exit":
        break

    messages.append({"role": "user", "content": question})
