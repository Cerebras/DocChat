
import json


def load_data(datapath):
    print("loading data from %s" % datapath)
    with open(datapath, "r") as f:
        data_list = json.load(f)

    return data_list

def apply_chat_template(tokenizer, messages, strip_bos=False):
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if strip_bos:
        assert formatted.startswith("<|begin_of_text|>")
        formatted = formatted[len("<|begin_of_text|>"):]
    return formatted 

def reformat_question(turn_list, dataset_name):

    ## only take the lastest 7 turns
    turn_list = turn_list[-7:]
    assert turn_list[-1]['role'] == 'user'

    long_answer_dataset_list = ["doc2dial", "quac", "qrecc", "inscit", "doqa_movies", "doqa_travel", "doqa_cooking", "hybridial"]
    cot_answer_dataset_list = ["convfinqa"]
    long_and_short_dataset_list = ["topiocqa"]
    entity_dataset_list = ["sqa"]
    short_dataset_list = ["coqa"]

    if dataset_name in long_answer_dataset_list:
        for item in turn_list:
            if item['role'] == 'user':
                ## only needs to add it on the first user turn
                item['content'] = 'Please give a full and complete answer for the question. ' + item['content']
                break
    elif dataset_name in cot_answer_dataset_list:
        for item in turn_list:
            if item['role'] == 'user':
                ## only needs to add it on the first user turn
                item['content'] = 'Answer the following question using the context by thinking step by step to produce a final number or math arithmetic using +, -, *, or /. ' + item['content']
                break
    
    elif dataset_name in long_and_short_dataset_list:
        turn_list[-1]['content'] = "Answer the following question with a short span, or a full and complete answer. " + turn_list[-1]['content']

    elif dataset_name in entity_dataset_list:
        turn_list[-1]['content'] = "Answer the following question with one or a list of items. " + turn_list[-1]['content']

    elif dataset_name in short_dataset_list:
        turn_list[-1]['content'] = "Answer the following question with a short span. The answer needs to be just in a few words. " + turn_list[-1]['content']

    else:
        raise Exception("please input a correct dataset name!")

    return turn_list


def get_inputs(data_list, dataset_name, tokenizer, num_ctx, max_output_len, max_seq_length=4096):

    system = "This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."

    prompt_list = []
    context_seperator_len = len(tokenizer.encode("\n</context>\n"))

    for item in data_list:
        turn_list = item['messages']
        turns = reformat_question(turn_list, dataset_name)
        turns = [{
            "role": "system",
            "content": system
        }] + turns
        
        formatted = apply_chat_template(tokenizer, turns)

        ctx_list = ["title: " + ctx["title"] + ", source: " + ctx["text"] for ctx in item['ctxs'][:num_ctx]]
        context = "\n\n".join(ctx_list)
        if not context.endswith("\n"):
            context += "\n"
        context = "<context>\n" + context + "</context>\n"

        context_tokens = tokenizer.encode(context)
        system_and_questions = tokenizer.encode(formatted)

        if len(context_tokens) + len(system_and_questions) + max_output_len >= max_seq_length:
            context_tokens = context_tokens[:max_seq_length - max_output_len - len(system_and_questions) - context_seperator_len]
            context = tokenizer.decode(context_tokens, skip_special_tokens=True)
            context += "\n</context>\n"

        # Add context to first user turn (which may not be turns[1])
        for i in range(len(turns)):
            if turns[i]["role"] == "user":
                turns[i]["content"] = context + turns[i]["content"]
                break
        else:
            assert False, "No user role found in messages!"

        model_input = apply_chat_template(tokenizer, turns)

        prompt_list.append(model_input)
    
    return prompt_list

