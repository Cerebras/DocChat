

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from arguments import get_args
from dataset import load_data, get_inputs
import torch
import os
from tqdm import tqdm

def get_prompt_list(args):

    ## get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    ## get input data
    if args.eval_dataset == "doc2dial":
        input_datapath = os.path.join(args.data_folder, args.doc2dial_path)
    elif args.eval_dataset == "convfinqa":
        input_datapath = os.path.join(args.data_folder, args.convfinqa_path)
    elif args.eval_dataset == "quac":
        input_datapath = os.path.join(args.data_folder, args.quac_path)
    elif args.eval_dataset == "qrecc":
        input_datapath = os.path.join(args.data_folder, args.qrecc_path)
    elif args.eval_dataset == "doqa_cooking":
        input_datapath = os.path.join(args.data_folder, args.doqa_cooking_path)
    elif args.eval_dataset == "doqa_travel":
        input_datapath = os.path.join(args.data_folder, args.doqa_travel_path)
    elif args.eval_dataset == "doqa_movies":
        input_datapath = os.path.join(args.data_folder, args.doqa_movies_path)
    elif args.eval_dataset == "coqa":
        input_datapath = os.path.join(args.data_folder, args.coqa_path)
    elif args.eval_dataset == "sqa":
        input_datapath = os.path.join(args.data_folder, args.sqa_path)
    elif args.eval_dataset == "topiocqa":
        input_datapath = os.path.join(args.data_folder, args.topiocqa_path)
    elif args.eval_dataset == "inscit":
        input_datapath = os.path.join(args.data_folder, args.inscit_path)
    elif args.eval_dataset == "hybridial":
        input_datapath = os.path.join(args.data_folder, args.hybridial_path)

    else:
        raise Exception("please input a correct eval_dataset name!")
    
    data_list = load_data(input_datapath)
    print("number of samples in the dataset:", len(data_list))
    prompt_list = get_inputs(data_list, args.eval_dataset, tokenizer, num_ctx=args.num_ctx, max_output_len=args.out_seq_len)

    return prompt_list


def main():
    args = get_args()

    ## get prompt_list
    prompt_list = get_prompt_list(args)
    
    ## get output_datapath
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    output_datapath = os.path.join(args.output_folder, "%s_output.txt" % args.eval_dataset)

    ## run inference
    sampling_params = SamplingParams(temperature=0, top_k=1, max_tokens=args.max_tokens)

    ## This changes the GPU support to 8
    model_vllm = LLM(args.model_id, tensor_parallel_size=1)

    output_list = []
    for prompt in tqdm(prompt_list):
        # prompt = bos_token + prompt
        output = model_vllm.generate([prompt], sampling_params, use_tqdm=False)[0]
        generated_text = output.outputs[0].text
        generated_text = generated_text.strip().replace("\n", " ")
        
        # print("generated_text:", generated_text)
        output_list.append(generated_text)

    print("writing to %s" % output_datapath)
    with open(output_datapath, "w") as f:
        for output in output_list:
            f.write(output + "\n")


if __name__ == "__main__":
    main()
