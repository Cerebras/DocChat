from transformers import AutoModel, AutoTokenizer
from dataset import get_data_for_evaluation
from arguments import get_args
from tqdm import tqdm
import torch
import os


def run_retrieval(eval_data, documents, query_encoder, context_encoder, tokenizer, max_seq_len=512):
    
    ranked_indices_list = []
    gold_index_list = []
    for doc_id in tqdm(eval_data):
        context_list = documents[doc_id]
        
        with torch.no_grad():
            # get chunk embeddings
            context_embs = []
            for chunk in context_list:
                chunk_ids = tokenizer(chunk, max_length=max_seq_len, truncation=True, return_tensors="pt").to("cuda")

                c_emb = context_encoder(input_ids=chunk_ids.input_ids, attention_mask=chunk_ids.attention_mask)
                c_emb = c_emb.last_hidden_state[:, 0, :]
                context_embs.append(c_emb)
            context_embs = torch.cat(context_embs, dim=0)   # (num_chunk, hidden_dim)
    
            sample_list = eval_data[doc_id]
            query_embs = []
            for item in sample_list:
                gold_idx = item['gold_idx']
                gold_index_list.append(gold_idx)

                query = item['query']
                query_ids = tokenizer(query, max_length=max_seq_len, truncation=True, return_tensors="pt").to("cuda")
                q_emb = query_encoder(input_ids=query_ids.input_ids, attention_mask=query_ids.attention_mask)
                q_emb = q_emb.last_hidden_state[:, 0, :]
                query_embs.append(q_emb)
            query_embs = torch.cat(query_embs, dim=0)   # (num_query, hidden_dim)

            similarities = query_embs.matmul(context_embs.transpose(0,1))     # (num_query, num_chunk)
            ranked_results = torch.argsort(similarities, dim=-1, descending=True)   # (num_query, num_chunk)
            ranked_indices_list.extend(ranked_results.tolist())

    return ranked_indices_list, gold_index_list


def calculate_recall(ranked_indices_list, gold_index_list, topk):
    hit = 0
    for ranked_indices, gold_index in zip(ranked_indices_list, gold_index_list):
        for idx in ranked_indices[:topk]:
            if idx == gold_index:
                hit += 1
                break
    recall = hit / len(ranked_indices_list)

    print("top-%d recall score: %.4f" % (topk, recall))


def main():
    args = get_args()

    ## get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.query_encoder_path)

    ## get retriever model
    query_encoder = AutoModel.from_pretrained(args.query_encoder_path)
    context_encoder = AutoModel.from_pretrained(args.context_encoder_path)
    query_encoder.to("cuda"), query_encoder.eval()
    context_encoder.to("cuda"), context_encoder.eval()

    ## get evaluation data
    if args.eval_dataset == "doc2dial":
        input_datapath = os.path.join(args.data_folder, args.doc2dial_datapath)
        input_docpath = os.path.join(args.data_folder, args.doc2dial_docpath)
    elif args.eval_dataset == "quac":
        input_datapath = os.path.join(args.data_folder, args.quac_datapath)
        input_docpath = os.path.join(args.data_folder, args.quac_docpath)
    elif args.eval_dataset == "qrecc":
        input_datapath = os.path.join(args.data_folder, args.qrecc_datapath)
        input_docpath = os.path.join(args.data_folder, args.qrecc_docpath)
    elif args.eval_dataset == "topiocqa" or args.eval_dataset == "inscit":
        raise Exception("We have prepare the function to get queries, but a wikipedia corpus needs to be downloaded")
    else:
        raise Exception("Please input a correct eval_dataset name!")

    eval_data, documents = get_data_for_evaluation(input_datapath, input_docpath, args.eval_dataset)
    
    ## run retrieval
    ranked_indices_list, gold_index_list = run_retrieval(eval_data, documents, query_encoder, context_encoder, tokenizer)
    print("number of the total test samples: %d" % len(ranked_indices_list))

    ## calculate recall scores
    print("evaluating on %s" % args.eval_dataset)
    topk_list = [1, 5, 20]
    for topk in topk_list:
        calculate_recall(ranked_indices_list, gold_index_list, topk=topk)


if __name__ == "__main__":
    main()
