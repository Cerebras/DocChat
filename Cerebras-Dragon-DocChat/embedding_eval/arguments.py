import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Dragon-multiturn")

    parser.add_argument('--query-encoder-path', type=str, default='nvidia/dragon-multiturn-query-encoder')
    parser.add_argument('--context-encoder-path', type=str, default='nvidia/dragon-multiturn-context-encoder')

    parser.add_argument('--data-folder', type=str, default='../../Nvidia-ChatQA/ChatRAG-Bench/data/', help='path to the datafolder of ChatRAG Bench')
    parser.add_argument('--eval-dataset', type=str, default='', help='evaluation dataset (e.g., doc2dial)')

    parser.add_argument('--doc2dial-datapath', type=str, default='doc2dial/test.json')
    parser.add_argument('--doc2dial-docpath', type=str, default='doc2dial/documents.json')

    parser.add_argument('--quac-datapath', type=str, default='quac/test.json')
    parser.add_argument('--quac-docpath', type=str, default='quac/documents.json')
    
    parser.add_argument('--qrecc-datapath', type=str, default='qrecc/test.json')
    parser.add_argument('--qrecc-docpath', type=str, default='qrecc/documents.json')
    
    parser.add_argument('--topiocqa-datapath', type=str, default='topiocqa/dev.json')
    parser.add_argument('--topiocqa-docpath', type=str, default='')
    
    parser.add_argument('--inscit-datapath', type=str, default='inscit/dev.json')
    parser.add_argument('--inscit-docpath', type=str, default='')

    args = parser.parse_args()

    return args