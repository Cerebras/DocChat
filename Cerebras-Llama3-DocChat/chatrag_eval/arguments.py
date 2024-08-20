
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="ChatQA-HF")

    ## model
    parser.add_argument('--model-id', type=str, default='', help='model id')

    ## dataset path
    parser.add_argument('--data-folder', type=str, default='', help='path to the datafolder of ConvRAG')
    parser.add_argument('--output-folder', type=str, default='', help='path to the datafolder of ConvRAG')
    parser.add_argument('--eval-dataset', type=str, default='')
    parser.add_argument('--doc2dial-path', type=str, default='doc2dial/test.json')
    parser.add_argument('--convfinqa-path', type=str, default='convfinqa/dev.json')
    parser.add_argument('--quac-path', type=str, default='quac/test.json')
    parser.add_argument('--qrecc-path', type=str, default='qrecc/test.json')
    parser.add_argument('--doqa-cooking-path', type=str, default='doqa/test_cooking.json')
    parser.add_argument('--doqa-travel-path', type=str, default='doqa/test_travel.json')
    parser.add_argument('--doqa-movies-path', type=str, default='doqa/test_movies.json')
    parser.add_argument('--coqa-path', type=str, default='coqa/dev.json')
    parser.add_argument('--hybridial-path', type=str, default='hybridial/test.json')
    parser.add_argument('--sqa-path', type=str, default='sqa/test.json')
    parser.add_argument('--topiocqa-path', type=str, default='topiocqa/dev.json')
    parser.add_argument('--inscit-path', type=str, default='inscit/dev.json')

    ## others
    parser.add_argument('--out-seq-len', type=int, default=64)
    parser.add_argument('--num-ctx', type=int, default=5)
    parser.add_argument('--max-tokens', type=int, default=64)

    args = parser.parse_args()

    return args
