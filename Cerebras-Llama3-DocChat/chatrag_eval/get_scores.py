
from evaluation_utils import quac_correct_retrieved_instance_idx_list
from evaluation_utils import unanswerable_keyphrases
import json
from metrics import F1Metric
import copy
import re
import argparse
import os

dataset_paths = {
    "convfinqa": "convfinqa/dev.json",
    "coqa": "coqa/dev.json",
    "doc2dial": "doc2dial/test.json",
    "doqa_cooking": "doqa/test_cooking.json",
    "doqa_movies": "doqa/test_movies.json",
    "doqa_travel": "doqa/test_travel.json",
    "hybridial": "hybridial/test.json",
    "inscit": "inscit/dev.json",
    "qrecc": "qrecc/test.json",
    "quac": "quac/test.json",
    "sqa": "sqa/test.json",
    "topiocqa": "topiocqa/dev.json",
}


def compute_f1_score(predicted_answers, groundtruth_answer, exp_name="default"):
    """Evaluating F1 Score"""
    print(len(predicted_answers), len(groundtruth_answer))
    if len(predicted_answers) != len(groundtruth_answer):
        groundtruth_answer = groundtruth_answer[:len(predicted_answers)]

    guess_list = []
    for guess in predicted_answers:
        guess = guess.strip()
        if "</s>" in guess:
            guess = guess.replace("</s>", "")
        guess_list.append(guess)

    answer_list = []
    for answer in groundtruth_answer:
        answer_list.append(answer)

    assert len(guess_list) == len(answer_list), \
        "lengths of guess and answer are different!"

    precision, recall, f1, individual_f1 = F1Metric.compute_all_pairs(guess_list, answer_list)
    print('Method: %s; Precision: %.4f; recall: %.4f; f1: %.4f' % (\
        exp_name, precision, recall, f1))

    return individual_f1


def load_groundtruth_file(data_file):
    
    with open(data_file, "r") as f:
        examples = json.load(f)

    data = []
    for instance in examples:
        if "answers" in instance:
            answers = instance["answers"]
        elif "answer" in instance:
            if type(instance["answer"]) is str:
                answers = [instance["answer"]]
            elif type(instance["answer"]) is list:
                answers = instance["answer"]
            else:
                answers = [str(instance["answer"])]
        else:
            raise ValueError("need to have answer or answers")
        data.append(answers)

    return data


def load_prediction(data_file):

    data = []
    with open(data_file, "r") as f:
        for line in f.readlines():
            data.append(line.strip())

    return data


def evaluate_f1(ground_truth_file, prediction_file):

    groundtruth_answers = load_groundtruth_file(ground_truth_file)
    with open(ground_truth_file, "r") as f:
        samples = json.load(f)
        messages = [sample["messages"] for sample in samples]
        contexts = [sample["ctxs"] for sample in samples]

    if "inscit" in ground_truth_file:
        groundtruth_answers_update = []
        for answers in groundtruth_answers:
            answers_update = []
            for ans in answers:
                ## this answer is additionally added to the answer_list for inscit dataset, needs to remove
                if ans != "Sorry. I cannot find the answer based on the context.":
                    answers_update.append(ans)
            assert len(answers_update) > 0
            groundtruth_answers_update.append(copy.deepcopy(answers_update))
        groundtruth_answers = groundtruth_answers_update

    predicted_answers = load_prediction(prediction_file)
    predicted_answers_original = copy.deepcopy(predicted_answers)
    if "quac" in prediction_file or "doqa" in prediction_file:
        predicted_answers_new = []
        for pred in predicted_answers:
            pred = pred.lower()
            for keyphrase in unanswerable_keyphrases:
                if keyphrase in pred:
                    pred = "Sorry. I cannot find the answer based on the context."
                    break
            predicted_answers_new.append(pred)
        predicted_answers = predicted_answers_new

    individual_f1 = compute_f1_score(predicted_answers, groundtruth_answers)

    err_file = os.path.splitext(prediction_file)[0] + "_errors.jsonl"
    with open(err_file, "w") as f:
        for i in range(len(predicted_answers)):
            f.write(json.dumps({
                "contexts": contexts[i],
                "messages": messages[i],
                "pred": predicted_answers[i],
                "pred_original": predicted_answers_original[i],
                "gold": groundtruth_answers[i],
                "f1": individual_f1.get(i, None)
            }) + "\n")




def separate_cannot_answer(ground_truth_file, prediction_file):
    # load ground truth
    with open(ground_truth_file, "r") as f:
        groundtruth_answers = json.load(f)
    # load prediction
    predicted_answers = load_prediction(prediction_file)
    print(len(predicted_answers), len(groundtruth_answers))
    if len(predicted_answers) != len(groundtruth_answers):
        groundtruth_answers = groundtruth_answers[:len(predicted_answers)]

    if "quac" in prediction_file:
        """
        For answerable cases, we want to make sure the retrieved context list contains the gold chunk.
        For QuAC dataset, we use top-5 retrieved contexts as inputs, quac_correct_retrieved_instance_idx_list 
        is the index list where the top-5 retrieved context contains the gold answer
        """
        answerable_instance_idx_list = quac_correct_retrieved_instance_idx_list
    else:
        answerable_instance_idx_list = None

    predicted_answers_new = []
    for pred in predicted_answers:
        pred = pred.lower()
        for keyphrase in unanswerable_keyphrases:
            if keyphrase in pred:
                pred = "Sorry. I cannot find the answer based on the context."
                break
        predicted_answers_new.append(pred)
    predicted_answers = predicted_answers_new

    cannot_answer_idx_list = []
    answerable_idx_list = []
    if answerable_instance_idx_list:
        count_idx = 0
    for idx, item in enumerate(groundtruth_answers):
        if 'answers' in item:
            answer = item["answers"][0]
        else:
            answer = item['answer']
        noanswer_response = "Sorry. I cannot find the answer based on the context."

        if answer == noanswer_response:
            cannot_answer_idx_list.append(idx)
            continue
        
        if answerable_instance_idx_list:
            if count_idx in answerable_instance_idx_list:
                answerable_idx_list.append(idx)
            count_idx += 1
        else:
            answerable_idx_list.append(idx)

    print("number of cannot answer cases: %d (out of %d)" % (len(cannot_answer_idx_list), len(groundtruth_answers)))
    print("number of answerable cases: %d (out of %d)" % (len(answerable_idx_list), len(groundtruth_answers)))

    return predicted_answers, cannot_answer_idx_list, answerable_idx_list


def get_cannot_answer_and_answerable_acc(predicted_answers, cannot_answer_idx_list, answerable_idx_list):
    # cannot answer
    noanswer_count = 0
    for idx in cannot_answer_idx_list:
        prediction = predicted_answers[idx]
        prediction = prediction.lower()
        # print(prediction)
        if "sorry" in prediction and "cannot find the answer" in prediction:
            # print(prediction)
            noanswer_count += 1
    cannot_answer_acc = noanswer_count / len(cannot_answer_idx_list)
    print("accuracy of cannot answer cases: %.4f" % cannot_answer_acc)

    # answerable
    answerable_count = 0
    for idx in answerable_idx_list:
        prediction = predicted_answers[idx]
        prediction = prediction.lower()
        if "sorry" in prediction and "cannot find the answer" in prediction:
            # print(prediction)
            continue
        answerable_count += 1
    answerable_acc = answerable_count / len(answerable_idx_list)
    print("accuracy of answerable cases: %.4f" % answerable_acc)




def evaluate_cannot_answer_acc(ground_truth_file, prediction_file):
    predicted_answers, cannot_answer_idx_list, answerable_idx_list = \
                                separate_cannot_answer(ground_truth_file, prediction_file)

    get_cannot_answer_and_answerable_acc(predicted_answers, cannot_answer_idx_list, answerable_idx_list)
    print("F1 for *only* answerable:")
    groundtruth_answers = load_groundtruth_file(ground_truth_file)
    compute_f1_score([predicted_answers[idx] for idx in answerable_idx_list], [groundtruth_answers[idx] for idx in answerable_idx_list])



def evaluate_convfinqa(ground_truth_file, prediction_file):
    """
    Since the model will give a long answer output, while the gold answer for ConvFinQA are either 
    a arithmetic formula or a final executed number.
    We consider the output containing either the executed number or the arithmetic formula as correct.
    This script is to measure the proportion of the outputs containing these elements.
    """

    def _is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    with open(ground_truth_file, "r") as f:
        gold_list = json.load(f)
    
    groundtruth_answers = [item['exe_answer'] for item in gold_list]
    groundtruth_answers_formula = [item['answers'][0] for item in gold_list]

    ## last turn question_list
    question_list = [item['messages'][-1]['content'] for item in gold_list]
    predicted_answers = load_prediction(prediction_file)

    print(len(predicted_answers), len(groundtruth_answers))
    if len(predicted_answers) != len(groundtruth_answers):
        groundtruth_answers = groundtruth_answers[:len(predicted_answers)]

    err_file = os.path.splitext(prediction_file)[0] + "_errors.jsonl"
    with open(err_file, "w") as f:

        count_exact_match = 0
        for question, pred, gold, gold_formula in zip(question_list, predicted_answers, groundtruth_answers, groundtruth_answers_formula):

            original_pred = pred
            ## convert 1,000,000 into 1000000
            original_pred = original_pred.replace(",", "")

            ## convert $10 million + $20 million into 10 + 20
            original_pred = original_pred.replace("$", "").replace("million", "").replace("billion", "")

            ## convert 10 (2017) + 20 (2018) into 10 + 20
            pattern = r'\((\b\w+\b)\)'
            original_pred = re.sub(pattern, '', original_pred)

            ## make sure it each token only has one space in between
            original_pred = " ".join(original_pred.split())
            
            if str(gold) in original_pred:
                count_exact_match += 1
            
            elif str(gold_formula) in original_pred:
                count_exact_match += 1
            
            elif _is_float(gold) and (str(round(float(gold), 3)) in original_pred or str(round(float(gold), 2)) in original_pred):
                count_exact_match += 1
            
            elif "percent" in question and (str(float(gold)*100) in original_pred or str(round(float(gold)*100, 1)) in original_pred or str(round(float(gold)*100, 2)) in original_pred):
                count_exact_match += 1
            
            elif str(gold).endswith(".0") and str(int(gold)) in original_pred:
                ## gold is a integer like 80.0 then convert it into 80
                count_exact_match += 1
            
            elif "decrease" in original_pred and _is_float(gold) and gold < 0 and (str(-1 * gold) in original_pred):
                ## for the case where model generates something like a decrese of 10 million, while gold is -10.
                count_exact_match += 1
            else:
                f.write(json.dumps({
                    "question": question,
                    "pred": pred,
                    "gold": gold,
                    "gold_formula": gold_formula
                }) + "\n")

    print("accuracy of exact match: %.4f" % (count_exact_match/len(predicted_answers)))

def print_dataset_header(dataset_name):
    header = f"[Evaluating {dataset_name}]"
    num_dashes = (80 - len(header)) // 2
    print("-" * num_dashes + header + "-" * num_dashes)



def main():
    parser = argparse.ArgumentParser(description='Get Scores')
    parser.add_argument(
        '--eval-dataset',
        type=str,
        required=True,
        choices=dataset_paths.keys(),
        help='Eval dataset name',
    )

    parser.add_argument(
        '--prediction-file',
        type=str,
        required=True,
        help='Prediction File',
    )

    parser.add_argument(
        '--data-folder',
        type=str,
        required=True,
        help='Ground Truth File',
    )

    args = parser.parse_args()

    dataset = args.eval_dataset
    prediction_file = args.prediction_file
    ground_truth_file = os.path.join(args.data_folder, dataset_paths[dataset])

    print_dataset_header(dataset)

    ## doc2dial
    if dataset in ["doc2dial", "qrecc", "topiocqa", "inscit", "coqa", "hybridial", "sqa"]:
        print(prediction_file)
        print(ground_truth_file)
        evaluate_f1(ground_truth_file, prediction_file)
    elif dataset in ["quac", "doqa_cooking", "doqa_travel", "doqa_movies"]:
        print(prediction_file)
        print(ground_truth_file)
        evaluate_f1(ground_truth_file, prediction_file)
        evaluate_cannot_answer_acc(ground_truth_file, prediction_file)
    elif dataset == "convfinqa":
        print(prediction_file)
        print(ground_truth_file)
        evaluate_convfinqa(ground_truth_file, prediction_file)
    else:
        raise RuntimeError("Invalid dataset name:", dataset)


if __name__ == "__main__":
    main()

