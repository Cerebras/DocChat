import json


def get_query(messages, num_turns=5):
    ## convert query into a format as follows:
    ## user: {user}\nagent: {agent}\nuser: {user}
    query = ""
    for item in messages[-num_turns:]:
        item['role'] = item['role'].replace("assistant", "agent")
        query += "{}: {}\n".format(item['role'], item['content'])
    query = query.strip()
    
    return query


def get_query_with_topic(messages, topic, num_turns=3):
    ## convert query into a format as follows:
    ## user: this is a question about {topic}. {user}\nagent: {agent}\nuser: this is a question about {topic}. {user}
    query = ""
    for item in messages[-num_turns:]:
        item['role'] = item['role'].replace("assistant", "agent")
        if item['role'] == 'user':
            query += "{}: this is a question about {}. {}\n".format(item['role'], topic, item['content'])
        else:
            query += "{}: {}\n".format(item['role'], item['content'])
    query = query.strip()

    return query


def get_data_for_evaluation(input_datapath, document_datapath, dataset_name):

    print('reading evaluation data from %s' % input_datapath)
    with open(input_datapath, "r") as f:
        input_list = json.load(f)
    
    print('reading documents from %s' % document_datapath)
    with open(document_datapath, "r") as f:
        documents = json.load(f)

    eval_data = {}
    for item in input_list:
        """
        We incorporate topic information for topiocqa and inscit datasets:
        query = get_query_with_topic(item['messages'], item['topic'])
        """
        query = get_query(item['messages'])

        doc_id = item['document']
        gold_idx = item['ground_truth_ctx']['index']

        if dataset_name == 'qrecc':
            """
            The 'gold context' for the qrecc dataset is obtained based on the word
            overlaps between gold answer and each context in the document, which might
            not be the real gold context.
            To improve the evaluation quality of this dataset,
            we further add the answer of the query into the 'gold context' 
            to ensure the 'gold context' is the most relevant chunk to the query.
            
            Note that this is just for the retrieval evaluation purpose, we do not
            add answer to the context for the ChatRAG evaluation.
            """
            answer = item['answers'][0]
            documents[doc_id][gold_idx] += " || "  + answer
        
        if doc_id not in eval_data:
            eval_data[doc_id] = [{"query": query, "gold_idx": gold_idx}]
        else:
            eval_data[doc_id].append({"query": query, "gold_idx": gold_idx})

    return eval_data, documents
