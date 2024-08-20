# Cerebras DocChat: GPT-4 Level Conversational QA Trained In a Few Hours

This repository contains dataset preparation, training, and evaluation code for DocChat. The corresponding code for the LLM can be found in the `Cerebras-Llama3-DocChat` while code for the retriever is located at `Cerebras-Dragon-DocChat`.

The dataset preparation & training scripts are designed to be used with [Cerebras Model Zoo](https://github.com/Cerebras/modelzoo) Release 2.3. Note that although the training scripts _can_ be run on non-Cerebras hardware, they haven't been optimized for other hardware, and so your mileage may vary. The evaluation code does not depend on Model Zoo and is hardware agnostic (supports CPU & GPU) in order to ensure that everyone can evaluate our model (including those who don't have access to Cerebras machines).

Additional Links:
* [Blog post](https://www.cerebras.net/blog/train-a-gpt-4-level-conversational-qa-in-a-few-hours)
* [LLM model weights on HuggingFace](https://huggingface.co/cerebras/Llama3-DocChat-1.0-8B)
* Embedding model weights on HuggingFace: [Query Encoder](https://huggingface.co/cerebras/Dragon-DocChat-Query-Encoder), [Context Encoder](https://huggingface.co/cerebras/Dragon-DocChat-Context-Encoder)
* [Data preparation, training, and evaluation code](https://github.com/Cerebras/DocChat)

## About DocChat

We are excited to announce the release of Cerebras DocChat, our first iteration of models designed for document-based conversational question answering. This series includes two models: Cerebras Llama3-DocChat, a large language model (LLM), and Cerebras Dragon-DocChat, a multi-turn retriever model. These models were not only developed leveraging our deep ML expertise, but were also trained with remarkable speed. Using a single Cerebras System, Llama3-DocChat was trained in a few hours, while Cerebras Dragon-DocChat was fine-tuned in just *minutes* (yes you read that correctly).

**DocChat LLM**

Cerebras Llama3-DocChat was built on top of Llama 3 base using insights from the latest research on document-based Q&A, most notably Nvidia’s ChatQA model series. As part of this work, we leveraged our experience in LLM model training and dataset curation to overcome the gaps in ChatQA's released datasets and training recipes. Additionally, we employed synthetic data generation to address limitations that couldn't be fully resolved with the available real data.

**DocChat Retriever**

Similarly, Cerebras Dragon-DocChat was built on top of the Dragon+ model and trained on ChatQA’s conversational Q&A dataset. By finetuning using contrastive loss with hard negatives, we see absolute improvements in recall of 8.9% over Dragon+ and 3.5% over ChatQA Dragon-Multiturn respectively (top-1).

**Open Source Commitment**

In line with our commitment to open source, we are releasing not only the model weights but also the complete training recipes and associated datasets. This transparency allows the AI community to replicate, build upon, and innovate with our work. See below for links.

**Benchmarks**

The DocChat models have been evaluated across a variety of benchmarks, and achieve top of the line performance for their model sizes.



| **ChatRAG Benchmark** | **Llama3 Instruct 8B** | **Command-R-Plus** | **Nvidia Llama3-ChatQA 1.5 8B** | **GPT-4-Turbo-2024-04-09** | **Cerebras Llama3-DocChat 1.0 8B** |
| --- | --- | --- | --- | --- | --- |
| Doc2Dial | 31.33 | 33.51 | 39.33 | 35.35 | 39.19 |
| QuAC | 32.64 | 34.16 | 39.73 | 40.1 | 36 |
| QReCC | 43.4 | 49.77 | 49.03 | 51.46 | 50.27 |
| CoQA | 73.25 | 69.71 | 76.46 | 77.73 | 79.56 |
| DoQA | 30.34 | 40.67 | 49.6 | 41.6 | 48.77 |
| ConvFinQA | 53.15 | 71.21 | 78.46 | 84.16 | 80.13 |
| SQA | 36.6 | 74.07 | 73.28 | 79.98 | 74.19 |
| TopioCQA | 34.64 | 53.77 | 49.96 | 48.32 | 52.13 |
| HybriDial\* | 40.77 | 46.7 | 65.76 | 47.86 | 64 |
| INSCIT | 32.09 | 35.76 | 30.1 | 33.75 | 32.88 |
| Average (all) | 40.82 | 50.93 | 55.17 | 54.03 | 55.71 |
| Average (exclude HybriDial) | 40.83 | 51.4 | 53.99 | 54.72 | 54.79 |



| **Eleuther Eval Harness**  | **Llama3 Instruct 8B** | **Nvidia Llama3-ChatQA 1.5 8B** | **Cerebras Llama3-DocChat 1.0 8B** |
| --- | --- | --- | --- |
| hellaswag | 57.68 | 61.37 | 61.68 |
| winogrande | 71.98 | 73.95 | 74.11 |
| truthfulqa_mc1 | 36.23 | 28.52 | 29.25 |
| truthfulqa_mc2 | 51.65 | 43.56 | 45.14 |
| mmlu | 63.84 | 60.68 | 62.86 |
| gsm8k | 76.12 | 13.72 | 55.57 |
| arc_easy | 81.61 | 80.56 | 82.03 |
| arc_challenge | 52.99 | 51.02 | 53.92 |
| Average | 61.51 | 51.67 | 58.07 |


<table id="retriever-benchmarks">
   <tbody>
      <tr>
         <td rowspan="1" colspan="1">
            <div data-align="center">
               <strong data-renderer-mark="true">Benchmark</strong>
            </div>
         </td>
         <td rowspan="1" colspan="1">
            <strong data-renderer-mark="true">Metric</strong>
         </td>
         <td rowspan="1" colspan="1">
            <strong data-renderer-mark="true">Facebook Dragon+</strong>
         </td>
         <td rowspan="1" colspan="1">
            <strong data-renderer-mark="true">Nvidia Dragon-Multiturn</strong>
         </td>
         <td rowspan="1" colspan="1">
            <strong data-renderer-mark="true">Cerebras Dragon-DocChat</strong>
         </td>
      </tr>
      <tr>
         <td rowspan="3" colspan="1">
            Doc2Dial
         </td>
         <td rowspan="1" colspan="1">
            Recall@1
         </td>
         <td rowspan="1" colspan="1">
            43.95
         </td>
         <td rowspan="1" colspan="1">
            50.11
         </td>
         <td rowspan="1" colspan="1">
            51.54
         </td>
      </tr>
      <tr>
         <td rowspan="1" colspan="1">
            Recall@5
         </td>
         <td rowspan="1" colspan="1">
            77.61
         </td>
         <td rowspan="1" colspan="1">
            83.85
         </td>
         <td rowspan="1" colspan="1">
            83.12
         </td>
      </tr>
      <tr>
         <td rowspan="1" colspan="1">
            Recall@20
         </td>
         <td rowspan="1" colspan="1">
            92.05
         </td>
         <td rowspan="1" colspan="1">
            95.33
         </td>
         <td rowspan="1" colspan="1">
            95.25
         </td>
      </tr>
      <tr>
         <td rowspan="3" colspan="1">
            QuAC
         </td>
         <td rowspan="1" colspan="1">
            Recall@1
         </td>
         <td rowspan="1" colspan="1">
            62.09
         </td>
         <td rowspan="1" colspan="1">
            60.02
         </td>
         <td rowspan="1" colspan="1">
            61.30
         </td>
      </tr>
      <tr>
         <td rowspan="1" colspan="1">
            Recall@5
         </td>
         <td rowspan="1" colspan="1">
            86.01
         </td>
         <td rowspan="1" colspan="1">
            86.51
         </td>
         <td rowspan="1" colspan="1">
            87.69
         </td>
      </tr>
      <tr>
         <td rowspan="1" colspan="1">
            Recall@20
         </td>
         <td rowspan="1" colspan="1">
            96.48
         </td>
         <td rowspan="1" colspan="1">
            96.60
         </td>
         <td rowspan="1" colspan="1">
            97.25
         </td>
      </tr>
      <tr>
         <td rowspan="3" colspan="1">
            QReCC
         </td>
         <td rowspan="1" colspan="1">
            Recall@1
         </td>
         <td rowspan="1" colspan="1">
            49.00
         </td>
         <td rowspan="1" colspan="1">
            49.43
         </td>
         <td rowspan="1" colspan="1">
            55.41
         </td>
      </tr>
      <tr>
         <td rowspan="1" colspan="1">
            Recall@5
         </td>
         <td rowspan="1" colspan="1">
            85.14
         </td>
         <td rowspan="1" colspan="1">
            86.6
         </td>
         <td rowspan="1" colspan="1">
            90.11
         </td>
      </tr>
      <tr>
         <td rowspan="1" colspan="1">
            Recall@20
         </td>
         <td rowspan="1" colspan="1">
            97.21
         </td>
         <td rowspan="1" colspan="1">
            98.28
         </td>
         <td rowspan="1" colspan="1">
            98.39
         </td>
      </tr>
      <tr>
         <td rowspan="3" colspan="1">
            INSCIT*
         </td>
         <td rowspan="1" colspan="1">
            Recall@1
         </td>
         <td rowspan="1" colspan="1">
            11.13
         </td>
         <td rowspan="1" colspan="1">
            18.35
         </td>
         <td rowspan="1" colspan="1">
            21.65
         </td>
      </tr>
      <tr>
         <td rowspan="1" colspan="1">
            Recall@5
         </td>
         <td rowspan="1" colspan="1">
            29.27
         </td>
         <td rowspan="1" colspan="1">
            48.45
         </td>
         <td rowspan="1" colspan="1">
            50.72
         </td>
      </tr>
      <tr>
         <td rowspan="1" colspan="1">
            Recall@20
         </td>
         <td rowspan="1" colspan="1">
            49.07
         </td>
         <td rowspan="1" colspan="1">
            66.19
         </td>
         <td rowspan="1" colspan="1">
            72.78
         </td>
      </tr>
      <tr>
         <td rowspan="3" colspan="1">
            Topiocqa*
         </td>
         <td rowspan="1" colspan="1">
            Recall@1
         </td>
         <td rowspan="1" colspan="1">
            29.19
         </td>
         <td rowspan="1" colspan="1">
            31.34
         </td>
         <td rowspan="1" colspan="1">
            38.19
         </td>
      </tr>
      <tr>
         <td rowspan="1" colspan="1">
            Recall@5
         </td>
         <td rowspan="1" colspan="1">
            62.52
         </td>
         <td rowspan="1" colspan="1">
            65.79
         </td>
         <td rowspan="1" colspan="1">
            72.47
         </td>
      </tr>
      <tr>
         <td rowspan="1" colspan="1">
            Recall@20
         </td>
         <td rowspan="1" colspan="1">
            83.69
         </td>
         <td rowspan="1" colspan="1">
            84.37
         </td>
         <td rowspan="1" colspan="1">
            87.23
         </td>
      </tr>
      <tr>
         <td rowspan="2" colspan="1">
            Average**
         </td>
         <td rowspan="1" colspan="1">
            Avg top 1
         </td>
         <td rowspan="1" colspan="1">
            49.36
         </td>
         <td rowspan="1" colspan="1">
            54.76
         </td>
         <td rowspan="1" colspan="1">
            58.29
         </td>
      </tr>
      <tr>
         <td rowspan="1" colspan="1">
            Avg top 5
         </td>
         <td rowspan="1" colspan="1">
            76.30
         </td>
         <td rowspan="1" colspan="1">
            81.50
         </td>
         <td rowspan="1" colspan="1">
            84.19
         </td>
      </tr>
   </tbody>
</table>

\*Evaluated on a subset of the wikipedia corpus that was available to us. All models use the same evaluation strategy to ensure apples-to-apples comparisons.

\*\* We follow the same convention as in ChatQA, where we compare top-5 and top-20 of TopiOCQA and INSCIT to top-1 and top-5, respectively, of the other datasets, in order match differences in average context length.


### The Recipe

While ChatQA provided a valuable foundation, we identified several gaps in their released datasets and training recipes. We crafted our final recipe by combining insights from analyzing their model as well as our own experience in LLM model training and dataset curation. Notably, we addressed the challenge of handling unanswerable questions, improving arithmetic performance and entity extraction:

* **Handling Unanswerable Questions**: In our initial attempts, the model struggled with unanswerable questions (i.e. responding “I can’t answer …”). The ChatQA paper notes that training on the synthetic conversational Q&A dataset yields worse performance on unanswerable benchmarks relative to when trained with their human dataset (which we don’t have access to); even so, the gap that we observed in initial experiments was substantial, requiring an alternative approach. Through experimentation, we found that upsampling the samples corresponding to unanswerable questions improves performance. Despite the improvement, the performance delta to SOTA on the QuAC and DoQA benchmarks (which test handling of unanswerable questions) show that there’s still room for further improvement.
* **Arithmetic Performance**: We observed that initial iterations of our model frequently made errors on arithmetic tasks (such as ConvFinQA) because it was trained to produce a final equation in a single shot. Inspired by Chain of Thought (CoT), we synthetically generated a CoT variant of the TAT-QA arithmetic dataset using Llama 3 70B Instruct which teaches the model to explicitly reason through it’s response before providing a final answer. We found that this addition led to a substantial boost in accuracy (+10 on ConvFinQA). We also mix in a small amount of the NuminaMath-CoT dataset, a collection of math problems with CoT solutions that range from high-school to olympiad difficulty levels, in order to improve our model’s GSM8k score.
* **Entity Extraction**: Our baseline performed poorly on entity extraction style tasks such as SQA, due to very few high quality samples that demonstrate the intended behavior. We solve this by mixing in a subset of SKGInstruct, an instruction tuning dataset constructed from a mix of structured knowledge grounding datasets.

### What’s Next?

Our model series demonstrates strong conversational Q&A and multi-turn retrieval capabilities, and we can’t wait to see what the community can achieve with it. We invite you to explore, experiment, and improve upon our work. There are many exciting directions we’re exploring that we plan to use this model series as the foundation for, such as long context support, improved math/reasoning, larger model sizes, etc. Stay tuned for more updates and improvements. 🚀


## Acknowledgements

DocChat was built on top of a large body of ML work, spanning training datasets, recipes, and evaluation. We want to thank each of these resources.

```
@inproceedings{dua2019drop,
  title={DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs},
  author={Dua, Dheeru and Wang, Yizhong and Dasigi, Pradeep and Stanovsky, Gabriel and Singh, Sameer and Gardner, Matt},
  booktitle={Proceedings of the 2019 Conference on NAACL},
  year={2019}
}
@article{kocisky2018narrativeqa,
  title={The NarrativeQA Reading Comprehension Challenge},
  author={Kocisky, Tomas and Schwarz, Jonathan and Blunsom, Phil and Dyer, Chris and Hermann, Karl Moritz and Melis, Gabor and Grefenstette, Edward},
  journal={Transactions of the Association for Computational Linguistics},
  year={2018}
}
@inproceedings{dasigi2019quoref,
  title={Quoref: A Reading Comprehension Dataset with Questions Requiring Coreferential Reasoning},
  author={Dasigi, Pradeep and Liu, Nelson F and Marasovi{\'c}, Ana and Smith, Noah A and Gardner, Matt},
  booktitle={Proceedings of the 2019 Conference on EMNLP},
  year={2019}
}
@inproceedings{lin2019reasoning,
  title={Reasoning Over Paragraph Effects in Situations},
  author={Lin, Kevin and Tafjord, Oyvind and Clark, Peter and Gardner, Matt},
  booktitle={Proceedings of the 2nd Workshop on Machine Reading for Question Answering},
  year={2019}
}
@inproceedings{rajpurkar2016squad,
  title={SQuAD: 100,000+ Questions for Machine Comprehension of Text},
  author={Rajpurkar, Pranav and Zhang, Jian and Lopyrev, Konstantin and Liang, Percy},
  booktitle={Proceedings of the 2016 Conference on EMNLP},
  year={2016}
}
@inproceedings{rajpurkar2018know,
  title={Know What You Don’t Know: Unanswerable Questions for SQuAD},
  author={Rajpurkar, Pranav and Jia, Robin and Liang, Percy},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
  pages={784--789},
  year={2018}
}
@inproceedings{trischler2017newsqa,
  title={NewsQA: A Machine Comprehension Dataset},
  author={Trischler, Adam and Wang, Tong and Yuan, Xingdi and Harris, Justin and Sordoni, Alessandro and Bachman, Philip and Suleman, Kaheer},
  booktitle={Proceedings of the 2nd Workshop on Representation Learning for NLP},
  year={2017}
}
@inproceedings{zhu2021tat,
  title={TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance},
  author={Zhu, Fengbin and Lei, Wenqiang and Huang, Youcheng and Wang, Chao and Zhang, Shuo and Lv, Jiancheng and Feng, Fuli and Chua, Tat-Seng},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
  year={2021}
}
@inproceedings{kim2023soda,
  title={SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization},
  author={Kim, Hyunwoo and Hessel, Jack and Jiang, Liwei and West, Peter and Lu, Ximing and Yu, Youngjae and Zhou, Pei and Bras, Ronan and Alikhani, Malihe and Kim, Gunhee and others},
  booktitle={Proceedings of the 2023 Conference on EMNLP},
  year={2023}
}
@inproceedings{fan2019eli5,
  title={ELI5: Long Form Question Answering},
  author={Fan, Angela and Jernite, Yacine and Perez, Ethan and Grangier, David and Weston, Jason and Auli, Michael},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  year={2019}
}
@article{chung2024scaling,
  title={Scaling instruction-finetuned language models},
  author={Chung, Hyung Won and Hou, Le and Longpre, Shayne and Zoph, Barret and Tay, Yi and Fedus, William and Li, Yunxuan and Wang, Xuezhi and Dehghani, Mostafa and Brahma, Siddhartha and others},
  journal={Journal of Machine Learning Research},
  year={2024}
}
@inproceedings{longpre2023flan,
  title={The flan collection: Designing data and methods for effective instruction tuning},
  author={Longpre, Shayne and Hou, Le and Vu, Tu and Webson, Albert and Chung, Hyung Won and Tay, Yi and Zhou, Denny and Le, Quoc V and Zoph, Barret and Wei, Jason and others},
  booktitle={International Conference on Machine Learning},
  year={2023},
}
@inproceedings{wang2023self,
  title={Self-Instruct: Aligning Language Models with Self-Generated Instructions},
  author={Wang, Yizhong and Kordi, Yeganeh and Mishra, Swaroop and Liu, Alisa and Smith, Noah A and Khashabi, Daniel and Hajishirzi, Hannaneh},
  booktitle={Proceedings of the 61st Annual Meeting Of The Association For Computational Linguistics},
  year={2023}
}
@inproceedings{honovich2023unnatural,
  title={Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor},
  author={Honovich, Or and Scialom, Thomas and Levy, Omer and Schick, Timo},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
  year={2023}
}
@article{kopf2024openassistant,
  title={Openassistant conversations-democratizing large language model alignment},
  author={K{\"o}pf, Andreas and Kilcher, Yannic and von R{\"u}tte, Dimitri and Anagnostidis, Sotiris and Tam, Zhi Rui and Stevens, Keith and Barhoum, Abdullah and Nguyen, Duc and Stanley, Oliver and Nagyfi, Rich{\'a}rd and others},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
@online{DatabricksBlog2023DollyV2,
    author    = {Mike Conover and Matt Hayes and Ankit Mathur and Jianwei Xie and Jun Wan and Sam Shah and Ali Ghodsi and Patrick Wendell and Matei Zaharia and Reynold Xin},
    title     = {Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM},
    year      = {2023},
    url       = {https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm},
    urldate   = {2023-06-30}
}
@misc{numina_math_datasets,
  author = {Jia LI and Edward Beeching and Lewis Tunstall and Ben Lipkin and Roman Soletskyi and Shengyi Costa Huang and Kashif Rasul and Longhui Yu and Albert Jiang and Ziju Shen and Zihan Qin and Bin Dong and Li Zhou and Yann Fleureau and Guillaume Lample and Stanislas Polu},
  title = {NuminaMath},
  year = {2024},
  publisher = {Numina},
  journal = {Hugging Face repository},
  howpublished = {\url{[https://huggingface.co/AI-MO/NuminaMath-CoT](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf)}}
}
@misc{zhuang2024structlm,
      title={StructLM: Towards Building Generalist Models for Structured Knowledge Grounding}, 
      author={Alex Zhuang and Ge Zhang and Tianyu Zheng and Xinrun Du and Junjie Wang and Weiming Ren and Stephen W. Huang and Jie Fu and Xiang Yue and Wenhu Chen},
      year={2024},
      eprint={2402.16671},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@article{llama3modelcard,
  title={Llama 3 Model Card},
  author={AI@Meta},
  year={2024},
  url = {https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md}
}
@article{liu2024chatqa,
  title={ChatQA: Surpassing GPT-4 on Conversational QA and RAG},
  author={Liu, Zihan and Ping, Wei and Roy, Rajarshi and Xu, Peng and Lee, Chankyu and Shoeybi, Mohammad and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:2401.10225},
  year={2024}}
@article{lin_dragon_2023,
	title = How to Train Your DRAGON: Diverse Augmentation Towards Generalizable Dense Retrieval,
	url = {https://arxiv.org/abs/2302.07452},
	publisher = {arXiv},
	journal = {arXiv e-print 2302.07452},
	author = {Sheng-Chieh Lin, Akari Asai, Minghan Li, Barlas Oguz, Jimmy Lin, Yashar Mehdad, Wen-tau Yih, Xilun Chen},
	year = {2023},
}
@inproceedings{feng2020doc2dial,
  title={doc2dial: A Goal-Oriented Document-Grounded Dialogue Dataset},
  author={Feng, Song and Wan, Hui and Gunasekara, Chulaka and Patel, Siva and Joshi, Sachindra and Lastras, Luis},
  booktitle={Proceedings of the 2020 Conference on EMNLP},
  year={2020}
}
@inproceedings{choi2018quac,
  title={QuAC: Question Answering in Context},
  author={Choi, Eunsol and He, He and Iyyer, Mohit and Yatskar, Mark and Yih, Wen-tau and Choi, Yejin and Liang, Percy and Zettlemoyer, Luke},
  booktitle={Proceedings of the 2018 Conference on EMNLP},
  year={2018}
}
@inproceedings{anantha2021open,
  title={Open-Domain Question Answering Goes Conversational via Question Rewriting},
  author={Anantha, Raviteja and Vakulenko, Svitlana and Tu, Zhucheng and Longpre, Shayne and Pulman, Stephen and Chappidi, Srinivas},
  booktitle={Proceedings of the 2021 Conference on NAACL},
  year={2021}
}
@article{reddy2019coqa,
  title={CoQA: A Conversational Question Answering Challenge},
  author={Reddy, Siva and Chen, Danqi and Manning, Christopher D},
  journal={Transactions of the Association for Computational Linguistics},
  year={2019}
}
@inproceedings{campos2020doqa,
  title={DoQA-Accessing Domain-Specific FAQs via Conversational QA},
  author={Campos, Jon Ander and Otegi, Arantxa and Soroa, Aitor and Deriu, Jan Milan and Cieliebak, Mark and Agirre, Eneko},
  booktitle={Proceedings of the 2020 Conference on ACL},
  year={2020}
}
@inproceedings{chen2022convfinqa,
  title={ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering},
  author={Chen, Zhiyu and Li, Shiyang and Smiley, Charese and Ma, Zhiqiang and Shah, Sameena and Wang, William Yang},
  booktitle={Proceedings of the 2022 Conference on EMNLP},
  year={2022}
}
@inproceedings{iyyer2017search,
  title={Search-based neural structured learning for sequential question answering},
  author={Iyyer, Mohit and Yih, Wen-tau and Chang, Ming-Wei},
  booktitle={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics},
  year={2017}
}
@article{adlakha2022topiocqa,
  title={TopiOCQA: Open-domain Conversational Question Answering with Topic Switching},
  author={Adlakha, Vaibhav and Dhuliawala, Shehzaad and Suleman, Kaheer and de Vries, Harm and Reddy, Siva},
  journal={Transactions of the Association for Computational Linguistics},
  year={2022}
}
@inproceedings{nakamura2022hybridialogue,
  title={HybriDialogue: An Information-Seeking Dialogue Dataset Grounded on Tabular and Textual Data},
  author={Nakamura, Kai and Levy, Sharon and Tuan, Yi-Lin and Chen, Wenhu and Wang, William Yang},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2022},
  year={2022}
}
@article{wu2023inscit,
  title={InSCIt: Information-Seeking Conversations with Mixed-Initiative Interactions},
  author={Wu, Zeqiu and Parish, Ryu and Cheng, Hao and Min, Sewon and Ammanabrolu, Prithviraj and Ostendorf, Mari and Hajishirzi, Hannaneh},
  journal={Transactions of the Association for Computational Linguistics},
  year={2023}
}
```
