import json
import chatbot
from chatbot import Preprocess

tokenize = Preprocess.tokenize
tokenizer = Preprocess.tokenizer

# generate tokenized qna for training transformer
qna_tokenized = []

with open('../clinicscraper/qna.json', 'r', encoding='utf-8') as f:
    qna_list = json.load(f)
    # print(qna_list[0])

for qna in qna_list:

    # skip empty question/answer
    if qna['answer'] == "" or qna['question'] == "":
        continue

    # create list of tokenized sentences
    qna_tokenized.append({
        'question' : tokenize(qna['question']),
        'answer' : tokenize(qna['answer'])
    })

with open('qna_tokenized.json', 'w', encoding='utf-8') as f:
    json.dump(qna_tokenized, f, indent=4)