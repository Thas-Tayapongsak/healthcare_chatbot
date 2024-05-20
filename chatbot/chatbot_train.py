import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import transformer
from transformer import Transformer
from transformer import Preprocess

NUM_EPOCH = 10

src_vocab_size = 50257
tgt_vocab_size = 50257
d_model = 50
num_heads = 5
num_layers = 6
d_ff = 500
max_seq_length = 200
dropout = 0.1

with open('qna_tokenized.json', 'r', encoding='utf-8') as f:
    qna_list = json.load(f)

random.shuffle(qna_list)

qna_train = []
qna_val = []

training_line = int(len(qna_list)*0.8)

for qna in qna_list[:training_line]:
    qna_train.append(qna)
for qna in qna_list[training_line:]:
    qna_val.append(qna)

transformer1 = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer1.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer1.train()

for epoch in range(NUM_EPOCH):
    total_loss = 0
    for qna in qna_train[:5000]:
        if len(qna['question'])>200:
            qna_question = qna['question'][:199] + [50256]
        else:
            qna_question = qna['question']

        if len(qna['answer'])>200:
            qna_answer = [50256] + qna['answer'][:198] + [50256]
        else:
            qna_answer = qna['answer']

        src_data = torch.LongTensor(qna_question)
        tgt_data = torch.LongTensor(qna_answer)

        optimizer.zero_grad()
        output = transformer1(src_data, tgt_data[:-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{NUM_EPOCH}], Loss: {total_loss:.4f}')

torch.save(transformer1, "./weights/transformer1.pt")

