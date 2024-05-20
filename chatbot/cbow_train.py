import json
import chatbot
from chatbot import Preprocess

# tokenize = Preprocess.tokenize
# tokenizer = Preprocess.tokenizer

# print(tokenize('What is an A1C test?'))


# print('What is an A1C test?')
# print(tokenize('What is an A1C test?'))
# print(tokenize('What is an A1C test? <|endoftext|>'))
# print(tokenizer.decode(tokenize('What is an A1C test?')))

WINDOW_SIZE = 3


def createcorpus():
    """Create corpus from tokenized sequences

    Run gencorpus.py to create qna_tokenized.json before using this function.

    """
    with open('qna_tokenized.json', 'r', encoding='utf-8') as f:
        qna_list = json.load(f)

    corpus = []
    for qna in qna_list:
        corpus.append(qna['question'])
        corpus.append(qna['answer'])

    # print(corpus)
    return corpus


def create_CT_pairs(corpus):
    """Create context target pais from corpus

    Parameter
    ----------
    corpus : list of int
        list of all token sequences

    Returns
    -------
    ct_pairs : list of tuple
        list of target token and its context

    """
    contexts = []
    targets = []
    for sequence in corpus:
        for i in range(WINDOW_SIZE, len(sequence) - WINDOW_SIZE):
            context = sequence[i - WINDOW_SIZE:i] +\
                sequence[i + 1:i + WINDOW_SIZE + 1]
            target = sequence[i]
            contexts.append(context)
            targets.append(target)

    ct_pairs = [(key, value) for key, value in zip(targets, contexts)]
    # print(len(ct_pairs))
    return ct_pairs

def int2onehot(num):
    """Take an integer and create a one hot tensor at num of size vocab size

    Parameter
    ---------
    num : int

    Returns
    -------
    onehot: tensor

    """
    onehot = torch.zeros(VOCAB_SIZE).long()
    onehot[num] = 1
    return onehot


### train CBOW
from chatbot import CBOW
import torch
import torch.nn as nn
import torch.optim as optim

VOCAB_SIZE = 50257 
D_MODEL = 20
NUM_EPOCH = 4

model = CBOW(VOCAB_SIZE, D_MODEL, WINDOW_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()
losses = []
corpus = createcorpus()
ct_pairs = create_CT_pairs(corpus)

for epoch in range(NUM_EPOCH):
    total_loss = 0
    for ct_pair in ct_pairs[:100]:
        target = int2onehot(ct_pair[0])
        context = torch.LongTensor(ct_pair[1]).unsqueeze(1)
        model.zero_grad()
        pred = model(context).squeeze()
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    losses.append(total_loss)

    print(f'Epoch [{epoch+1}/{NUM_EPOCH}], Loss: {loss.item():.4f}')

torch.save(model, "./weights/cbow.pt")

model = torch.load("./weights/cbow.pt")
model.eval()

embed_mat = model.embed.weight.detach()

print(embed_mat[317])
print(embed_mat[17])
print(embed_mat[34])