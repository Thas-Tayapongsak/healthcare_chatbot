import torch
import transformer
from transformer import Preprocess

transformer1 = torch.load("./weights/transformer1.pt")
transformer1.eval()

tokenize = Preprocess.tokenize
tokenizer = Preprocess.tokenizer
# change input here vvvv
input_seq  = torch.LongTensor(tokenize('What are the causes and symptoms of trichinellosis?'))
output_seq = torch.LongTensor([50256])
max_output_seq = 100
next_token = 0
while max_output_seq > 0 or next_token == 50256:
    output = transformer1(input_seq, output_seq)
    next_token = torch.multinomial(output, 1)[-1]
    # next_token = output.tolist().index(max(output.tolist()))
    output_seq = torch.LongTensor([next_token])
    # output_seq = torch.LongTensor(output_seq.tolist() + [next_token])
    print(tokenizer.decode([next_token]), end = "")
    max_output_seq -= 1