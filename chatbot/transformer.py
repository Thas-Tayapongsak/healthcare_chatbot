import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F


class Preprocess():

    tokenizer = tiktoken.get_encoding("gpt2")

    def __init__():
        pass

    def tokenize(s):
        s += "<|endoftext|>"
        return Preprocess.tokenizer.encode(s, allowed_special={"<|endoftext|>"})
    
    def embed(tokens):
        model = torch.load("./weights/cbow.pt")
        model.eval()

        embed_mat = model.embed.weight.detach()

        embedding = embed_mat[tokens]
        
        return embedding
    
class CBOW(nn.Module):
    def __init__(self, vocab_size, d_model, window_size):
        super(CBOW, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)
        self.window_size = window_size

    def forward(self, tokens):
        embedding = torch.sum(self.embed(tokens), dim=0) 
        out = self.linear(embedding) 
        log_probs = F.log_softmax(out, dim=1) 

        return log_probs

#TODO: PositionalEncoding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # matrix of attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # replace with -1e9 where mask is equal to 0
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # do softmax on each column
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # get the value vector to add to embedding
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
#TODO: Encoder, Decoder, Transformer