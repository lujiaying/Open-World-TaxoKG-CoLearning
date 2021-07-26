"""
Infer CGC by aggregate relation(bi-direction)
Infer OLP by aggregate taxonomy
Author: Jiaying Lu
Create Date: Jul 15, 2021
"""

# aggregate over relation: https://arxiv.org/pdf/1911.03082.pdf
# k-hop ego local graph of entity `e` for k-layer COMPGCN
# h(e/c/r) = LSTM(emb(e_t1, e_t2, ...))
# score func for entity `e` and concept `c`: sigmoid(COMPGCN(h(e))^T * MLP(h(c)))
