import math
import torch
import torch.nn as nn
import sys

class DenseCoAttn(nn.Module):

	def __init__(self, dim1, dim2, dim3, dropout):
		super(DenseCoAttn, self).__init__()
		dim = dim1 + dim2 + dim3
		self.dropouts = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(3)])
		self.query_linear = nn.Linear(dim, dim)

		self.key1_linear = nn.Linear(300, 300)
		self.key2_linear = nn.Linear(300, 300)
		self.key3_linear = nn.Linear(300, 300)
		self.value1_linear = nn.Linear(dim1, dim1)
		self.value2_linear = nn.Linear(dim2, dim2)
		self.value3_linear = nn.Linear(dim3, dim3)
		self.relu = nn.ReLU()

	def forward(self, value1, value2, value3):

		joint = torch.cat((value1, value2, value3), dim=-1)

		# audio  audio*W*joint
		va_joint = self.query_linear(joint)

		key1 = self.key1_linear(value1.transpose(1, 2))
		key2 = self.key2_linear(value2.transpose(1, 2))
		key3 = self.key3_linear(value3.transpose(1, 2))

		value1 = self.value1_linear(value1)
		value2 = self.value2_linear(value2)
		value3 = self.value3_linear(value3)

		weighted1, attn1 = self.qkv_attention(joint, key1, value1, dropout=self.dropouts[0])
		weighted2, attn2 = self.qkv_attention(joint, key2, value2, dropout=self.dropouts[1])
		weighted3, attn3 = self.qkv_attention(joint, key3, value3, dropout=self.dropouts[2])

		return weighted1, weighted2, weighted3

	def qkv_attention(self, query, key, value, dropout=None):
		d_k = query.size(-1)
		scores = torch.bmm(key, query) / math.sqrt(d_k)
		scores = torch.tanh(scores)
		if dropout:
			scores = dropout(scores)

		weighted = torch.tanh(torch.bmm(value, scores))
		return self.relu(weighted), scores
