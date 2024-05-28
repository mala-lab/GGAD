import torch
import torch.nn as nn
from torch.nn import init




class PCALayer(nn.Module):


	def __init__(self, num_classes, inter1, lambda_1):

		super(PCALayer, self).__init__()
		self.inter1 = inter1
		self.xent = nn.CrossEntropyLoss()

		# the parameter to transform the final embedding
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, inter1.embed_dim))
		init.xavier_uniform_(self.weight)
		self.lambda_1 = lambda_1
		self.epsilon = 0.1

	def forward(self, nodes, labels, train_flag=True):
		embeds1, affinity = self.inter1(nodes, labels, train_flag)
		scores = self.weight.mm(embeds1)
		return scores.t(), affinity

	def to_prob(self, nodes, labels, train_flag=True):
		gnn_logits, label_logits = self.forward(nodes, labels, train_flag)
		gnn_scores = torch.sigmoid(gnn_logits)
		label_scores = torch.sigmoid(label_logits)
		return gnn_scores, label_scores

	# Add loss function on affinity
	def affinity(self, affinity, labels):
		# marigin loss function
		confidence_margin = 1
		affinity_normal_mean = torch.mean(affinity[torch.argwhere(labels ==0)], 0)
		affinity_abnormal_mean = torch.mean(affinity[torch.argwhere(labels ==1)], 0)
		loss_margin = (confidence_margin - (affinity_normal_mean - affinity_abnormal_mean)).clamp_min(min=0)
		return loss_margin

	def loss(self, nodes, labels, train_flag=True):
		label_scores, affinity = self.forward(nodes, labels, train_flag)
		# Simi loss, Eq. (7) in the paper
		loss_cls = self.xent(label_scores, labels.squeeze())
		loss_constraint = self.affinity(affinity, labels)
		return loss_cls+5*loss_constraint, loss_constraint
