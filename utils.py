import networkx as nx
import numpy as np
import random
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import os.path as osp

from algorithm import *
from netrd.distance import netsimile
from torch_geometric.data import InMemoryDataset, Data
from scipy.io import loadmat
from scipy.sparse import coo_matrix
from sklearn.preprocessing import StandardScaler


def load_adj(dataset):
	if (dataset == "celegans"):
		# S = torch.load("data/celegans.pt")
		S = torch.load("data/celegans.pt", weights_only=False)
	elif(dataset == "arenas"):
		# S = torch.load("data/arenas.pt")
		S = torch.load("data/arenas.pt", weights_only=False)
	elif (dataset == "douban"):
		# S = torch.load("data/douban.pt")
		S = torch.load("data/douban.pt", weights_only=False)
	elif(dataset == "Online"):
		# S = torch.load("data/online.pt")
		S = torch.load("data/online.pt", weights_only=False)
	elif(dataset == "Offline"):
		# S = torch.load("data/offline.pt")
		S = torch.load("data/offline.pt", weights_only=False)
	elif (dataset == "ACM"):
		# S = torch.load("data/ACM.pt")
		S = torch.load("data/ACM.pt", weights_only=False)
	elif (dataset == "DBLP"):
		# S = torch.load("data/DBLP.pt")
		S = torch.load("data/DBLP.pt", weights_only=False)
	else:
		filepath = "data/" + dataset + ".npz"
		loader = load_npz(filepath)
		data = loader["adj_matrix"]
		samples = data.shape[0]
		features = data.shape[1]
		values = data.data
		coo_data = data.tocoo()
		# indices = torch.LongTensor([coo_data.row, coo_data.col])
		indices = torch.from_numpy(np.array([coo_data.row, coo_data.col]))
		# S = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features]).to_dense()
		S = torch.sparse_coo_tensor(indices, torch.from_numpy(values).float(), [samples, features]).to_dense()
		if (not torch.all(S.transpose(0, 1) == S)):
			S = torch.add(S, S.transpose(0, 1))
		S = S.int()
		ones = torch.ones_like(S)
		S = torch.where(S > 1, ones, S)
	return S

def test_matching(TGAE, S_hat_samples, p_samples, S_hat_features, S_emb, device, algorithm, metric):
	if (metric == "accuracy"):
		results = []
	else:
		results = {}
		results["hit@1"] = []
		results["hit@5"] = []
		results["hit@10"] = []
		results["hit@50"] = []
	for i in range(len(S_hat_samples)):
		S_hat_cur = S_hat_samples[i]
		adj = coo_matrix(S_hat_cur.numpy())
		adj_norm = preprocess_graph(adj)
		""" adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
											torch.FloatTensor(adj_norm[1]),
											torch.Size(adj_norm[2])).to(device) """
		adj_norm = torch.sparse_coo_tensor(torch.LongTensor(adj_norm[0].T),
											torch.FloatTensor(adj_norm[1]),
											torch.Size(adj_norm[2])).to(device)
		initial_feature = S_hat_features[i].to(device)
		z = TGAE(initial_feature, adj_norm).detach()
		D = torch.cdist(S_emb, z, p=2)
		if (metric == "accuracy"):
			if(algorithm == "greedy"):
				P_HG = greedy_hungarian(D, device)
			elif(algorithm == "exact"):
				P_HG = hungarian(D)
			elif(algorithm == "approxNN"):
				P_HG = approximate_NN(S_emb,z)
			else:
				print("Matching algorithm undefined")
				exit()
			c = 0
			P = p_samples[i]
			for j in range(P_HG.size(0)):
				r1 = P_HG[j].cpu()
				r2 = P[j].cpu()
				if (r1.equal(r2)): c += 1
			results.append(c / S_emb.shape[0])
		else:
			P = p_samples[i].T
			hitAtOne = 0
			hitAtFive = 0
			hitAtTen = 0
			hitAtFifty = 0
			for j in range(P.size(0)):
				label = torch.nonzero(P)[j][1]
				dist_list = D[j]
				sorted_neighbors = torch.argsort(dist_list).cpu()
				for hit in range(50):
					if (sorted_neighbors[hit].item() == label):
						if (hit == 0):
							hitAtOne += 1
							hitAtFive += 1
							hitAtTen += 1
							hitAtFifty += 1
							break
						elif (hit <= 4):
							hitAtFive += 1
							hitAtTen += 1
							hitAtFifty += 1
							break
						elif (hit <= 9):
							hitAtTen += 1
							hitAtFifty += 1
							break
						elif (hit <= 49):
							hitAtFifty += 1
							break
			results["hit@1"].append(hitAtOne)
			results["hit@5"].append(hitAtFive)
			results["hit@10"].append(hitAtTen)
			results["hit@50"].append(hitAtFifty)

	if (metric == "accuracy"):
		results = np.array(results)
		avg = np.average(results)
		std = np.std(results)
		return avg, std
	else:
		hitAtOne = np.average(np.array(results["hit@1"]))
		stdAtOne = np.std(np.array(results["hit@1"]))
		hitAtFive = np.average(np.array(results["hit@5"]))
		stdAtFive = np.std(np.array(results["hit@5"]))
		hitAtTen = np.average(np.array(results["hit@10"]))
		stdAtTen = np.std(np.array(results["hit@10"]))
		hitAtFifty = np.average(np.array(results["hit@50"]))
		stdAtFifty = np.std(np.array(results["hit@50"]))
		num_nodes = S_emb.shape[0]
		print("Hit@1: ", end="")
		print(str(hitAtOne / num_nodes)[:6] + "+-" + str(stdAtOne / num_nodes)[:6])
		print("Hit@5: ", end="")
		print(str(hitAtFive / num_nodes)[:6] + "+-" + str(stdAtFive / num_nodes)[:6])
		print("Hit@10: ", end="")
		print(str(hitAtTen / num_nodes)[:6] + "+-" + str(stdAtTen / num_nodes)[:6])
		print("Hit@50: ", end="")
		print(str(hitAtFifty / num_nodes)[:6] + "+-" + str(stdAtFifty / num_nodes)[:6])
		print()

def gen_test_set(device,S, no_samples_each_level, perturbation_levels,method):
	S_hat_samples = {}
	S_prime_samples = {}
	p_samples = {}
	for level in perturbation_levels:
		S_hat_samples[str(level)] = []
		S_prime_samples[str(level)] = []
		p_samples[str(level)] = []
	for level in perturbation_levels:
		num_edges = int(torch.count_nonzero(S).item() / 2)
		total_purturbations = int(num_edges*level)
		if(method == "degree"):
			S = torch.triu(S, diagonal=0)
			ones_long = torch.ones((S.shape[0], 1)).type(torch.LongTensor)
			ones_int = torch.ones((S.shape[0], 1)).type(torch.IntTensor)
			ones_float = torch.ones((S.shape[0], 1)).type(torch.FloatTensor)
			try:
				D = S @ ones_long
			except:
				try:
					D = S @ ones_int
				except:
					D = S @ ones_float
			sum = torch.sum(torch.mul(D @ D.T, S))
			edge_index = S.nonzero().t().contiguous()
			edge_index = np.array(edge_index)
			prob = []
			for i in range(edge_index.shape[1]):
				d1 = edge_index[0, i]
				d2 = edge_index[1, i]
				prob.append(D[d1] * D[d2] / sum)
			prob = np.array(prob, dtype='float64')
			prob = np.squeeze(prob)
		for i in range(no_samples_each_level):
			if(method == "uniform"):
				add_edge = random.randint(0, total_purturbations)
				delete_edge = total_purturbations - add_edge
				S, S_prime, S_hat, P = gen_dataset(S.to(device), add_edge, delete_edge)
			elif(method == "degree"):
				edges_to_remove = np.random.choice(edge_index.shape[1], total_purturbations, False, prob)
				edges_remain = np.setdiff1d(np.array(range(edge_index.shape[1])), edges_to_remove)
				edges_index = edge_index[:, edges_remain]
				S_prime = torch.zeros_like(S)
				for j in range(edges_index.shape[1]):
					n1 = edges_index[:, j][0]
					n2 = edges_index[:, j][1]
					S_prime[n1][n2] = 1
					if (S_prime[n2][n1] == 0):
						S_prime[n2][n1] = 1
				SIZE = S_prime.shape[0]
				permutator = torch.randperm(SIZE)
				S_hat = S_prime[permutator]
				S_hat = S_hat.t()[permutator].t()
				P = torch.zeros(SIZE, SIZE)
				for i in range(permutator.shape[0]):
					P[i, permutator[i]] = 1
			else:
				print("Probability model not defined")
				exit()
			S_hat_samples[str(level)].append(S_hat)
			p_samples[str(level)].append(P)
			S_prime_samples[str(level)].append(S_prime)
	return S_hat_samples, S_prime_samples, p_samples

def generate_features(purturbated_S):
	features = []
	for S in purturbated_S:
		feature = gen_netsmile(S)
		features.append(feature)
	return features

def gen_dataset(S, NUM_TO_ADD, NUM_TO_DELETE):
	SIZE = S.shape[0]
	num_added = 0
	num_deleted = 0
	E = torch.zeros(S.shape[0], S.shape[0])
	edge_indexes = (S == 1).nonzero(as_tuple=False).cpu()
	blank_indexes = (S == 0).nonzero(as_tuple=False).cpu()
	"""
	delete edges
	"""
	while(num_deleted < NUM_TO_DELETE):

		delete_index = random.randint(0, edge_indexes.shape[0]-1)
		index = edge_indexes[delete_index]
		E[index[0]][index[1]] = -1
		E[index[1]][index[0]] = -1
		num_deleted += 1

	"""
	add edges
	"""
	while (num_added < NUM_TO_ADD):

		add_index = random.randint(0, blank_indexes.shape[0] - 1)
		index = blank_indexes[add_index]
		E[index[0]][index[1]] = 1
		E[index[1]][index[0]] = 1
		num_added += 1

	S_prime = torch.add(S.cpu(),E.cpu())
	permutator = torch.randperm(SIZE)
	S_hat = S_prime[permutator]
	S_hat = S_hat.t()[permutator].t()
	P = torch.zeros(SIZE, SIZE)
	for i in range(permutator.shape[0]):
		P[i, permutator[i]] = 1
	return S, S_prime, S_hat, P

def sparse_to_tuple(sparse_mx):
	if not sp.isspmatrix_coo(sparse_mx):
		sparse_mx = sparse_mx.tocoo()
	coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
	values = sparse_mx.data
	shape = sparse_mx.shape
	return coords, values, shape

def preprocess_graph(adj):
	adj = sp.coo_matrix(adj)
	adj_ = adj + sp.eye(adj.shape[0])
	rowsum = np.array(adj_.sum(1))
	degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
	adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
	return sparse_to_tuple(adj_normalized)

def generate_purturbations(device, S, perturbation_level, no_samples, method):
	purturbated_samples = []
	if(method == "uniform"):
		for i in range(no_samples):
			num_edges = int(torch.count_nonzero(S).item()/2)
			total_purturbations = int(perturbation_level * num_edges)
			add_edge = random.randint(0,total_purturbations)
			delete_edge = total_purturbations - add_edge
			S, S_prime, S_hat, P = gen_dataset(S.to(device), add_edge, delete_edge)
			purturbated_samples.append(S_prime)
	elif(method == "degree"):
		num_edges = int(torch.count_nonzero(S).item() / 2)
		total_purturbations = int(perturbation_level * num_edges)
		S = torch.triu(S, diagonal=0)
		ones_float = torch.ones((S.shape[0], 1)).type(torch.FloatTensor)
		ones_long = torch.ones((S.shape[0], 1)).type(torch.LongTensor)
		ones_int = torch.ones((S.shape[0], 1)).type(torch.IntTensor)
		try:
			D = S @ ones_long
		except:
			try:
				D = S @ ones_int
			except:
				D = S @ ones_float

		sum = torch.sum(torch.mul(D@D.T,S))
		edge_index = S.nonzero().t().contiguous()
		edge_index = np.array(edge_index)
		prob = []
		for i in range(edge_index.shape[1]):
			d1 = edge_index[0,i]
			d2 = edge_index[1,i]
			prob.append(D[d1]*D[d2]/sum)
		prob = np.array(prob,dtype='float64')
		prob = np.squeeze(prob)
		for i in range(no_samples):
			edges_to_remove = np.random.choice(edge_index.shape[1], total_purturbations,False,p=prob)
			edges_remain = np.setdiff1d(np.array(range(edge_index.shape[1])), edges_to_remove)
			edges_index = edge_index[:,edges_remain]
			S_prime = torch.zeros_like(S)
			for j in range(edges_index.shape[1]):
				n1 = edges_index[:,j][0]
				n2 = edges_index[:,j][1]
				S_prime[n1][n2] = 1
				S_prime[n2][n1] = 1
			purturbated_samples.append(S_prime)
	else:
		print("Probability model not defined.")
		exit()
	return purturbated_samples

def gen_netsmile(S):
	np_S = S.numpy()
	G = nx.from_numpy_array(np_S)
	feat = netsimile.feature_extraction(G)
	feat = torch.tensor(feat, dtype=torch.float)
	return feat

def load_npz(filepath):
	filepath = osp.abspath(osp.expanduser(filepath))
	if not filepath.endswith('.npz'):
		filepath = filepath + '.npz'
	if osp.isfile(filepath):
		with np.load(filepath, allow_pickle=True) as loader:
			loader = dict(loader)
			for k, v in loader.items():
				if v.dtype.kind in {'O', 'U'}:
					loader[k] = v.tolist()

			return loader
	else:
		raise ValueError(f"{filepath} doesn't exist.")

def load_douban():
	x = loadmat("data/douban.mat")
	return (x['online_edge_label'][0][1],
			x['online_node_label'],
			x['offline_edge_label'][0][1],
			x['offline_node_label'],
			x['ground_truth'].T)
#---
# Utils
#---
def info_data(data, include_vectors=False):
	print("Validate:\t {}".format(data.validate(raise_on_error=True)))
	print("Num. nodes:\t {}".format(data.num_nodes))
	print("Num. edges:\t {}".format(data.num_edges))
	print("Num. features:\t {}".format(data.num_node_features))
	print("Has isolated:\t {}".format(data.has_isolated_nodes()))
	print("Has loops:\t {}".format(data.has_self_loops()))
	print("Is directed:\t {}".format(data.is_directed()))
	print("Is undirected:\t {}".format(data.is_undirected()))
	if include_vectors:
		print("{}".format(data.edge_index))
		print("{}".format(data.x))
		print("{}".format(data.edge_attr))

def compute_num_neg_samples(edge_index, num_nodes, ratio):
	E = edge_index.size(1)
	max_neg = num_nodes * num_nodes - E
	return min(int(ratio * E), max_neg)

def neg_ratio_schedule(epoch, max_epoch):
	start = 5.0
	end = 1.0
	return start - (start - end) * (epoch / max_epoch)

class EarlyStopping:
	def __init__(self, patience=5, delta=0, warmup=5, verbose=False):
		self.patience = patience
		self.delta = delta
		self.warmup = warmup
		self.verbose = verbose
		self.best_loss = None
		self.no_improvement_count = 0
		self.stop_training = False
	
	def check_early_stop(self, loss, epoch):
		if epoch >= self.warmup:
			if self.best_loss is None or loss < self.best_loss - self.delta:
				self.best_loss = loss
				self.no_improvement_count = 0
			else:
				self.no_improvement_count += 1
				if self.no_improvement_count >= self.patience:
					self.stop_training = True
					if self.verbose:
						print("Stopping early as no improvement has been observed.")
#---
# Create data (PyG)
#---
def add_edge_attributes(data: Data) -> Data:
	"""
	Compute edge attributes:
		1. Common Neighbors
		2. Jaccard Similarity
		3. Adamic-Adar
		4. Feature Similarity (cosine)

	The resulting edge_attr has shape [num_edges, 4].
	"""

	edge_index = data.edge_index
	x = data.x

	num_nodes = data.num_nodes
	num_edges = edge_index.size(1)

	# ---------------------------------------------------------
	# 1. Build an undirected NetworkX graph
	# ---------------------------------------------------------
	G = nx.Graph()
	G.add_nodes_from(range(num_nodes))

	edges = edge_index.t().tolist()

	# Remove self-loops and duplicate edges
	G.add_edges_from(
		(u, v) for u, v in edges if u != v
	)

	# ---------------------------------------------------------
	# 2. Compute node degrees
	# ---------------------------------------------------------
	degree = dict(G.degree())

	# ---------------------------------------------------------
	# 3. Compute feature similarity
	# ---------------------------------------------------------
	if x is not None:

		# Normalize node feature vectors
		x_norm = F.normalize(x.float(), p=2, dim=1)

		# Cosine similarity for each edge
		u = edge_index[0]
		v = edge_index[1]

		feature_sim = (x_norm[u] * x_norm[v]).sum(dim=1)

	else:
		feature_sim = torch.zeros(
			num_edges,
			dtype=torch.float
		)

	# ---------------------------------------------------------
	# 4. Compute structural edge attributes
	# ---------------------------------------------------------
	cn_values = []
	jaccard_values = []
	aa_values = []

	for u, v in edges:

		# Self-loops
		if u == v:
			cn = 0.0
			jaccard = 0.0
			aa = 0.0

		else:

			# Common neighbors
			common = set(nx.common_neighbors(G, u, v))
			cn = float(len(common))

			# Jaccard similarity
			neighbors_u = set(G.neighbors(u))
			neighbors_v = set(G.neighbors(v))

			union = neighbors_u | neighbors_v

			if len(union) > 0:
				jaccard = len(common) / len(union)
			else:
				jaccard = 0.0

			# Adamic-Adar
			aa = 0.0

			for z in common:
				deg_z = degree[z]

				if deg_z > 1:
					aa += 1.0 / torch.log(
						torch.tensor(float(deg_z))
					).item()

		cn_values.append(cn)
		jaccard_values.append(jaccard)
		aa_values.append(aa)

	# ---------------------------------------------------------
	# 5. Convert structural attributes to tensors
	# ---------------------------------------------------------
	cn = torch.tensor(
		cn_values,
		dtype=torch.float
	)

	jaccard = torch.tensor(
		jaccard_values,
		dtype=torch.float
	)

	adamic_adar = torch.tensor(
		aa_values,
		dtype=torch.float
	)

	# ---------------------------------------------------------
	# 6. Combine all edge attributes
	# ---------------------------------------------------------
	edge_attr = torch.stack(
		[
			cn,
			jaccard,
			adamic_adar,
			feature_sim
		],
		dim=1
	)

	# ---------------------------------------------------------
	# 7. Store in the PyG Data object
	# ---------------------------------------------------------
	data.edge_attr = edge_attr

	return data

def fit_edge_normalization(data1, data2, eps=1e-8):

	edge_attr = torch.cat(
		[data1.edge_attr, data2.edge_attr],
		dim=0
	).float()

	mean = edge_attr.mean(dim=0, keepdim=True)
	std = edge_attr.std(dim=0, keepdim=True)

	std = std.clamp_min(eps)

	return mean, std

def apply_edge_normalization(data, mean, std):

	data.edge_attr = (
		data.edge_attr.float() - mean
	) / std

	return data

def normalize_node_edge_features(data1, data2, with_mean1, with_mean2):
	node_scaler = StandardScaler(with_mean=with_mean1)
	edge_scaler = StandardScaler(with_mean=with_mean2)

	# -------------------------
	# Node features
	# -------------------------
	x_all = np.vstack([
		data1.x.cpu().numpy(),
		data2.x.cpu().numpy()
	])

	node_scaler.fit(x_all)

	data1.x = torch.tensor(
		node_scaler.transform(data1.x.cpu().numpy()),
		dtype=torch.float
	)

	data2.x = torch.tensor(
		node_scaler.transform(data2.x.cpu().numpy()),
		dtype=torch.float
	)

	# -------------------------
	# Edge attributes
	# -------------------------
	edge_all = np.vstack([
		data1.edge_attr.cpu().numpy(),
		data2.edge_attr.cpu().numpy()
	])

	edge_scaler.fit(edge_all)

	data1.edge_attr = torch.tensor(
		edge_scaler.transform(data1.edge_attr.cpu().numpy()),
		dtype=torch.float
	)

	data2.edge_attr = torch.tensor(
		edge_scaler.transform(data2.edge_attr.cpu().numpy()),
		dtype=torch.float
	)

# Based on PlanetAlign
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

def mrr_score(similarity: torch.Tensor,
			test_pairs: torch.Tensor,
			mode: str = 'mean') -> float:
	r"""Mean Reciprocal Rank (MRR) score of pairwise alignment results.

	Parameters
	----------
	similarity : torch.Tensor
		Similarity matrix of shape (n1, n2) where n1 and n2 are the number of nodes in graph1 and graph2.
	test_pairs : torch.Tensor
		Test pairs of shape (m, 2) where m is the number of test pairs.
	mode : str, optional
		Mode for MRR score. Options are 'mean', 'max', 'ltr' (left-to-right), 'rtl' (right-to-left). Default is 'mean'.

	"""

	if mode == 'mean':
		mrr = mrr_mean_score(similarity, test_pairs)
	elif mode == 'max':
		mrr = mrr_max_score(similarity, test_pairs)
	elif mode == 'ltr':
		mrr = mrr_ltr_score(similarity, test_pairs)
	elif mode == 'rtl':
		mrr = mrr_rtl_score(similarity, test_pairs)
	else:
		raise ValueError(f"Invalid mode: {mode}")
	return mrr


def mrr_ltr_score(similarity, test_pairs):
	r"""Mean Reciprocal Rank (MRR) score of graph1(left) to graph2(right) alignment."""
	test_pairs = test_pairs.to(similarity.device)
	ranks1 = torch.argsort(-similarity[test_pairs[:, 0]], dim=1)
	signal1_hit = ranks1 == test_pairs[:, 1].view(-1, 1)
	mrr = torch.mean(1 / (torch.where(signal1_hit)[1].float() + 1)).item()
	return mrr


def mrr_rtl_score(similarity, test_pairs):
	r"""Mean Reciprocal Rank (MRR) score of graph2(right) to graph1(left) alignment."""
	test_pairs = test_pairs.to(similarity.device)
	ranks2 = torch.argsort(-similarity.T[test_pairs[:, 1]], dim=1)
	signal2_hit = ranks2 == test_pairs[:, 0].view(-1, 1)
	mrr = torch.mean(1 / (torch.where(signal2_hit)[1].float() + 1)).item()
	return mrr


def mrr_max_score(similarity, test_pairs):
	r"""Max Mean Reciprocal Rank (MRR) score of left-to-right and right-to-left alignments."""
	mrr_ltr = mrr_ltr_score(similarity, test_pairs)
	mrr_rtl = mrr_rtl_score(similarity, test_pairs)
	mrr = max(mrr_ltr, mrr_rtl)

	return mrr


def mrr_mean_score(similarity, test_pairs):
	r"""Mean Mean Reciprocal Rank (MRR) score of left-to-right and right-to-left alignments."""
	mrr_ltr = mrr_ltr_score(similarity, test_pairs)
	mrr_rtl = mrr_rtl_score(similarity, test_pairs)
	mrr = (mrr_ltr + mrr_rtl) / 2

	return mrr

from typing import Union
import torch


def hits_ks_scores(simiarity: torch.Tensor,
				   test_pairs: torch.Tensor,
				   ks: Union[list[int], tuple[int, ...]] = (1, 10, 30, 50),
				   mode: str = 'mean') -> dict[int, float]:
	r"""Hits@K scores of pairwise alignment results.

	Parameters
	----------
	simiarity : torch.Tensor
		Similarity matrix of shape (n1, n2) where n1 and n2 are the number of nodes in graph1 and graph2.
	test_pairs : torch.Tensor
		Test pairs of shape (m, 2) where m is the number of test pairs.
	ks : list[int] or tuple[int, ...], optional
		List of k values for Hits@K scores. Default is (1, 10, 30, 50).
	mode : str, optional
		Mode for Hits@K scores. Options are 'mean', 'max', 'ltr' (left-to-right), 'rtl' (right-to-left). Default is 'mean'.
	"""

	if mode == 'mean':
		hits_ks = hits_ks_mean_scores(simiarity, test_pairs, ks=ks)
	elif mode == 'max':
		hits_ks = hits_ks_max_scores(simiarity, test_pairs, ks=ks)
	elif mode == 'ltr':
		hits_ks = hits_ks_ltr_scores(simiarity, test_pairs, ks=ks)
	elif mode == 'rtl':
		hits_ks = hits_ks_rtl_scores(simiarity, test_pairs, ks=ks)
	else:
		raise ValueError(f"Invalid mode: {mode}")
	return hits_ks


def hits_ks_ltr_scores(similarity, test_pairs, ks=None):
	r"""Hits@K scores of graph1(left) to graph2(right) alignment."""
	test_pairs = test_pairs.to(similarity.device)
	hits_ks = {}
	ranks1 = torch.argsort(-similarity[test_pairs[:, 0]], dim=1)
	signal1_hit = ranks1 == test_pairs[:, 1].view(-1, 1)
	for k in ks:
		hits_ks[k] = (torch.sum(signal1_hit[:, :k]) / test_pairs.shape[0]).item()

	return hits_ks


def hits_ks_rtl_scores(similarity, test_pairs, ks=None):
	r"""Hits@K scores of graph2(right) to graph1(left) alignment."""
	test_pairs = test_pairs.to(similarity.device)
	hits_ks = {}
	ranks2 = torch.argsort(-similarity.T[test_pairs[:, 1]], dim=1)
	signal2_hit = ranks2 == test_pairs[:, 0].view(-1, 1)
	for k in ks:
		hits_ks[k] = (torch.sum(signal2_hit[:, :k]) / test_pairs.shape[0]).item()

	return hits_ks


def hits_ks_max_scores(similarity, test_pairs, ks=None):
	r"""Max Hits@K scores of left-to-right and right-to-left alignments."""
	hits_ks = {}

	hits_ks_ltr = hits_ks_ltr_scores(similarity, test_pairs, ks=ks)
	hits_ks_rtl = hits_ks_rtl_scores(similarity, test_pairs, ks=ks)
	for k in ks:
		hits_ks[k] = max(hits_ks_ltr[k], hits_ks_rtl[k])

	return hits_ks


def hits_ks_mean_scores(similarity, test_pairs, ks=None):
	r"""Mean Hits@K scores of left-to-right and right-to-left alignments."""
	hits_ks = {}

	hits_ks_ltr = hits_ks_ltr_scores(similarity, test_pairs, ks=ks)
	hits_ks_rtl = hits_ks_rtl_scores(similarity, test_pairs, ks=ks)
	for k in ks:
		hits_ks[k] = (hits_ks_ltr[k] + hits_ks_rtl[k]) / 2

	return hits_ks


def get_normalized_neg_exp_dist(emb1, emb2, p=2, device='cpu'):
	emb1 = emb1.to(device)
	emb2 = emb2.to(device)
	normalized_emb1 = F.normalize(emb1, p=p, dim=1)
	normalized_emb2 = F.normalize(emb2, p=p, dim=1)
	return torch.exp(-(normalized_emb1 @ normalized_emb2.T))


def pairwise_cosine_similarity(x, y, p=2):
	normalized_x = F.normalize(x, p=p, dim=1)
	normalized_y = F.normalize(y, p=p, dim=1)
	return normalized_x @ normalized_y.T