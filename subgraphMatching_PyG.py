#!/usr/bin/env python
# coding: utf-8

# Experiments for search best features

# In[1]:


# Torch version
# get_ipython().system('python -c "import torch; print(torch.__version__)"')

# Cuda version
# get_ipython().system('python -c "import torch; print(torch.version.cuda)"')


# In[2]:


# Uninstall
# !pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib -y


# In[3]:


# Update Torch
# !pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124


# In[4]:


# Install PyG (automatic)
# !pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-{torch.__version__}.html
# !pip install torch_geometric


# In[5]:


# Verify instalation
import torch
import torch_geometric
import torch_scatter

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch_scatter.__version__)
print(torch_geometric.__version__)


# In[6]:


from model_PyG import *
from utils import *


# In[7]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import Compose
from torch_geometric.utils import dense_to_sparse, negative_sampling
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import Adam


# In[8]:


import torch_geometric
print(torch_geometric.__version__)




# In[9]:


def info(data):
	print("Validate:\t {}".format(data.validate(raise_on_error=True)))
	print("Num. nodes:\t {}".format(data.num_nodes))
	print("Num. edges:\t {}".format(data.num_edges))
	print("Num. features:\t {}".format(data.num_node_features))
	print("Has isolated:\t {}".format(data.has_isolated_nodes()))
	print("Has loops:\t {}".format(data.has_self_loops()))
	print("Is directed:\t {}".format(data.is_directed()))
	print("Is undirected:\t {}".format(data.is_undirected()))
	print("{}".format(data.edge_index))
	print("{}".format(data.x))
	print("{}".format(data.edge_attr))


# ### Setup

# In[10]:


dataset = "mentos_05" # "vanessa_05", "mentos_05", "Douban Online_Offline", "ACM_DBLP" # args.dataset
encoder = "GIN" # Change GIN, GINE
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if (dataset == "ACM_DBLP"):
	train_set = ["ACM", "DBLP"]
	b = np.load("data/ACM-DBLP.npz")
	# train_features["ACM"] = [torch.from_numpy(b["x1"]).float()]
	# train_features["DBLP"] = [torch.from_numpy(b["x2"]).float()]
	test_pairs = b["test_pairs"].astype(np.int32)
	NUM_HIDDEN_LAYERS = 12
	HIDDEN_DIM = [1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024]
	# input_dim = 17
	output_feature_size = 128
	lr = 1e-4 # 1e-4
	epochs = 100
elif (dataset == "Douban Online_Offline"):
	train_set = ["Online", "Offline"]
	a1, f1, a2, f2, test_pairs = load_douban()
	# f1 = f1.A
	# f2 = f2.A
	test_pairs = torch.tensor(np.array(test_pairs, dtype=int)) - 1
	test_pairs = test_pairs.numpy()
	# train_features["Online"] = [torch.from_numpy(f1).float()]
	# train_features["Offline"] = [torch.from_numpy(f2).float()]
	NUM_HIDDEN_LAYERS = 6
	HIDDEN_DIM = [512, 512, 512, 512, 512, 512, 512]
	# input_dim = 538
	output_feature_size = 512
	lr = 0.0001
	epochs = 100
elif (dataset == "mentos_05"):
	train_set = [
		# "Orange_1", "Orange_2",
		"Red_1", "Red_2",
		# "Yellow_1", "Yellow_2"
	]
	NUM_HIDDEN_LAYERS = 12
	HIDDEN_DIM = [1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024]
	output_feature_size = 128
	lr = 0.0001
	epochs = 100
elif (dataset == "vanessa_05"):
	train_set = [
		"FrescoAmazonas_1", "FrescoAmazonas_2",
		# "FrescoCusco_1", "FrescoCusco_2",
		# "FrescoSanMartin_1", "FrescoSanMartin_2",
		# "SecoAmazonas_1", "SecoAmazonas_2",
		# "SecoCusco_1", "SecoCusco_2",
		# "SecoSanMartin_1", "SecoSanMartin_2"
	]
	NUM_HIDDEN_LAYERS = 12
	HIDDEN_DIM = [1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024]
	output_feature_size = 128
	lr = 0.0001
	epochs = 150



# Only for GIN
""" transform = Compose([
	# T.NormalizeFeatures(),
	T.ToUndirected(reduce="mean"),
	T.AddSelfLoops(fill_value=1.0),
	T.ToDevice(device)
]) """

# For GIN and GINE
transform = T.Compose([
	# T.NormalizeFeatures(),
	T.ToUndirected(reduce="mean"),
	T.AddSelfLoops(attr="edge_attr", fill_value="mean"),
	T.ToDevice(device)
])


# In[12]:

from itertools import combinations

index_features = [0, 1, 2, 3, 4, 5]
list_features_all = [list(c) for r in range(1, len(index_features) + 1) for c in combinations(index_features, r)]
print(len(list_features_all))

list_features = []
for item in list_features_all:
    if len(item) > 2:
        list_features.append(item)
print(len(list_features))
list_features = [
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3],
    [0, 1, 2],
    [0, 1, 2, 4, 5],
    [2, 3, 4, 5],
    [2, 4, 5]]

for features in list_features:
	try:
		print("Loading training datasets")

		np.random.seed(0)
		torch.manual_seed(0)
		torch.cuda.manual_seed(0)
		torch.cuda.manual_seed_all(0)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

		train_loader = {}
		# scaler = StandardScaler()

		if dataset == "ACM_DBLP":
			for i, ts in enumerate(train_set):
				edge_index = torch.tensor(b[f"edge_index{i+1}"], dtype=torch.long)
				x = torch.tensor(b[f"x{i+1}"], dtype=torch.float)
				# x = torch.tensor(scaler.fit_transform(x.numpy())) # scaling
				
				""" if i==1:
					x = x[torch.randperm(x.size(0))] # permutations for test """

				edge_attr = torch.ones((edge_index.size(1), 1)) # Only for test GINE

				data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
				info(data)

				data = transform(data)
				train_loader[ts] = data
				info(data)
		elif dataset == "Douban Online_Offline":
			edge_index1, _= dense_to_sparse(torch.from_numpy(a1.toarray()))
			x1 = torch.from_numpy(f1.toarray()).float()
			data1 = Data(x=x1, edge_index=edge_index1)
			data1 = transform(data1)
			train_loader[train_set[0]] = data1
			info(data1)

			edge_index2, _= dense_to_sparse(torch.from_numpy(a2.toarray()))
			x2 = torch.from_numpy(f2.toarray()).float()
			data2 = Data(x=x2, edge_index=edge_index2)
			data2 = transform(data2)
			train_loader[train_set[1]] = data2
			info(data2)
		elif dataset in ["vanessa_05", "mentos_05"]: # Change
			for ts in train_set:
				df_nodes = pd.read_csv("data/{}/nodes_{}.csv".format(dataset, ts))
				# idx, id, mz, rt, 0, 1, 2, ...

				df_intensity = df_nodes.iloc[:, 4:]
				# 0, 1, 2, ...

				df_edges = pd.read_csv("data/{}/edges_{}.csv".format(dataset, ts))
				# source, target, weight, subgroup

				# Node features
				""" mz = np.log10(df_nodes["mz"].values + 1e-8) # Log-transform m/z to stabilize scale differences
				rt = df_nodes["rt"].values
				rt = (rt - rt.mean()) / (rt.std() + 1e-8) # Z-score normalization for retention time (RT)

				intensity = df_intensity.values.astype(np.float32)
				intensity = intensity / (intensity.sum(axis=0, keepdims=True) + 1e-8)
				intensity_mean = intensity.mean(axis=1)

				intensity_std = intensity.std(axis=1)
				intensity_cv = intensity_std / (intensity_mean + 1e-8)
				intensity_cv = np.log1p(np.clip(intensity_cv, 0, 2.0))

				presence_ratio = (df_intensity > 0).mean(axis=1) # Acts as a reliability / confidence signal """

				# df_intensity = np.sign(df_intensity) * np.log10(np.abs(df_intensity) + 1e-8)

				mz = np.log10(df_nodes.iloc[:, 2].values + 1e-8)
				rt = df_nodes.iloc[:, 3].values
				rt = (rt - rt.mean()) / (rt.std() + 1e-8)
				intensity_mean = df_intensity.mean(axis=1).values
				intensity_std = df_intensity.std(axis=1).values
				intensity_cv = intensity_std / (intensity_mean + 1e-9)
				presence_ratio = (df_intensity > 0).mean(axis=1)

				x = np.stack([
						mz,             			# 0 physicochemical identity
						rt,							# 1 chromatographic alignment
						np.log10(intensity_mean), 	# 2 global abundance
						intensity_std,  			# 3
						intensity_cv,   			# 4 robustness (anti-oversmoothing)
						presence_ratio  			# 5 reliability
					], axis=1)

				x = torch.tensor(x[:, features], dtype=torch.float) # [N, F]
				# x = torch.tensor(scaler.fit_transform(x.numpy())) # scaling

				# Edge index
				edge_index = torch.tensor(df_edges.iloc[:, [0, 1]].values.T, dtype=torch.long) # [2, E]

				# Edge attribute
				edge_weight = torch.tensor(df_edges.iloc[:, 2].values, dtype=torch.float) # [E]

				# edge_attr = edge_weight.view(-1, 1) # [E,1]
				edge_attr = torch.stack([
									edge_weight.abs(),        # strength
									torch.sign(edge_weight),  # direction (+1, -1)
									edge_weight ** 2          # nonlinearity
								], dim=1) # [E, 3]

				# Reduce number of edges
				""" mask = torch.abs(edge_weight) > 0.95
				edge_index = edge_index[:, mask]
				edge_attr  = edge_attr[mask] """

				data = Data(x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr)

				data = transform(data)
				# data = data.to(device)

				train_loader[ts] = data
				info(data)

				test_pairs = None # No use


		# In[13]:


		train_loader


		# ### Train

		# In[14]:


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


		# In[15]:


		def fit_TGAE_subgraph(encoder, dataset, no_samples, model, epochs, train_loader, lr, test_pairs=None):
			best_hitAtOne = 0
			best_hitAtFive = 0
			best_hitAtTen = 0
			best_hitAtFifty = 0
			list_loss = []

			optimizer = Adam(model.parameters(), lr=lr,weight_decay=5e-4)
			
			# Initialize early stopping
			patience = 10
			delta = 1e-4 # 1e-4
			warmup = 10
			early_stopping = EarlyStopping(patience=patience, delta=delta, warmup=warmup, verbose=True)

			loop_obj = tqdm(range(1, epochs + 1))
			for epoch in loop_obj:
				loop_obj.set_description(f"Epoch: {epoch}")
				
				# Train
				model.train()
				loss = 0.0
				
				for ts in random.sample(train_set, k=len(train_set)): # shuffle train_set
					data = train_loader[ts]

					# Encoder
					if encoder == "GIN":
						z = model(data.x, data.edge_index)
						# z = F.normalize(z, dim=1)
					elif encoder == "GINE":
						z = model(data.x, data.edge_index, data.edge_attr)

					# Positive edges
					pos_edge_index = data.edge_index
					
					# Negative edges
					# option 1
					neg_edge_index = negative_sampling(
						edge_index=data.edge_index,
						num_nodes=z.size(0),
						num_neg_samples=pos_edge_index.size(1), # Change 2 to other value if needed
						method="sparse"
					)

					# option 2 Negative edges (dynamic)
					""" ratio = neg_ratio_schedule(epoch, epochs)
					num_neg = compute_num_neg_samples(
						edge_index=edge_index,
						num_nodes=z.size(0),
						ratio=ratio
					)
					neg_edge_index = negative_sampling(
						edge_index=edge_index,
						num_nodes=z.size(0),
						num_neg_samples=num_neg,
						method="sparse"
					) """
					
					# Decoder
					# option 1
					pos_logits = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
					neg_logits = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
					
					# option 2
					""" pos_logits = F.cosine_similarity(
						z[pos_edge_index[0]],
						z[pos_edge_index[1]],
						dim=1
					)
					neg_logits = F.cosine_similarity(
						z[neg_edge_index[0]],
						z[neg_edge_index[1]],
						dim=1
					) """

					# Loss
					pos_labels = torch.ones_like(pos_logits)
					neg_labels = torch.zeros_like(neg_logits)

					# option 1
					""" loss_pos = binary_cross_entropy_with_logits(pos_logits, pos_labels)
					loss_neg = binary_cross_entropy_with_logits(neg_logits, neg_labels)
					loss += loss_pos + loss_neg """

					# option 2
					# num_pos = pos_edge_index.size(1)
					# num_neg = neg_edge_index.size(1)
					# pos_weight = torch.tensor([num_neg / num_pos], device=device)
					logits = torch.cat([pos_logits, neg_logits], dim=0)
					labels = torch.cat([pos_labels, neg_labels], dim=0)
					loss_temp = F.binary_cross_entropy_with_logits(logits, labels) #, pos_weight=pos_weight) # with pos_weight
					loss += loss_temp
					
				optimizer.zero_grad()
				loss = loss / no_samples
				loss.backward()
				optimizer.step()

				loop_obj.set_postfix_str(f"Loss: {loss.item():.4f}")
				list_loss.append(loss.item())

				# Check early stopping condition
				early_stopping.check_early_stop(loss.item(), epoch)
				if early_stopping.stop_training:
					print(f"Early stopping at epoch {epoch}")
					break

				# Evaluation (for firts dataset)
				""" model.eval()
				with torch.no_grad():
					keys = list(train_loader.keys())
					data1 = train_loader[keys[0]]
					data2 = train_loader[keys[1]]

					z1 = model(data1.x, data1.edge_index).detach()
					z2 = model(data2.x, data2.edge_index).detach()
					
					# Similarity matrix
					# option 1
					D = torch.cdist(z1, z2, 2)

					# option 2 (GPU problem)
					# D = 1 - F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)

					# option 3 (Decoder cosine similarity)
					" "" z1n = F.normalize(z1, dim=1)
					z2n = F.normalize(z2, dim=1)
					D = 1 - (z1n @ z2n.T) " ""

					if dataset == "ACM_DBLP":
						test_idx = test_pairs[:, 0].astype(int)
						labels = test_pairs[:, 1].astype(int)
					else:
						test_idx = test_pairs[0, :].astype(int)
						labels = test_pairs[1, :].astype(int)
						
					hitAtOne = 0
					hitAtFive = 0
					hitAtTen = 0
					hitAtFifty = 0
					hitAtHundred = 0
					for i in range(len(test_idx)):
						dist_list = D[test_idx[i]]
						sorted_neighbors = torch.argsort(dist_list).cpu()
						label = labels[i]
						for j in range(100):
							if (sorted_neighbors[j].item() == label):
								if (j == 0):
									hitAtOne += 1
									hitAtFive += 1
									hitAtTen += 1
									hitAtFifty += 1
									hitAtHundred += 1
									break
								elif (j <= 4):
									hitAtFive += 1
									hitAtTen += 1
									hitAtFifty += 1
									hitAtHundred += 1
									break
								elif (j <= 9):
									hitAtTen += 1
									hitAtFifty += 1
									hitAtHundred += 1
									break
								elif (j <= 49):
									hitAtFifty += 1
									hitAtHundred += 1
									break
								elif (j <= 100):
									hitAtHundred += 1
									break
					cur_hitAtOne = hitAtOne / len(test_idx)
					cur_hitAtFive = hitAtFive / len(test_idx)
					cur_hitAtTen = hitAtTen / len(test_idx)
					cur_hitAtFifty = hitAtFifty / len(test_idx)

					if(cur_hitAtOne > best_hitAtOne): best_hitAtOne = cur_hitAtOne
					if (cur_hitAtFive > best_hitAtFive): best_hitAtFive = cur_hitAtFive
					if (cur_hitAtTen > best_hitAtTen): best_hitAtTen = cur_hitAtTen
					if (cur_hitAtFifty > best_hitAtFifty): best_hitAtFifty = cur_hitAtFifty

			print("The best results achieved:")
			print("Hit@1: ", end="")
			print(best_hitAtOne)
			print("Hit@5: ", end="")
			print(best_hitAtFive)
			print("Hit@10: ", end="")
			print(best_hitAtTen)
			print("Hit@50: ", end="")
			print(best_hitAtFifty) """

			# Evaluation (for others dataset)
			dict_node_embeddings = {}
			model.eval()
			with torch.no_grad():
				for ts in train_set:
					data = train_loader[ts]
					if encoder == "GIN":
						z = model(data.x, data.edge_index)
					elif encoder == "GINE":
						z = model(data.x, data.edge_index, data.edge_attr)
					dict_node_embeddings[ts] = z.cpu().numpy()

			del loss, z
			# torch.cuda.synchronize()
			torch.cuda.empty_cache()
			
			return dict_node_embeddings, list_loss


		# In[16]:


		train_set


		# In[17]:


		no_samples = len(train_set) # * (1 + 1)  # num datasets * num of samples by dataset 
		input_dim = train_loader[train_set[0]].num_node_features

		if encoder == "GIN":
			model = TGAE_GIN(NUM_HIDDEN_LAYERS,
						input_dim,
						HIDDEN_DIM,
						output_feature_size).to(device)
		elif encoder == "GINE":
			edge_dim = train_loader[train_set[0]].edge_attr.size(1)

			model = TGAE_GINE(NUM_HIDDEN_LAYERS,
						input_dim,
						HIDDEN_DIM,
						output_feature_size, edge_dim).to(device)

		print("Generating training features")
		print("Fitting model")
		print(encoder, dataset, lr, epochs, input_dim, output_feature_size, no_samples)

		dict_node_embeddings, list_loss = fit_TGAE_subgraph(encoder, dataset, no_samples, model, epochs, train_loader, lr, test_pairs)


		# ### Get embeddings

		# In[18]:


		dict_node_embeddings


		# ### Plot

		# In[19]:


		# Concatenate embeddings

		node_embeddings_cat = np.concatenate(list(dict_node_embeddings.values()), axis=0)
		print(node_embeddings_cat.shape)
		node_embeddings_cat


		# In[20]:


		# Get labels

		labels = []
		for i, node_embeddings in enumerate(list(dict_node_embeddings.values())):
			labels += [i] * node_embeddings.shape[0]
		print(len(labels))
		# print(labels)


		# In[21]:


		list_loss


		# In[22]:


		# Loss

		plt.figure()
		plt.plot(range(1, len(list_loss) + 1), list_loss) #, marker=".")
		# plt.plot(range(1, len(list_loss) + 1), np.log(list_loss)) #, marker=".")
		plt.title("Training Loss over Epochs")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.tight_layout()
		plt.savefig(f"data/{dataset}/output/plot/loss_{encoder}_{str(train_set)}_{features}.pdf", format="pdf", bbox_inches="tight")
		plt.show()


		# In[23]:


		# Node embeddings 3D

		""" if node_embeddings_cat.shape[1] > 3:
			pca = PCA(n_components=3)
			node_embeddings_cat_3d = pca.fit_transform(node_embeddings_cat)
		else:
			node_embeddings_cat_3d = node_embeddings_cat.copy()

		fig = plt.figure()
		ax = fig.add_subplot(projection="3d")

		for c in np.unique(labels):
			ax.scatter(
				node_embeddings_cat_3d[:, 0][labels == c],
				node_embeddings_cat_3d[:, 1][labels == c], 
				node_embeddings_cat_3d[:, 2][labels == c],
				s=10,
				alpha=0.5,
				label=f"{train_set[c]}"
			)

		ax.set_xlabel("X")
		ax.set_ylabel("Y")
		ax.set_zlabel("Z")
		ax.legend()
		plt.tight_layout()
		plt.show() """


		# In[24]:


		# Node embeddings 2D

		if node_embeddings_cat.shape[1] > 2:
			pca = PCA(n_components=2)
			node_embeddings_cat_2d = pca.fit_transform(node_embeddings_cat)
		else:
			node_embeddings_cat_2d = node_embeddings_cat.copy()

		fig, ax = plt.subplots()

		for c in np.unique(labels):
			idx = labels == c
			ax.scatter(
				node_embeddings_cat_2d[idx, 0],
				node_embeddings_cat_2d[idx, 1],
				s=10,
				alpha=0.5,
				label=f"{train_set[c]}"
			)

		ax.set_xlabel("X")
		ax.set_ylabel("Y")
		ax.legend()
		plt.tight_layout()
		plt.savefig(f"data/{dataset}/output/plot/node_embeddings_{encoder}_{str(train_set)}_{features}.pdf", format="pdf", bbox_inches="tight")
		plt.show()


		# ### Similarity analysis (KNN)

		# In[25]:


		train_set


		# In[26]:


		# Get node ids

		dict_node_id = {}

		if dataset in ["vanessa_05", "mentos_05"]:
			for ts in train_set:
				df_nodes = pd.read_csv("data/{}/nodes_{}.csv".format(dataset, ts))
				# idx,id,mz,rt,intensity_mean,intensity_cv

				dict_node_id[ts] = df_nodes["id"].values
		else:
			for ts in train_set:
				dict_node_id[ts] = np.arange(len(dict_node_embeddings[ts]))
		dict_node_id


		# In[27]:


		# Calculate distance matrix (KNN)

		k = 1 # Change
		knn = NearestNeighbors(n_neighbors=k, metric="euclidean")

		first_ts = train_set[0]
		x = dict_node_embeddings[first_ts]

		df_node_alignment = pd.DataFrame()
		df_node_alignment[first_ts] = dict_node_id[first_ts]

		for ts in train_set[1:]:
			y = dict_node_embeddings[ts]
			
			knn.fit(y)
			distances, indices = knn.kneighbors(x)
			
			df_node_alignment[ts] = dict_node_id[ts][indices]
		df_node_alignment


		# In[28]:


		# Find node alignment 2 by 2

		col1, col2 = train_set[:2] # Change
		print(col1, col2)

		df_node_alignment_filter = df_node_alignment[df_node_alignment.apply(lambda row: row[col1] == row[col2], axis=1)]
		df_node_alignment_filter


		# In[29]:


		# Find node alignment for all datasets

		df_node_alignment_filter = df_node_alignment[df_node_alignment.nunique(axis=1) == 1]
		print(len(df_node_alignment_filter))
		df_node_alignment_filter


		# In[30]:


		# Comparison (with test_pairs)

		if dataset not in ["vanessa_05", "mentos_05"]:
			print(len(test_pairs))
			# print(test_pairs)
			# print(df_node_alignment.values)
			mask = np.array([tuple(row) in map(tuple, test_pairs) for row in df_node_alignment.values])
			df_node_alignment["mask"] = mask
			print(df_node_alignment[df_node_alignment["mask"] == True])


		# ### Filter MS data

		# In[31]:


		common_node_id = df_node_alignment_filter.iloc[:, 0].values
		common_node_id


		# In[32]:


		# Read raw data

		df_join_raw = pd.read_csv("data/{}/raw.csv".format(dataset), index_col=0)
		df_join_raw


		# In[33]:


		print(len(common_node_id), len(df_join_raw))


		# In[34]:


		df_join_raw_filter = df_join_raw.loc[common_node_id].iloc[:, [0, 1, 2]]
		df_join_raw_filter.to_csv(f"data/{dataset}/output/node_alignment.csv", sep=";", decimal=",", index_label="Id")
		df_join_raw_filter


		# In[35]:


		# Comparison (sta vs Van)

		list_node_id_sta = [39, 52, 70, 79, 94, 91, 90, 116, 123, 126, 127, 159, 157, 160, 175, 188, 190, 189, 173, 205, 202, 211, 212]

		match = set(list_node_id_sta) & set(common_node_id)
		print(train_set)
		print(f"Alignment: {len(common_node_id)} / {len(df_join_raw)}")
		print(f"Match comp: {len(match)}/{len(list_node_id_sta)}")
		print(match)


		# ### Clustering analysis

		# In[36]:


		df_join_raw


		# In[37]:


		df_join_raw_signal = df_join_raw.loc[common_node_id].iloc[:, 3:-2] # Important two last column no only to Mentos
		df_join_raw_signal


		# In[38]:


		df_join_raw_signal_t = df_join_raw_signal.T
		df_join_raw_signal_t


		# In[39]:


		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(df_join_raw_signal_t.values)
		X_scaled


		# In[40]:


		pca = PCA(n_components=2)
		X_pca = pca.fit_transform(X_scaled)
		X_pca


		# In[41]:


		labels = [item.split("_")[0] for item in df_join_raw_signal_t.index]
		labels


		# In[47]:


		x, y = X_pca[:, 0], X_pca[:, 1]

		unique_groups = ["Red", "Orange", "Yellow"] # np.unique(labels)

		plt.figure()
		for group in unique_groups:
			xi = [x[i] for i in range(len(x)) if labels[i] == group]
			yi = [y[i] for i in range(len(y)) if labels[i] == group]
			plt.scatter(xi, yi, label=group)

		plt.legend()
		plt.xlabel("C1")
		plt.ylabel("C2")
		plt.title(f"Clustering {train_set}")
		plt.tight_layout()
		plt.savefig(f"data/{dataset}/output/plot/clustering_{encoder}_{str(train_set)}_{features}.pdf", format="pdf", bbox_inches="tight")
		plt.show()



		# In[45]:


		# import torch
		# print(torch.cuda.memory_summary())

	except Exception as e:
		print("Error")