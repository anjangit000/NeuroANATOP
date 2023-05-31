from nilearn import datasets, input_data, connectome, plotting
import supereeg as se
import numpy as np
import networkx as nx
from intbitset import *
import math
import random
import pickle
from anjanUtil import getNetworkxGraphFromAdj, getNetworkxGraphFromAdjByValue
from myData import fetch_adni_data
import matplotlib.pyplot as plt
import networkx.algorithms.community as nx_comm
from argparse import ArgumentParser
import random
from gensim.models import Word2Vec
from numpy import dot
from numpy.linalg import norm
import scipy.sparse as sp
import scipy.sparse.linalg as slinalg
import scipy.linalg as linalg
import node2vec
import collections
from tqdm import trange
from line import *
from gnnModels import *

def absorption_probability(W, alpha=1e-6):
	n = W.shape[0]
	print('Calculate absorption probability...')
	W = W.copy().astype(np.float32)
	D = W.sum(1).flat
	L = sp.diags(D, dtype=np.float32) - W
	L += alpha * sp.eye(W.shape[0], dtype=L.dtype)
	L = sp.csc_matrix(L)
	A = slinalg.inv(L).toarray()
	return A

def getParwalk(G, path_length=10):
	adj = nx.adjacency_matrix(G)
	A = absorption_probability(adj, alpha=1e-6)
	all_indices = []
	for n in G.nodes():
		oneHotVec = np.array([0 for i in range(G.number_of_nodes())])	
		oneHotVec[n] = 1
		a = A.dot(oneHotVec)
		gate = (-np.sort(-a, axis=0))[path_length]
		index = np.where(a.flat > gate)[0]
		all_indices.append([str(node) for node in index])
	return all_indices

def get_randomwalk(G, node, path_length):
    
    random_walk = [str(node)]
    
    for i in range(path_length-1):
        temp = list(G.neighbors(node))
        temp = list(set(temp) - set(random_walk))    
        if len(temp) == 0:
            break

        random_node = random.choice(temp)
        random_walk.append(str(random_node))
        node = random_node
        
    return random_walk


def get_emb_vector_similarity(G_emb_list,u,v):
	cos_sim = 0
	if u in G_emb_list and v in G_emb_list:
		emb_vector1 = G_emb_list[u]
		emb_vector2 = G_emb_list[v]
		cos_sim = dot(emb_vector1, emb_vector2) / (norm(emb_vector1) * norm(emb_vector2))

	return cos_sim


def getPL_mn(km, kn, E):
	num = 1
	den = 1
	for i in range(kn):
		num = num * (E-km-i+1)
		den = den * (E-i+1)

	PL_mn = 1 - (num/den)
	return 1/PL_mn

def getPL_mn_z(G,z):
	PL_mn_z = nx.clustering(G, z)
	return PL_mn_z 

#Computing all nrighborhood pair for a node
def getAllNbdPairs(G,z):
	nbd_pair_dic = {}
	for u in G.neighbors(z):
		for v in G.neighbors(z):
			if u < v:
				nbd_pair_dic[(u,v)] = 1
	return nbd_pair_dic, len(list(G.neighbors(z)))


#getting common neighborhood
def getCommonNbd(G,u,v):
	N1 = intbitset(list(G.neighbors(u)))
	#print(N1)
	N2 = intbitset(list(G.neighbors(v)))
	return N1 & N2


############################### topological similarity #################################

#computing the topological similarity between a pair using MINM
def getTopologicalSimilarity_MINM(G, u, v, ku, kv):
	E = G.size()
	CN = getCommonNbd(G,u,v)
	#print(CN)
	#print(len(CN))
	outer_temp_sum = 0
	for z in CN:
		all_nbd_pairs, gamma_z = getAllNbdPairs(G,z)
		first_term = 1/(gamma_z*(gamma_z-1))
		inner_temp_sum = 0
		for key in all_nbd_pairs:
			m = key[0]
			n = key[1]
			km = G.degree(m)
			kn = G.degree(n)
			PL_mn = getPL_mn(km, kn, E)
			PL_mn_z = getPL_mn_z(G,z)
			logPL_mn = math.log(PL_mn) if PL_mn !=0 else 0 #Need to ask whether I can take log(0) as 0
			logPL_mn_z = math.log(PL_mn_z) if PL_mn_z !=0 else 0 
			inner_temp_sum += (logPL_mn + logPL_mn_z)
		outer_temp_sum += first_term*inner_temp_sum #Equation 13
	logPL_uv = math.log(1/getPL_mn(ku, kv, E))	
	return outer_temp_sum - logPL_uv


#computing the topological similarity between a pair using CN
def getTopologicalSimilarity_CN(G, u, v, ku, kv):
	E = G.size()
	CN = getCommonNbd(G,u,v)
	return len(CN)


#computing the topological similarity between a pair using PA
def getTopologicalSimilarity_PA(G, u, v, ku, kv):
	#E = G.size()
	#CN = getCommonNbd(G,u,v)
	#degree_u = len(list(G.neighbors(u)))
	#degree_v = len(list(G.neighbors(v)))
	return ku*kv


#computing the topological similarity between a pair using Jaccard
def getTopologicalSimilarity_JC(G, u, v, ku, kv):
	#E = G.size()
	CN = getCommonNbd(G,u,v)

	N_1 = intbitset(list(G.neighbors(u)))
	#print(N1)
	N_2 = intbitset(list(G.neighbors(v)))	
	all_N = N_1 | N_2
	#print(all_N)
	#print(len(all_N))
	return len(CN)/len(all_N)



#computing the topological similarity between a pair using Adamic-Adar
def getTopologicalSimilarity_AA(G, u, v, ku, kv):
	E = G.size()
	CN = getCommonNbd(G,u,v)
	#print(CN)
	#print(len(CN))
	aa_sum = 0
	for z in CN:
		gamma_z = G.degree(z)
		inner_term = 1/math.log(gamma_z)
		aa_sum += inner_term
	return aa_sum


#computing the topological similarity between a pair using Resource-Allocation
def getTopologicalSimilarity_RA(G, u, v, ku, kv):
	E = G.size()
	CN = getCommonNbd(G,u,v)
	#print(CN)
	#print(len(CN))
	ra_sum = 0
	for z in CN:
		gamma_z = G.degree(z)
		inner_term = 1/gamma_z
		ra_sum += inner_term
	return ra_sum


################################################################################################

#Computing all pair topological similarity
def computeAllPairTopologicalSimilarity(G,G_emb_dic):
	TopologicalSimilarity_dic_MINM = {}
	TopologicalSimilarity_dic_CN = {}
	TopologicalSimilarity_dic_PA = {}
	TopologicalSimilarity_dic_JC = {}
	TopologicalSimilarity_dic_AA = {}
	TopologicalSimilarity_dic_RA = {}
	TopologicalSimilarity_dic_Proposed ={}


	for u in G.nodes():
		ku = G.degree(u)
		for v in G.nodes():
			kv = G.degree(v)
			if u < v:
				if ku == 0 or kv == 0:# In case u and v are not connected
					TopologicalSimilarity_dic_MINM[(u,v)] = 0
					TopologicalSimilarity_dic_CN[(u,v)] = 0
					TopologicalSimilarity_dic_PA[(u,v)] = 0
					TopologicalSimilarity_dic_JC[(u,v)] = 0
					TopologicalSimilarity_dic_AA[(u,v)] = 0
					TopologicalSimilarity_dic_RA[(u,v)] = 0
					TopologicalSimilarity_dic_Proposed[(u,v)] = 0

					continue
				TopologicalSimilarity_dic_MINM[(u,v)] = getTopologicalSimilarity_MINM(G, u, v, ku, kv)
				TopologicalSimilarity_dic_CN[(u,v)] = getTopologicalSimilarity_CN(G, u, v, ku, kv)
				TopologicalSimilarity_dic_PA[(u,v)] = getTopologicalSimilarity_PA(G, u, v, ku, kv)
				TopologicalSimilarity_dic_JC[(u,v)] = getTopologicalSimilarity_JC(G, u, v, ku, kv)
				TopologicalSimilarity_dic_AA[(u,v)] = getTopologicalSimilarity_AA(G, u, v, ku, kv)
				TopologicalSimilarity_dic_RA[(u,v)] = getTopologicalSimilarity_RA(G, u, v, ku, kv)
				TopologicalSimilarity_dic_Proposed[(u,v)] = get_emb_vector_similarity(G_emb_dic, u, v)

	return [TopologicalSimilarity_dic_MINM,TopologicalSimilarity_dic_CN,TopologicalSimilarity_dic_PA,TopologicalSimilarity_dic_JC,TopologicalSimilarity_dic_AA,TopologicalSimilarity_dic_RA,TopologicalSimilarity_dic_Proposed]

#Getting the Eucledian distance between two coordinates
def computeED(coor1, coor2):
	point1 = np.array(coor1)
	point2 = np.array(coor2) 
	#print(coor1, coor2, point1, point2)
	dist = np.linalg.norm(point1 - point2)	
	return dist

def computeAllPairED(mni_coords):
	AllPairEucledianDistance_dic = {}
	for u in range(len(mni_coords)):
		for v in range(len(mni_coords)):
			if u < v: 
				AllPairEucledianDistance_dic[(u,v)] = computeED(mni_coords[u],mni_coords[v])
	return AllPairEucledianDistance_dic

def getEdgeDiff(G1, G2):
	alpha = []
	beta = []
	for u in G1.nodes():
		for v in G1.nodes():
			if u < v:
				if not G1.has_edge(u,v) and G2.has_edge(u,v): #e is absent in G1 but present in G2
					alpha.append((u,v))
				elif G1.has_edge(u,v) and not G2.has_edge(u,v): #e is present in G1 but absent in G2 
					beta.append((u,v))
	return alpha, beta

def computeAllPairConnectionProb(AllPairTopologicalSimilarity, AllPairEucledianDistance, gamma, eta):
	AllPairConnectionProb = {}
	for edge in AllPairTopologicalSimilarity:
		AllPairConnectionProb[edge] = (AllPairTopologicalSimilarity[edge]**gamma)*(AllPairEucledianDistance[edge]**(-eta))
	return AllPairConnectionProb

def computeAllPairConnectionProbList(TopologicalSimilarity_dic_List, AllPairEucledianDistance, gamma, eta):
	AllPairConnectionProbList = []
	for AllPairTopologicalSimilarity in TopologicalSimilarity_dic_List:
		AllPairConnectionProb = {}
		for edge in AllPairTopologicalSimilarity:
			AllPairConnectionProb[edge] = (AllPairTopologicalSimilarity[edge]**gamma)*(AllPairEucledianDistance[edge]**(-eta))
		AllPairConnectionProbList.append(AllPairConnectionProb)
	return AllPairConnectionProbList


def getNodePairToAddOrDelete(G_init, sorted_AllPair_P_uv, starting_index, mode='add'):
	if mode == 'add':
		start = starting_index
		end = len(sorted_AllPair_P_uv)
		step = 1
	elif mode == 'delete':
		start = starting_index
		end = -1
		step = -1
	for i in range(start, end, step):
		(u,v) = sorted_AllPair_P_uv[i][0]
		if mode == 'add' and not G_init.has_edge(*(u,v)):
			return (u,v), i+1
		elif mode == 'delete' and G_init.has_edge(*(u,v)):
			return (u,v), i-1
		

def makeEvolution(G_init, sorted_AllPair_P_uv, alpha, beta, thresh=0.5):
	G_temp = G_init.copy()
	no_of_edge_added = 0
	no_of_edge_deleted = 0
	left_index = 0
	right_index = len(sorted_AllPair_P_uv)-1
	while True:
		rnd = random.uniform(0, 1)
		if rnd > thresh and no_of_edge_added < len(alpha) and left_index < len(sorted_AllPair_P_uv): #Add 
			#find the node pair having no connection but high connection probability
			e, left_index = getNodePairToAddOrDelete(G_init, sorted_AllPair_P_uv, left_index, mode='add')
			#Add an edge between that node pair in G_init
			if not G_temp.has_edge(*e):
				G_temp.add_edge(*e)
				no_of_edge_added += 1
		elif rnd < thresh and no_of_edge_deleted < len(beta) and right_index > -1: #Delete
			#find the node pair having a connection but low connection probability
			e, right_index = getNodePairToAddOrDelete(G_init, sorted_AllPair_P_uv, right_index, mode='delete')
			#Delete that node pair in G_init
			if G_temp.has_edge(*e):
				G_temp.remove_edge(*e)
				no_of_edge_deleted += 1
		#print(no_of_edge_added, len(alpha), no_of_edge_deleted, len(beta))
		if no_of_edge_added == len(alpha) and no_of_edge_deleted == len(beta):
		 	break
		#if no_of_edge_added == len(alpha) and right_index == -1:
		# 	break
		#if no_of_edge_deleted == len(beta) and left_index == len(sorted_AllPair_P_uv):
		# 	break
		
	print('no_of_edge_added',no_of_edge_added)  
	print('no_of_edge_deleted',no_of_edge_deleted)
	return G_temp


def makeEvolutionRandom(G_init, sorted_AllPair_P_uv, alpha, beta, thresh=0.5):
	G_temp = G_init.copy()
	no_of_edge_added = 0
	no_of_edge_deleted = 0
	#left_index = 0
	#right_index = len(sorted_AllPair_P_uv)-1
	while True:
		rnd = random.uniform(0, 1)
		#if rnd > thresh and no_of_edge_added < len(alpha) and left_index < len(sorted_AllPair_P_uv): #Add 
		if rnd > thresh and no_of_edge_added < len(alpha): #Add 

			#find the node pair having no connection but high connection probability
			# e, left_index = getNodePairToAddOrDelete(G_init, sorted_AllPair_P_uv, left_index, mode='add')
			#  find random pair
			rand_index=random.randint(0, len(sorted_AllPair_P_uv)-1)
			e=sorted_AllPair_P_uv[rand_index][0]
			#Add an edge between that node pair in G_init
			if not G_temp.has_edge(*e):
				G_temp.add_edge(*e)
				no_of_edge_added += 1
		#elif rnd < thresh and no_of_edge_deleted < len(beta) and right_index > -1: #Delete
		elif rnd < thresh and no_of_edge_deleted < len(beta): #Delete

			#find the node pair having a connection but low connection probability
			# e, right_index = getNodePairToAddOrDelete(G_init, sorted_AllPair_P_uv, right_index, mode='delete')
			# find random pair
			rand_index=random.randint(0, len(sorted_AllPair_P_uv)-1)
			e=sorted_AllPair_P_uv[rand_index][0]
			#Delete that node pair in G_init
			if G_temp.has_edge(*e):
				G_temp.remove_edge(*e)
				no_of_edge_deleted += 1
		#print(no_of_edge_added, len(alpha), no_of_edge_deleted, len(beta))
		if no_of_edge_added == len(alpha) and no_of_edge_deleted == len(beta):
		 	break
		#if no_of_edge_added == len(alpha) and right_index == -1:
		# 	break
		#if no_of_edge_deleted == len(beta) and left_index == len(sorted_AllPair_P_uv):
		# 	break
		
	print('no_of_edge_added_random',no_of_edge_added)  
	print('no_of_edge_deleted_random',no_of_edge_deleted)
	return G_temp

def getMatrix(args):
	if args.corr == 'correlation':
	    fname1 = "mean_AD_correlation_matrix.pkl"
	    fname2 = "mean_MCI_correlation_matrix.pkl"
	    fname3 = "mean_NC_correlation_matrix.pkl"
	elif args.corr == 'partial correlation':
	    fname1 = "mean_AD_part_correlation_matrix.pkl"
	    fname2 = "mean_MCI_part_correlation_matrix.pkl"
	    fname3 = "mean_NC_part_correlation_matrix.pkl"
	elif args.corr == 'tangent':
	    fname1 = "mean_AD_tangent_matrix.pkl"
	    fname2 = "mean_MCI_tangent_matrix.pkl"
	    fname3 = "mean_NC_tangent_matrix.pkl"
	mean_AD_correlation_matrix = pickle.load(open(fname1, 'rb'))
	mean_MCI_correlation_matrix = pickle.load(open(fname2,'rb'))
	mean_NC_correlation_matrix = pickle.load(open(fname3,'rb'))

	return mean_AD_correlation_matrix, mean_MCI_correlation_matrix, mean_NC_correlation_matrix

def getRichClub(G):
	sum_val = 0
	rich_dict = nx.rich_club_coefficient(G, normalized=False)
	for key in rich_dict:
		sum_val += rich_dict[key]
	return sum_val/len(rich_dict)

def getNodeEmbeddingDeepWalk(G):
	random_walks = []
	for n in G.nodes():
		for i in range(10):
			random_walks.append(get_randomwalk(G, n, 120))

	model = Word2Vec(window = 10, sg = 1, hs = 0,  negative = 5,  alpha=0.03, min_alpha=0.0007, seed = 14)
	model.build_vocab(random_walks, progress_per=2)
	model.train(random_walks, total_examples = model.corpus_count, epochs=20, report_delay=1)
	emb_vector_dic = {}
	for node in G.nodes():
		emb_vector_dic[node]=model.wv[str(node)]

	return emb_vector_dic

def getNodeEmbeddingParwalk(G):
	random_walks = getParwalk(G, path_length=100)			

	model = Word2Vec(window = 10, sg = 1, hs = 0,  negative = 5,  alpha=0.03, min_alpha=0.0007, seed = 14)
	model.build_vocab(random_walks, progress_per=2)
	model.train(random_walks, total_examples = model.corpus_count, epochs=20, report_delay=1)
	emb_vector_dic = {}
	for node in G.nodes():
		emb_vector_dic[node]=model.wv[str(node)]

	return emb_vector_dic


def getNodeEmbeddingNode2Vec(nx_G, args):
	print('Node to Vec Embedding')
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	walks = [str(walk) for walk in walks]
	model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, epochs=args.iter)
	emb_vector_dic = {}
	for node in nx_G.nodes():
		if str(node) in model.wv:
			emb_vector_dic[node]=model.wv[str(node)]

	return emb_vector_dic

def getLineEmbedding(nx_G, args):
    print('Line Embedding')
    maxindex = max(nx_G.nodes())
    line = Line(maxindex + 1, embed_dim=args.dimensions, order=args.order)
    # Create dict of distribution when opening file
    edgedistdict = collections.defaultdict(int)
    nodedistdict = collections.defaultdict(int)

    weightsdict = collections.defaultdict(int)
    nodedegrees = collections.defaultdict(int)

    weightsum = 0
    negprobsum = 0

    for edge in nx_G.edges():
        weight = nx_G[edge[0]][edge[1]]['weight']
        weightsdict[edge] = weight
        nodedegrees[edge[0]] += weight
        weightsum += weight
        negprobsum += np.power(weight, args.negativepower)

    for node, outdegree in nodedegrees.items():
        nodedistdict[node] = np.power(outdegree, args.negativepower) / negprobsum

    for edge, weight in weightsdict.items():
        edgedistdict[edge] = weight / weightsum
    
    edgesaliassampler = VoseAlias(edgedistdict)
    nodesaliassampler = VoseAlias(nodedistdict)

    batchrange = int(len(edgedistdict) / args.batchsize)
    opt = torch.optim.SGD(line.parameters(), lr=args.learning_rate,
                    momentum=0.9, nesterov=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lossdata = {"it": [], "loss": []}
    it = 0
    print("\nTraining on {}...\n".format(device))
    for epoch in range(args.iter):
        print("Epoch {}".format(epoch))
        for b in trange(batchrange):
            samplededges = edgesaliassampler.sample_n(args.batchsize)
            batch = list(makeData(samplededges, args.negsamplesize, weightsdict, nodedegrees,
                                  nodesaliassampler))
            batch = torch.LongTensor(batch)
            v_i = batch[:, 0]
            v_j = batch[:, 1]
            negsamples = batch[:, 2:]
            line.zero_grad()
            loss = line(v_i, v_j, negsamples, device)
            loss.backward()
            opt.step()

            lossdata["loss"].append(loss.item())
            lossdata["it"].append(it)
            it += 1

    emb_vector_dic = {}
    for node in nx_G.nodes():
        emb_vector_dic[node]=line.nodes_embeddings(torch.tensor(node)).cpu().detach().numpy()

    return emb_vector_dic

def getGCNEmbedding(nx_G, args):
    #get the dataset
    dataset = MyDataset(nx_G, args.dimensions)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(dataset, args).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data)
    emb_vector_dic = {}
    for node in nx_G.nodes():
        emb_vector_dic[node]=pred[node].cpu().detach().numpy()

    return emb_vector_dic

def getSAGEEmbedding(nx_G, args):
    #get the dataset
    dataset = MyDataset(nx_G, args.dimensions)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAGE_Net(dataset, args).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data)
    emb_vector_dic = {}
    for node in nx_G.nodes():
        emb_vector_dic[node]=pred[node].cpu().detach().numpy()

    return emb_vector_dic

def getChebNetEmbedding(nx_G, args):
    #get the dataset
    dataset = MyDataset(nx_G, args.dimensions)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChebNet(dataset, args).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data)
    emb_vector_dic = {}
    for node in nx_G.nodes():
        emb_vector_dic[node]=pred[node].cpu().detach().numpy()

    return emb_vector_dic

def getGATEmbedding(nx_G, args):
    #get the dataset
    dataset = MyDataset(nx_G, args.dimensions)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT_Net(dataset, args).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data)
    emb_vector_dic = {}
    for node in nx_G.nodes():
        emb_vector_dic[node]=pred[node].cpu().detach().numpy()

    return emb_vector_dic

if __name__=='__main__':
	parser = ArgumentParser(description='Process some parameters.')
	parser.add_argument('-corr', metavar='corr', type=str, nargs = '?', default = 'correlation' , help='choose from \'correlation\' , \'partial correlation\' ,\'tangent\' ')
	parser.add_argument('--directed', dest='directed', action='store_true', help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.add_argument('--p', type=float, default=1, help='Return hyperparameter. Default is 1.')
	parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter. Default is 1.')
	parser.add_argument('--walk-length', type=int, default=80, help='Length of walk per source. Default is 80.')
	parser.add_argument('--num-walks', type=int, default=10, help='Number of walks per source. Default is 10.')
	parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')
	parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 10.')
	parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')
	parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers. Default is 8.')
	parser.add_argument('--order', type=int, default=1, help='orders')
	parser.add_argument("-negpow", "--negativepower", type=float, default=0.75)
	parser.add_argument("-batchsize", "--batchsize", type=int, default=5)
	parser.add_argument("-lr", "--learning_rate", type=float, default=0.025)  # As starting value in paper
	parser.add_argument("-neg", "--negsamplesize", type=int, default=5)
	parser.add_argument("-hidden", "--hidden", type=int, default=16)
	parser.add_argument('--heads', default=8, type=int)
	parser.add_argument('--output_heads', default=1, type=int)
	parser.add_argument('--dropout', type=float, default=0.5)

	args = parser.parse_args()

	#Loading the atlas
	atlas = datasets.fetch_atlas_aal(version='SPM12', data_dir=None, url=None, resume=True, verbose=1)
	coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas['maps'])

	#Load the correlation matrices
	mean_AD_correlation_matrix, mean_MCI_correlation_matrix, mean_NC_correlation_matrix = getMatrix(args)

	#generate the graph from the matrices
	# 0.05 for partial correlation, 0.5 correlation and 0.001 for tangent 
	edge_threshold=0.50
	Ga = getNetworkxGraphFromAdjByValue(np.log(1+mean_AD_correlation_matrix)-np.log(1-mean_AD_correlation_matrix), edge_threshold=edge_threshold)
	Gm = getNetworkxGraphFromAdjByValue(np.log(1+mean_MCI_correlation_matrix)-np.log(1-mean_MCI_correlation_matrix), edge_threshold=edge_threshold)
	Gn = getNetworkxGraphFromAdjByValue(np.log(1+mean_NC_correlation_matrix)-np.log(1-mean_NC_correlation_matrix), edge_threshold=edge_threshold)

	'''
	Gn_degree_sequence = sorted([d for n, d in Gn.degree()], reverse=True)
	Gm_degree_sequence = sorted([d for n, d in Gm.degree()], reverse=True)
	Ga_degree_sequence = sorted([d for n, d in Ga.degree()], reverse=True)

	Gn_node_deg, Gn_deg_freq = np.unique(Gn_degree_sequence, return_counts=True)
	Gm_node_deg, Gm_deg_freq = np.unique(Gm_degree_sequence, return_counts=True)
	Ga_node_deg, Ga_deg_freq = np.unique(Ga_degree_sequence, return_counts=True)

	Gn_degree_freq_list = []
	for nd, df in zip(Gn_node_deg, Gn_deg_freq):
		Gn_degree_freq_list.append((nd, df))
	Gm_degree_freq_list = []
	for nd, df in zip(Gm_node_deg, Gm_deg_freq):
		Gm_degree_freq_list.append((nd, df))
	Ga_degree_freq_list = []
	for nd, df in zip(Ga_node_deg, Ga_deg_freq):
		Ga_degree_freq_list.append((nd, df))

	print(Gn_degree_freq_list, Gm_degree_freq_list, Ga_degree_freq_list)
    '''

	#generating node embedding 
	#Gn_emb_vec_dic=getNodeEmbeddingDeepWalk(Gn)
	#Gn_emb_vec_dic=getNodeEmbeddingParwalk(Gn)
	#Gn_emb_vec_dic=getNodeEmbeddingNode2Vec(Gn, args)
	#Gn_emb_vec_dic=getLineEmbedding(Gn, args)
	#Gn_emb_vec_dic=getGCNEmbedding(Gn, args)
	#Gn_emb_vec_dic=getSAGEEmbedding(Gn, args)
	#Gn_emb_vec_dic=getChebNetEmbedding(Gn, args)
	Gn_emb_vec_dic=getSAGEEmbedding(Gn, args)
	
	#computing the edge differencs between Gn and Gm. alpha => edges to be added and beta => edges to be deleted
	alpha1, beta1 = getEdgeDiff(Gn, Gm)
	alpha2, beta2 = getEdgeDiff(Gn, Ga)

	print('--------------------------------------------------')	
	print('Edge difference between Gn and Gm: ', 'No of edges need to be added: ',len(alpha1), 'No of edges need to be deleted: ',len(beta1))
	print('Edge difference between Gn and Ga: ', 'No of edges need to be added: ',len(alpha2), 'No of edges need to be deleted: ',len(beta2))

	print('--------------------------------------------------')		

	#Get all pair Eucledian distance
	AllPairEucledianDistance = computeAllPairED(coordinates)

	#Get All pair topological similarity
	TopologicalSimilarity_dic_List = computeAllPairTopologicalSimilarity(Gn,Gn_emb_vec_dic)

	#calculate the connection probability
	gamma = 20
	eta = 8
	AllPair_P_uv_List = computeAllPairConnectionProbList(TopologicalSimilarity_dic_List, AllPairEucledianDistance, gamma, eta)

	#sort the connection probability lists
	sorted_AllPair_P_uv_List = []
	for AllPair_P_uv in AllPair_P_uv_List:
		sorted_AllPair_P_uv = sorted(AllPair_P_uv.items(), key=lambda kv: kv[1], reverse=True)
		sorted_AllPair_P_uv_List.append(sorted_AllPair_P_uv)


	sorted_AllPair_P_uv_MINM = sorted_AllPair_P_uv_List[0]
	sorted_AllPair_P_uv_CN = sorted_AllPair_P_uv_List[1]
	sorted_AllPair_P_uv_PA = sorted_AllPair_P_uv_List[2]
	sorted_AllPair_P_uv_JC = sorted_AllPair_P_uv_List[3]
	sorted_AllPair_P_uv_AA = sorted_AllPair_P_uv_List[4]
	sorted_AllPair_P_uv_RA = sorted_AllPair_P_uv_List[5]
	sorted_AllPair_P_uv_Proposed = sorted_AllPair_P_uv_List[6]


	#Evolution
	print('Obtaining Gm sythetic...')
	Gm_synthetic_MINM = makeEvolution(Gn, sorted_AllPair_P_uv_MINM, alpha1, beta1)
	Gm_synthetic_CN = makeEvolution(Gn, sorted_AllPair_P_uv_CN, alpha1, beta1)
	Gm_synthetic_PA = makeEvolution(Gn, sorted_AllPair_P_uv_PA, alpha1, beta1)
	Gm_synthetic_JC = makeEvolution(Gn, sorted_AllPair_P_uv_JC, alpha1, beta1)
	Gm_synthetic_AA = makeEvolution(Gn, sorted_AllPair_P_uv_AA, alpha1, beta1)
	Gm_synthetic_RA = makeEvolution(Gn, sorted_AllPair_P_uv_RA, alpha1, beta1)
	Gm_synthetic_Proposed = makeEvolution(Gn, sorted_AllPair_P_uv_Proposed, alpha1, beta1)
	Gm_synthetic_Random = makeEvolutionRandom(Gn, sorted_AllPair_P_uv_Proposed, alpha1, beta1)


	print('Obtaining Ga sythetic...')
	Ga_synthetic_MINM = makeEvolution(Gn, sorted_AllPair_P_uv_MINM, alpha2, beta2)
	Ga_synthetic_CN = makeEvolution(Gn, sorted_AllPair_P_uv_CN, alpha2, beta2)
	Ga_synthetic_PA = makeEvolution(Gn, sorted_AllPair_P_uv_PA, alpha2, beta2)
	Ga_synthetic_JC = makeEvolution(Gn, sorted_AllPair_P_uv_JC, alpha2, beta2)
	Ga_synthetic_AA = makeEvolution(Gn, sorted_AllPair_P_uv_AA, alpha2, beta2)
	Ga_synthetic_RA = makeEvolution(Gn, sorted_AllPair_P_uv_RA, alpha2, beta2)
	Ga_synthetic_Proposed = makeEvolution(Gn, sorted_AllPair_P_uv_Proposed, alpha2, beta2)
	Ga_synthetic_Random = makeEvolutionRandom(Gn, sorted_AllPair_P_uv_Proposed, alpha2, beta2)
	
	print('--------------------------------------------------')
	print('Number of Edges in Gn:', Gn.size())
	print('Number of Edges in Gm:', Gm.size())
	print('Number of Edges in Ga:', Ga.size())
	#print('Number of Edges in Gm_synthetic:', Gm_synthetic.size())
	#print('Number of Edges in Ga_synthetic:', Ga_synthetic.size())
	#print(Gm.size(), Ga.size(), Gm_synthetic.size(), Ga_synthetic.size())
	

	#### remove isolated nodes ##########

	Gn.remove_nodes_from(list(nx.isolates(Gn)))
	Gm.remove_nodes_from(list(nx.isolates(Gm)))
	Ga.remove_nodes_from(list(nx.isolates(Ga)))

	print('calculating properties of the real graphs...')
	print(nx_comm.modularity(Gn,nx_comm.label_propagation_communities(Gn)), nx_comm.modularity(Gm,nx_comm.label_propagation_communities(Gm)), nx_comm.modularity(Ga,nx_comm.label_propagation_communities(Ga)))
	print(nx.local_efficiency(Gn), nx.local_efficiency(Gm), nx.local_efficiency(Ga))
	print(nx.average_clustering(Gn), nx.average_clustering(Gm), nx.average_clustering(Ga))



	Gm_synthetic_MINM.remove_nodes_from(list(nx.isolates(Gm_synthetic_MINM)))
	Gm_synthetic_CN.remove_nodes_from(list(nx.isolates(Gm_synthetic_CN)))
	Gm_synthetic_PA.remove_nodes_from(list(nx.isolates(Gm_synthetic_PA)))
	Gm_synthetic_JC.remove_nodes_from(list(nx.isolates(Gm_synthetic_JC)))
	Gm_synthetic_AA.remove_nodes_from(list(nx.isolates(Gm_synthetic_AA)))
	Gm_synthetic_RA.remove_nodes_from(list(nx.isolates(Gm_synthetic_RA)))
	Gm_synthetic_Proposed.remove_nodes_from(list(nx.isolates(Gm_synthetic_Proposed)))
	Gm_synthetic_Random.remove_nodes_from(list(nx.isolates(Gm_synthetic_Random)))



	Ga_synthetic_MINM.remove_nodes_from(list(nx.isolates(Ga_synthetic_MINM)))
	Ga_synthetic_CN.remove_nodes_from(list(nx.isolates(Ga_synthetic_CN)))
	Ga_synthetic_PA.remove_nodes_from(list(nx.isolates(Ga_synthetic_PA)))
	Ga_synthetic_JC.remove_nodes_from(list(nx.isolates(Ga_synthetic_JC)))
	Ga_synthetic_AA.remove_nodes_from(list(nx.isolates(Ga_synthetic_AA)))
	Ga_synthetic_RA.remove_nodes_from(list(nx.isolates(Ga_synthetic_RA)))
	Ga_synthetic_Proposed.remove_nodes_from(list(nx.isolates(Ga_synthetic_Proposed)))
	Ga_synthetic_Random.remove_nodes_from(list(nx.isolates(Ga_synthetic_Random)))

	# Ga=nx.isolates(Ga)
	# Gm=nx.isolates(Gm)
	# Gn=nx.isolates(Ga)
	Gn_degree_sequence = sorted([d for n, d in Gn.degree()], reverse=True)
	Gm_degree_sequence = sorted([d for n, d in Gm.degree()], reverse=True)
	Ga_degree_sequence = sorted([d for n, d in Ga.degree()], reverse=True)


	Gm_synthetic_CN_degree_sequence = sorted([d for n, d in Gm_synthetic_CN.degree()], reverse=True)
	Gm_synthetic_PA_degree_sequence = sorted([d for n, d in Gm_synthetic_PA.degree()], reverse=True)
	Gm_synthetic_JC_degree_sequence = sorted([d for n, d in Gm_synthetic_JC.degree()], reverse=True)
	Gm_synthetic_AA_degree_sequence = sorted([d for n, d in Gm_synthetic_AA.degree()], reverse=True)
	Gm_synthetic_RA_degree_sequence = sorted([d for n, d in Gm_synthetic_RA.degree()], reverse=True)
	Gm_synthetic_Proposed_degree_sequence = sorted([d for n, d in Gm_synthetic_Proposed.degree()], reverse=True)
	Gm_synthetic_Random_degree_sequence = sorted([d for n, d in Gm_synthetic_Random.degree()], reverse=True)

	Ga_synthetic_CN_degree_sequence = sorted([d for n, d in Ga_synthetic_CN.degree()], reverse=True)
	Ga_synthetic_PA_degree_sequence = sorted([d for n, d in Ga_synthetic_PA.degree()], reverse=True)
	Ga_synthetic_JC_degree_sequence = sorted([d for n, d in Ga_synthetic_JC.degree()], reverse=True)
	Ga_synthetic_AA_degree_sequence = sorted([d for n, d in Ga_synthetic_AA.degree()], reverse=True)
	Ga_synthetic_RA_degree_sequence = sorted([d for n, d in Ga_synthetic_RA.degree()], reverse=True)
	Ga_synthetic_Proposed_degree_sequence = sorted([d for n, d in Ga_synthetic_Proposed.degree()], reverse=True)
	Ga_synthetic_Random_degree_sequence = sorted([d for n, d in Ga_synthetic_Random.degree()], reverse=True)

	Gn_node_deg, Gn_deg_freq = np.unique(Gn_degree_sequence, return_counts=True)
	Gm_node_deg, Gm_deg_freq = np.unique(Gm_degree_sequence, return_counts=True)
	Ga_node_deg, Ga_deg_freq = np.unique(Ga_degree_sequence, return_counts=True)

	Gm_synthetic_CN_node_deg, Gm_synthetic_CN_deg_freq = np.unique(Gm_synthetic_CN_degree_sequence, return_counts=True)
	Gm_synthetic_PA_node_deg, Gm_synthetic_PA_deg_freq = np.unique(Gm_synthetic_PA_degree_sequence, return_counts=True)
	Gm_synthetic_JC_node_deg, Gm_synthetic_JC_deg_freq = np.unique(Gm_synthetic_JC_degree_sequence, return_counts=True)
	Gm_synthetic_AA_node_deg, Gm_synthetic_AA_deg_freq = np.unique(Gm_synthetic_AA_degree_sequence, return_counts=True)
	Gm_synthetic_RA_node_deg, Gm_synthetic_RA_deg_freq = np.unique(Gm_synthetic_RA_degree_sequence, return_counts=True)
	Gm_synthetic_Proposed_node_deg, Gm_synthetic_Proposed_deg_freq = np.unique(Gm_synthetic_Proposed_degree_sequence, return_counts=True)
	Gm_synthetic_Random_node_deg, Gm_synthetic_Random_deg_freq = np.unique(Gm_synthetic_Random_degree_sequence, return_counts=True)


	Ga_synthetic_CN_node_deg, Ga_synthetic_CN_deg_freq = np.unique(Ga_synthetic_CN_degree_sequence, return_counts=True)
	Ga_synthetic_PA_node_deg, Ga_synthetic_PA_deg_freq = np.unique(Ga_synthetic_PA_degree_sequence, return_counts=True)
	Ga_synthetic_JC_node_deg, Ga_synthetic_JC_deg_freq = np.unique(Ga_synthetic_JC_degree_sequence, return_counts=True)
	Ga_synthetic_AA_node_deg, Ga_synthetic_AA_deg_freq = np.unique(Ga_synthetic_AA_degree_sequence, return_counts=True)
	Ga_synthetic_RA_node_deg, Ga_synthetic_RA_deg_freq = np.unique(Ga_synthetic_RA_degree_sequence, return_counts=True)
	Ga_synthetic_Proposed_node_deg, Ga_synthetic_Proposed_deg_freq = np.unique(Ga_synthetic_Proposed_degree_sequence, return_counts=True)
	Ga_synthetic_Random_node_deg, Ga_synthetic_Random_deg_freq = np.unique(Ga_synthetic_Random_degree_sequence, return_counts=True)


	Gm_degree_freq_list = []
	Gm_synthetic_CN_degree_freq_list = []
	Gm_synthetic_PA_degree_freq_list = []
	Gm_synthetic_JC_degree_freq_list = []
	Gm_synthetic_AA_degree_freq_list = []
	Gm_synthetic_RA_degree_freq_list = []
	Gm_synthetic_Proposed_degree_freq_list = []
	Gm_synthetic_Random_degree_freq_list = []

    
	for nd, df in zip(Gm_node_deg, Gm_deg_freq):
		Gm_degree_freq_list.append((nd, df))	
	print('Gm_degree_freq_list: ', Gm_degree_freq_list)

	for nd, df in zip(Gm_synthetic_CN_node_deg, Gm_synthetic_CN_deg_freq):
		Gm_synthetic_CN_degree_freq_list.append((nd, df))	
	print('Gm_synthetic_CN: ', Gm_synthetic_CN_degree_freq_list)

	for nd, df in zip(Ga_synthetic_PA_node_deg, Ga_synthetic_PA_deg_freq):
		Gm_synthetic_PA_degree_freq_list.append((nd, df))	
	print('Gm_synthetic_PA: ', Gm_synthetic_PA_degree_freq_list)

	for nd, df in zip(Gm_synthetic_JC_node_deg, Gm_synthetic_JC_deg_freq):
		Gm_synthetic_JC_degree_freq_list.append((nd, df))
	print('Gm_synthetic_JC: ', Gm_synthetic_JC_degree_freq_list)

	for nd, df in zip(Gm_synthetic_AA_node_deg, Gm_synthetic_AA_deg_freq):
		Gm_synthetic_AA_degree_freq_list.append((nd, df))	
	print('Gm_synthetic_AA: ', Gm_synthetic_AA_degree_freq_list)

	for nd, df in zip(Gm_synthetic_RA_node_deg, Gm_synthetic_RA_deg_freq):
		Gm_synthetic_RA_degree_freq_list.append((nd, df))	
	print('Gm_synthetic_RA: ', Gm_synthetic_RA_degree_freq_list)

	for nd, df in zip(Gm_synthetic_Proposed_node_deg, Gm_synthetic_Proposed_deg_freq):
		Gm_synthetic_Proposed_degree_freq_list.append((nd, df))	
	print('Gm_synthetic_Proposed: ', Gm_synthetic_Proposed_degree_freq_list)

	for nd, df in zip(Gm_synthetic_Random_node_deg, Gm_synthetic_Random_deg_freq):
		Gm_synthetic_Random_degree_freq_list.append((nd, df))	
	print('Gm_synthetic_Random: ', Gm_synthetic_Random_degree_freq_list)



	Ga_degree_freq_list = []
	Ga_synthetic_CN_degree_freq_list = []
	Ga_synthetic_PA_degree_freq_list = []
	Ga_synthetic_JC_degree_freq_list = []
	Ga_synthetic_AA_degree_freq_list = []
	Ga_synthetic_RA_degree_freq_list = []
	Ga_synthetic_Proposed_degree_freq_list = []
	Ga_synthetic_Random_degree_freq_list = []

	for nd, df in zip(Ga_node_deg, Ga_deg_freq):
		Ga_degree_freq_list.append((nd, df))	
	print('Ga_degree_freq_list: ', Ga_degree_freq_list)

	for nd, df in zip(Ga_synthetic_CN_node_deg, Ga_synthetic_CN_deg_freq):
		Ga_synthetic_CN_degree_freq_list.append((nd, df))	
	print('Ga_synthetic_CN: ', Ga_synthetic_CN_degree_freq_list)

	for nd, df in zip(Ga_synthetic_PA_node_deg, Ga_synthetic_PA_deg_freq):
		Ga_synthetic_PA_degree_freq_list.append((nd, df))	
	print('Ga_synthetic_PA: ', Ga_synthetic_PA_degree_freq_list)

	for nd, df in zip(Ga_synthetic_JC_node_deg, Ga_synthetic_JC_deg_freq):
		Ga_synthetic_JC_degree_freq_list.append((nd, df))
	print('Ga_synthetic_JC: ', Ga_synthetic_JC_degree_freq_list)

	for nd, df in zip(Ga_synthetic_AA_node_deg, Ga_synthetic_AA_deg_freq):
		Ga_synthetic_AA_degree_freq_list.append((nd, df))	
	print('Ga_synthetic_AA: ', Ga_synthetic_AA_degree_freq_list)

	for nd, df in zip(Ga_synthetic_RA_node_deg, Ga_synthetic_RA_deg_freq):
		Ga_synthetic_RA_degree_freq_list.append((nd, df))	
	print('Ga_synthetic_RA: ', Ga_synthetic_RA_degree_freq_list)

	for nd, df in zip(Ga_synthetic_Proposed_node_deg, Ga_synthetic_Proposed_deg_freq):
		Ga_synthetic_Proposed_degree_freq_list.append((nd, df))	
	print('Ga_synthetic_Proposed: ', Ga_synthetic_Proposed_degree_freq_list)

	for nd, df in zip(Ga_synthetic_Random_node_deg, Ga_synthetic_Random_deg_freq):
		Ga_synthetic_Random_degree_freq_list.append((nd, df))	
	print('Ga_synthetic_Random: ', Ga_synthetic_Random_degree_freq_list)


	# plot average clustering coefficient for different topological similarity values corresponding to Target Network (TN)=MCI ##################

	keys=["TN","MINM","CN","PA","JC","AA","RA","Proposed","Random"]
	x = np.arange(len(keys))  # the label locations
	#width=1

	print("average_clustering coefficient")
	print(nx.average_clustering(Gm), nx.average_clustering(Gm_synthetic_MINM),nx.average_clustering(Gm_synthetic_CN),nx.average_clustering(Gm_synthetic_PA),
		nx.average_clustering(Gm_synthetic_JC),nx.average_clustering(Gm_synthetic_AA),nx.average_clustering(Gm_synthetic_RA),nx.average_clustering(Gm_synthetic_Proposed),nx.average_clustering(Gm_synthetic_Random))
	
	values_clustering_coefficient = [nx.average_clustering(Gm),nx.average_clustering(Gm_synthetic_MINM),nx.average_clustering(Gm_synthetic_CN),nx.average_clustering(Gm_synthetic_PA),
		nx.average_clustering(Gm_synthetic_JC),nx.average_clustering(Gm_synthetic_AA),nx.average_clustering(Gm_synthetic_RA),nx.average_clustering(Gm_synthetic_Proposed),nx.average_clustering(Gm_synthetic_Random)]

	plt.bar(keys, values_clustering_coefficient, width=0.4, color='b', align='center',label='average_clustering_coefficient')
	plt.xlabel("Methods")
	plt.ylabel("average_clustering_coefficient")
	plt.savefig('average_clustering_coefficient.png',dpi=400)
	plt.close()
	
	print("Transitivity")
	print(nx.transitivity(Gm),nx.transitivity(Gm_synthetic_MINM),nx.transitivity(Gm_synthetic_CN),nx.transitivity(Gm_synthetic_PA),
		nx.transitivity(Gm_synthetic_JC),nx.transitivity(Gm_synthetic_AA),nx.transitivity(Gm_synthetic_RA),nx.transitivity(Gm_synthetic_Proposed),nx.transitivity(Gm_synthetic_Random))
	
	values_transitivity = [nx.transitivity(Gm),nx.transitivity(Gm_synthetic_MINM),nx.transitivity(Gm_synthetic_CN),nx.transitivity(Gm_synthetic_PA),
		nx.transitivity(Gm_synthetic_JC),nx.transitivity(Gm_synthetic_AA),nx.transitivity(Gm_synthetic_RA)]

	# plt.bar(keys, values_transitivity, width=0.4, color='b', align='center',label='Transitivity')
	# plt.xlabel("Methods")
	# plt.ylabel("Transitivityy")
	# plt.savefig('Transitivity.png',dpi=400)
	# plt.close()

	print("local_efficiency")
	print(nx.local_efficiency(Gm),nx.local_efficiency(Gm_synthetic_MINM),nx.local_efficiency(Gm_synthetic_CN),nx.local_efficiency(Gm_synthetic_PA),
		nx.local_efficiency(Gm_synthetic_JC),nx.local_efficiency(Gm_synthetic_AA),nx.local_efficiency(Gm_synthetic_RA),nx.local_efficiency(Gm_synthetic_Proposed),nx.local_efficiency(Gm_synthetic_Random))
	
	values_local_efficiency = [nx.local_efficiency(Gm),nx.local_efficiency(Gm_synthetic_MINM),nx.local_efficiency(Gm_synthetic_CN),nx.local_efficiency(Gm_synthetic_PA),
		nx.local_efficiency(Gm_synthetic_JC),nx.local_efficiency(Gm_synthetic_AA),nx.local_efficiency(Gm_synthetic_RA)]

	# plt.bar(keys, values_local_efficiency, width=0.4, color='b', align='center',label='local_efficiency')
	# plt.xlabel("Methods")
	# plt.ylabel("local_efficiency")
	# plt.savefig('local_efficiency.png',dpi=400)
	# plt.close()


	print("global_efficiency")
	print(nx.global_efficiency(Gm),nx.global_efficiency(Gm_synthetic_MINM),nx.global_efficiency(Gm_synthetic_CN),nx.global_efficiency(Gm_synthetic_PA),
		nx.global_efficiency(Gm_synthetic_JC),nx.global_efficiency(Gm_synthetic_AA),nx.global_efficiency(Gm_synthetic_RA) ,nx.global_efficiency(Gm_synthetic_Proposed),nx.global_efficiency(Gm_synthetic_Random))
	
	values_global_efficiency = [nx.global_efficiency(Gm),nx.global_efficiency(Gm_synthetic_MINM),nx.global_efficiency(Gm_synthetic_CN),nx.global_efficiency(Gm_synthetic_PA),
		nx.global_efficiency(Gm_synthetic_JC),nx.global_efficiency(Gm_synthetic_AA),nx.global_efficiency(Gm_synthetic_RA)]

	# plt.bar(keys, values_global_efficiency, width=0.4, color='b', align='center',label='global_efficiency')
	# plt.xlabel("Methods")
	# plt.ylabel("global_efficiency")
	# plt.savefig('global_efficiency.png',dpi=400)
	# plt.close()


	print("average_shortest_path_length")
	print(nx.average_shortest_path_length(Gm),nx.average_shortest_path_length(Gm_synthetic_MINM),nx.average_shortest_path_length(Gm_synthetic_CN),nx.average_shortest_path_length(Gm_synthetic_PA),
		nx.average_shortest_path_length(Gm_synthetic_JC),nx.average_shortest_path_length(Gm_synthetic_AA),nx.average_shortest_path_length(Gm_synthetic_RA),nx.average_shortest_path_length(Gm_synthetic_Proposed),nx.average_shortest_path_length(Gm_synthetic_Random))
	
	values_average_shortest_path_length = [nx.average_shortest_path_length(Gm),nx.average_shortest_path_length(Gm_synthetic_MINM),nx.average_shortest_path_length(Gm_synthetic_CN),nx.average_shortest_path_length(Gm_synthetic_PA),
		nx.average_shortest_path_length(Gm_synthetic_JC),nx.average_shortest_path_length(Gm_synthetic_AA),nx.average_shortest_path_length(Gm_synthetic_RA)]

	# plt.bar(keys, values_average_shortest_path_length, width=0.4, color='b', align='center',label='average_shortest_path_length')
	# plt.xlabel("Methods")
	# plt.ylabel("average_shortest_path_length")
	# plt.savefig('average_shortest_path_length.png',dpi=400)
	# plt.close()

	print("Modularity")
	print(nx_comm.modularity(Gm,nx_comm.label_propagation_communities(Gm)),nx_comm.modularity(Gm_synthetic_MINM,nx_comm.label_propagation_communities(Ga_synthetic_MINM)),nx_comm.modularity(Gm_synthetic_CN,nx_comm.label_propagation_communities(Gm_synthetic_CN)),nx_comm.modularity(Gm_synthetic_PA,nx_comm.label_propagation_communities(Gm_synthetic_PA)),
		nx_comm.modularity(Gm_synthetic_JC,nx_comm.label_propagation_communities(Gm_synthetic_JC)),nx_comm.modularity(Gm_synthetic_AA,nx_comm.label_propagation_communities(Gm_synthetic_AA)),nx_comm.modularity(Gm_synthetic_RA,nx_comm.label_propagation_communities(Gm_synthetic_RA)), nx_comm.modularity(Gm_synthetic_Proposed,nx_comm.label_propagation_communities(Gm_synthetic_Proposed)),nx_comm.modularity(Gm_synthetic_Random,nx_comm.label_propagation_communities(Gm_synthetic_Random)))
	
	values_Modularity = [nx_comm.modularity(Gm,nx_comm.label_propagation_communities(Gm)),nx_comm.modularity(Gm_synthetic_MINM,nx_comm.label_propagation_communities(Ga_synthetic_MINM)),nx_comm.modularity(Gm_synthetic_CN,nx_comm.label_propagation_communities(Gm_synthetic_CN)),nx_comm.modularity(Gm_synthetic_PA,nx_comm.label_propagation_communities(Gm_synthetic_PA)),
		nx_comm.modularity(Gm_synthetic_JC,nx_comm.label_propagation_communities(Gm_synthetic_JC)),nx_comm.modularity(Gm_synthetic_AA,nx_comm.label_propagation_communities(Gm_synthetic_AA)),nx_comm.modularity(Gm_synthetic_RA,nx_comm.label_propagation_communities(Gm_synthetic_RA))]

	# plt.bar(keys, values_Modularity, width=0.4, color='b', align='center',label='Modularity')
	# plt.xlabel("Methods")
	# plt.ylabel("Modularity")
	# plt.savefig('Modularity.png',dpi=400)
	# plt.close()

	print('Rich club coefficient')
	print(getRichClub(Gm),getRichClub(Gm_synthetic_MINM),getRichClub(Gm_synthetic_CN),getRichClub(Gm_synthetic_PA),
		getRichClub(Gm_synthetic_JC),getRichClub(Gm_synthetic_AA), getRichClub(Gm_synthetic_RA) ,getRichClub(Gm_synthetic_Proposed),getRichClub(Gm_synthetic_Random))


	#t1=get_emb_vector_all_nodes(Gn)
	#print(get_emb_vector_similarity(Gn,12,15))
########################################################################################################

	print('-------------------------------')
	print('Ga')
	print("average_clustering coefficient")
	print(nx.average_clustering(Ga), nx.average_clustering(Ga_synthetic_MINM),nx.average_clustering(Ga_synthetic_CN),nx.average_clustering(Ga_synthetic_PA),
		nx.average_clustering(Ga_synthetic_JC),nx.average_clustering(Ga_synthetic_AA),nx.average_clustering(Ga_synthetic_RA),nx.average_clustering(Ga_synthetic_Proposed),nx.average_clustering(Ga_synthetic_Random))
	
	print("Transitivity")
	print(nx.transitivity(Ga),nx.transitivity(Ga_synthetic_MINM),nx.transitivity(Ga_synthetic_CN),nx.transitivity(Ga_synthetic_PA),
		nx.transitivity(Ga_synthetic_JC),nx.transitivity(Ga_synthetic_AA),nx.transitivity(Ga_synthetic_RA),nx.transitivity(Ga_synthetic_Proposed),nx.transitivity(Ga_synthetic_Random))

	print("local_efficiency")
	print(nx.local_efficiency(Ga),nx.local_efficiency(Ga_synthetic_MINM),nx.local_efficiency(Ga_synthetic_CN),nx.local_efficiency(Ga_synthetic_PA),
		nx.local_efficiency(Ga_synthetic_JC),nx.local_efficiency(Ga_synthetic_AA),nx.local_efficiency(Ga_synthetic_RA),nx.local_efficiency(Ga_synthetic_Proposed),nx.local_efficiency(Ga_synthetic_Random))

	print("global_efficiency")
	print(nx.global_efficiency(Ga),nx.global_efficiency(Ga_synthetic_MINM),nx.global_efficiency(Ga_synthetic_CN),nx.global_efficiency(Ga_synthetic_PA),
		nx.global_efficiency(Ga_synthetic_JC),nx.global_efficiency(Ga_synthetic_AA),nx.global_efficiency(Ga_synthetic_RA) ,nx.global_efficiency(Ga_synthetic_Proposed),nx.global_efficiency(Ga_synthetic_Random))

	print("average_shortest_path_length")
	print(nx.average_shortest_path_length(Ga),nx.average_shortest_path_length(Ga_synthetic_MINM),nx.average_shortest_path_length(Ga_synthetic_CN),nx.average_shortest_path_length(Ga_synthetic_PA),
		nx.average_shortest_path_length(Ga_synthetic_JC),nx.average_shortest_path_length(Ga_synthetic_AA),nx.average_shortest_path_length(Ga_synthetic_RA),nx.average_shortest_path_length(Ga_synthetic_Proposed),nx.average_shortest_path_length(Ga_synthetic_Random))

	print("Modularity")
	print(nx_comm.modularity(Ga,nx_comm.label_propagation_communities(Ga)),nx_comm.modularity(Ga_synthetic_MINM,nx_comm.label_propagation_communities(Ga_synthetic_MINM)),nx_comm.modularity(Ga_synthetic_CN,nx_comm.label_propagation_communities(Ga_synthetic_CN)),nx_comm.modularity(Ga_synthetic_PA,nx_comm.label_propagation_communities(Ga_synthetic_PA)),
		nx_comm.modularity(Ga_synthetic_JC,nx_comm.label_propagation_communities(Ga_synthetic_JC)),nx_comm.modularity(Ga_synthetic_AA,nx_comm.label_propagation_communities(Ga_synthetic_AA)),nx_comm.modularity(Ga_synthetic_RA,nx_comm.label_propagation_communities(Ga_synthetic_RA)), nx_comm.modularity(Ga_synthetic_Proposed,nx_comm.label_propagation_communities(Ga_synthetic_Proposed)),nx_comm.modularity(Ga_synthetic_Random,nx_comm.label_propagation_communities(Gm_synthetic_Random)))


	print('Rich club coefficient')
	print(getRichClub(Ga),getRichClub(Ga_synthetic_MINM),getRichClub(Ga_synthetic_CN),getRichClub(Ga_synthetic_PA),
		getRichClub(Ga_synthetic_JC),getRichClub(Ga_synthetic_AA), getRichClub(Ga_synthetic_RA) ,getRichClub(Ga_synthetic_Proposed),getRichClub(Ga_synthetic_Random))
