import numpy as np
import os
import networkx as nx
import pandas as pd
from collections import defaultdict
import sys, copy, time
import scipy.io as sci
import re
import sympy
import math
from gensim import corpora
import gensim
from scipy.sparse import csr_matrix
# import hdf5storage
# from networkx.algorithms.traversal import breadth_first_search as bfs
import breadth_first_search as bfs
#from sklearn.preprocessing import normalize
# import collections as cl
# from sklearn.metrics.pairwise import laplacian_kernel
import ot
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.svm import SVC
# from sklearn.preprocessing import normalize
# from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
from sklearn.metrics import make_scorer, accuracy_score


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        # g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = np.zeros((len(g.node_tags), len(tagset)))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))
    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)


def compute_wasserstein_distance(graph_embeddings, sinkhorn=False, 
                                    categorical=False, sinkhorn_lambda=1e-2):
    '''
    Generate the Wasserstein distance matrix for the graphs embedded 
    in graph_embeddings
    '''
    # Get the iteration number from the embedding file
    n = len(graph_embeddings)
    
    M = np.zeros((n,n))
    # Iterate over pairs of graphs
    for graph_index_1, graph_1 in enumerate(graph_embeddings):
        # Only keep the embeddings for the first h iterations
        labels_1 = graph_embeddings[graph_index_1]
        for graph_index_2, graph_2 in enumerate(graph_embeddings[graph_index_1:]):
            labels_2 = graph_embeddings[graph_index_2 + graph_index_1]
            # Get cost matrix
            ground_distance = 'hamming' if categorical else 'euclidean'
            costs = ot.dist(labels_1, labels_2, metric=ground_distance)

            if sinkhorn:
                mat = ot.sinkhorn(np.ones(len(labels_1))/len(labels_1), 
                                    np.ones(len(labels_2))/len(labels_2), costs, sinkhorn_lambda, 
                                    numItermax=50)
                M[graph_index_1, graph_index_2 + graph_index_1] = np.sum(np.multiply(mat, costs))
            else:
                M[graph_index_1, graph_index_2 + graph_index_1] = \
                    ot.emd2([], [], costs)
                    
    M = (M + M.T)
    return M


def build_multiset(graph_data, maxh, depth):
    prob_map = {}
    graphs = {}
    labels = {}
    alllabels = {}
    num_graphs = len(graph_data)

    for gidx in range(num_graphs):
        adj = nx.to_numpy_array(graph_data[gidx].g)
        graphs[gidx] = adj
        # print(adj)

    for gidx in range(num_graphs):
        label = graph_data[gidx].node_tags
        labels[gidx] = label
    alllabels[0] = labels

    for deep in range(1, maxh):
        labeledtrees = []
        labels_set = set()
        labels = {}
        labels = alllabels[0]
        for gidx in range(num_graphs):
            nx_G = graph_data[gidx].g

            label = labels[gidx]
            #print(label)
            for node in range(len(nx_G)):
                edges = list(bfs.bfs_edges(nx_G, label, source=node, depth_limit=deep)) # 20230223
                #print(edges)
                bfstree = ''
                cnt = 0
                for u, v in edges:
                    bfstree += str(label[int(u)])
                    bfstree += ','
                    bfstree += str(label[int(v)])
                    if cnt < len(list(edges)):
                        bfstree += ','
                    cnt += 1
                # print(bfstree)
                labeledtrees.append(bfstree)
                labels_set.add(bfstree)
        labels_set = list(labels_set)
        labels_set = sorted(labels_set)
        index = 0
        labels = {}
        for gidx in range(num_graphs):
            adj = graphs[gidx]
            n = len(adj)
            label = np.zeros(n)
            for node in range(n):
                label[node] = labels_set.index(labeledtrees[node+index])
            index += n
            labels[gidx] = label
        alllabels[deep] = labels

    # labels = alllabels[0] # 20230223

    paths_graph = []
    for gidx in range(num_graphs):
        nx_G = graph_data[gidx].g
        judge_set = set()
        label = labels[gidx]
        for node in range(len(nx_G)):
            paths_node = []
            paths_node.append(str(node))
            # paths_graph.append(str(node))
            judge_set.add(str(node))
            edges = list(bfs.bfs_edges(nx_G, label, source=node, depth_limit=depth)) # 20230223
            node_in_path = []
            for u, v in edges:
                node_in_path.append(v)
            pathss = []
            for i in range(len(edges)):
                path = list(nx.shortest_path(nx_G, node, node_in_path[i]))
                strpath = ''
                cnt = 0
                for vertex in path:
                    cnt += 1
                    strpath += str(vertex)
                    if cnt < len(path):
                        strpath += ','
                pathss.append(strpath)

            for path in pathss:
                # print(path)
                vertices = re.split(',', path)
                # print(path)
                rvertices = list(reversed(vertices))
                rpath = ''
                cnt = 0
                for rv in rvertices:
                    cnt += 1
                    rpath += rv
                    if cnt < len(rvertices):
                        rpath += ','
                # print(rpath)
                if rpath not in judge_set:
                    # print(path)
                    judge_set.add(path)
                    paths_node.append(path)
                else:
                    paths_node.append(rpath)
            paths_graph.append(paths_node)

    PP = []
    for run in range(maxh):
        labels = alllabels[run]
        labeled_paths_graph = []
        gidx = 0
        index = 0
        for paths_node in paths_graph:
            n = len(graphs[gidx])
            label = labels[gidx]
            labeled_paths_node = []
            for path in paths_node:
                labeled_path = ''
                vertices = re.split(',', path)
                cnt = 0
                for vertex in vertices:
                    cnt += 1
                    labeled_path += str(int(label[int(vertex)]))
                    if cnt < len(vertices):
                        labeled_path += ','
                labeled_paths_node.append(labeled_path)
            labeled_paths_graph.append(labeled_paths_node)
            index += 1
            if index == n:
                gidx += 1
                index = 0
        dictionary = corpora.Dictionary(labeled_paths_graph)
        corpus = [dictionary.doc2bow(labeled_paths_node) for labeled_paths_node in labeled_paths_graph]
        M = gensim.matutils.corpus2csc(corpus)
        M = M.T
        M = M.todense()
        PP.append(M)
    embeddings = np.asarray(np.concatenate(PP, axis=1))
    print(embeddings.shape)

    graph_embedding = []
    index = 0
    num_features = len(embeddings[0])
    for gidx in range(num_graphs):
        n = len(graphs[gidx])
        feature_matrix = np.zeros((n, num_features))
        for node in range(n):
            feature_matrix[node,:] = embeddings[node+index,:]
        index += n
        graph_embedding.append(feature_matrix)
        # graph_embedding.append(feature_matrix)
        # print(sum(feature_matrix))

    return graph_embedding


def custom_grid_search_cv(model, param_grid, precomputed_kernels, y, cv=5):
    '''
    Custom grid search based on the sklearn grid search for an array of precomputed kernels
    '''
    # 1. Stratified K-fold
    cv = StratifiedKFold(n_splits=cv, shuffle=False)
    results = []
    for train_index, test_index in cv.split(precomputed_kernels[0], y):
        split_results = []
        params = [] # list of dict, its the same for every split
        # run over the kernels first
        for K_idx, K in enumerate(precomputed_kernels):
            # Run over parameters
            for p in list(ParameterGrid(param_grid)):
                sc = _fit_and_score(clone(model), K, y, scorer=make_scorer(accuracy_score), 
                        train=train_index, test=test_index, verbose=0, parameters=p, fit_params=None)
                split_results.append(sc.get('test_scores'))
                params.append({'K_idx': K_idx, 'params': p})
        results.append(split_results)
    # Collect results and average
    results = np.array(results)
    # print(results)
    fin_results = results.mean(axis=0)
    # select the best results
    best_idx = np.argmax(fin_results)
    # Return the fitted model and the best_parameters
    ret_model = clone(model).set_params(**params[best_idx]['params'])
    return ret_model.fit(precomputed_kernels[params[best_idx]['K_idx']], y), params[best_idx]


if __name__ == "__main__":
    # location to save the results
    OUTPUT_DIR = "./kernel"
    
    ds_name = sys.argv[1]
    maxh = int(sys.argv[2])
    depth = int(sys.argv[3])

    results_path = "./dataset/" + ds_name + '/'
    gridsearch = True
    crossvalidation = True
    sinkhorn = False
    num_iterations=3
    gamma=None
    categorical = False

    degree_as_tag = False

    # print("dataset: %s" % (ds_name))
    print("dataset=%s, k=%d, d=%d" % (ds_name, maxh - 1, depth - 1))
    graphs, num_classes = load_data(ds_name, degree_as_tag)

    graph_label = []
    for graph in graphs:
        graph_label.append(graph.label)

    start = time.time()

    graph_embedding = build_multiset(graphs, maxh, depth)
    distance_mmatrix = compute_wasserstein_distance(graph_embedding, sinkhorn=sinkhorn, 
                                    categorical=categorical, sinkhorn_lambda=1e-2)

    end = time.time()

    print("eclipsed time: %g" % (end - start))

    if gridsearch:
        # Gammas in eps(-gamma*M):
        # gammas = [0.001]
        gammas = np.logspace(-4,1,num=6)  
        param_grid = [
            {'C': np.logspace(-3,3,num=7)}
        ]
    else:
        gammas = [0.001]

    kernel_matrices = []
    kernel_params = []
    # Generate the full list of kernel matrices from which to select
    M = distance_mmatrix
    for ga in gammas:
        K = np.exp(-ga*M)
        kernel_matrices.append(K)
        kernel_params.append(ga)
    kernel_path = OUTPUT_DIR + '/' + ds_name
    sci.savemat("%s/mpg_kernel_%s_maxh_%d_depth_%d.mat"%(kernel_path, ds_name, maxh - 1, depth - 1), mdict={'kernel': kernel_matrices})
    #---------------------------------
    # Classification
    #---------------------------------
    # Run hyperparameter search if needed
    print(f'Running SVMs, crossvalidation: {crossvalidation}, gridsearch: {gridsearch}.')

    y = np.array(graph_label)

    # Contains accuracy scores for each cross validation step; the
    # means of this list will be used later on.
    accuracy_scores = []
    np.random.seed(42)
    
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    # Hyperparam logging
    best_C = []
    best_gamma = []

    for train_index, test_index in cv.split(kernel_matrices[0], y):
        K_train = [K[train_index][:, train_index] for K in kernel_matrices]
        K_test  = [K[test_index][:, train_index] for K in kernel_matrices]
        y_train, y_test = y[train_index], y[test_index]

        # Gridsearch
        if gridsearch:
            gs, best_params = custom_grid_search_cv(SVC(kernel='precomputed'), 
                    param_grid, K_train, y_train, cv=5)
            # Store best params
            C_ = best_params['params']['C']
            gamma_ = kernel_params[best_params['K_idx']]
            y_pred = gs.predict(K_test[best_params['K_idx']])
        else:
            gs = SVC(C=100, kernel='precomputed').fit(K_train[0], y_train)
            y_pred = gs.predict(K_test[0])
            gamma_, C_ = gammas[0], 100 
        best_C.append(C_)
        best_gamma.append(gamma_)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        if not crossvalidation:
            break
    
    #---------------------------------
    # Printing and logging
    #---------------------------------
    if crossvalidation:
        print('Mean 10-fold accuracy: {:2.2f} +- {:2.2f} %'.format(
                    np.mean(accuracy_scores) * 100,  
                    np.std(accuracy_scores) * 100))
    else:
        print('Final accuracy: {:2.3f} %'.format(np.mean(accuracy_scores)*100))

    # Save to file
    #if crossvalidation or gridsearch:
    #    extension = ''
    #    if crossvalidation:
    #        extension += '_crossvalidation'
    #    if gridsearch:
    #        extension += '_gridsearch'
    #    results_filename = os.path.join(results_path, f'results_{ds_name}'+extension+'.csv')
    #    n_splits = 10 if crossvalidation else 1
    #    pd.DataFrame(np.array([best_C, best_gamma, accuracy_scores]).T, 
    #            columns=[['C', 'gamma', 'accuracy']], 
    #            index=['fold_id{}'.format(i) for i in range(n_splits)]).to_csv(results_filename)
    #    print(f'Results saved in {results_filename}.')
    #else:
    #    print('No results saved to file as --crossvalidation or --gridsearch were not selected.')


