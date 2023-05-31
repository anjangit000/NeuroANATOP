import networkx as nx
from scipy import sparse, stats
import numpy as np

def check_threshold(threshold, data, percentile_func, name='threshold'):
    if isinstance(threshold, str):
        message = ('If "{0}" is given as string it '
                   'should be a number followed by the percent '
                   'sign, e.g. "25.3%"').format(name)
        if not threshold.endswith('%'):
            raise ValueError(message)

        try:
            percentile = float(threshold[:-1])
        except ValueError as exc:
            exc.args += (message, )
            raise

        threshold = percentile_func(data, percentile)
    elif isinstance(threshold, numbers.Real):
        # checks whether given float value exceeds the maximum
        # value of the image data
        value_check = abs(data).max()
        if abs(threshold) > value_check:
            warnings.warn("The given float value must not exceed {0}. "
                          "But, you have given threshold={1} ".format(value_check,
                                                                      threshold))
    else:
        raise TypeError('%s should be either a number '
                        'or a string finishing with a percent sign' % (name, ))
    return threshold

def getNetworkxGraphFromAdj(A, edge_threshold="80%"):
	# Make a large figure
	# Mask the main diagonal for visualization:
	np.fill_diagonal(A, 0)
	# decompress input matrix if sparse
	if sparse.issparse(A):
		A = A.toarray()
	# make the lines below well-behaved
	A = np.nan_to_num(A)

	lower_diagonal_indices = np.tril_indices_from(A, k=-1)
	lower_diagonal_values = A[lower_diagonal_indices]
	edge_threshold = check_threshold(edge_threshold, np.abs(lower_diagonal_values), stats.scoreatpercentile, 'edge_threshold')

	print(edge_threshold, max(lower_diagonal_values))
	A = A.copy()
	threshold_mask = np.abs(A) < edge_threshold
	A[threshold_mask] = 0

	#get the networkx graph from numpy matrix
	return nx.from_numpy_matrix(A)


def getNetworkxGraphFromAdjByValue(A, edge_threshold=1.0):
	# Make a large figure
	# Mask the main diagonal for visualization:
	np.fill_diagonal(A, 0)
	# decompress input matrix if sparse
	if sparse.issparse(A):
		A = A.toarray()
	# make the lines below well-behaved
	A = np.nan_to_num(A)

	lower_diagonal_indices = np.tril_indices_from(A, k=-1)
	lower_diagonal_values = A[lower_diagonal_indices]
	#print('Max threshold: ',max(lower_diagonal_values),' and min threshold: ',min(lower_diagonal_values) )

	#edge_threshold = check_threshold(edge_threshold, np.abs(lower_diagonal_values), stats.scoreatpercentile, 'edge_threshold')
	if edge_threshold > max(lower_diagonal_values) or edge_threshold < min(lower_diagonal_values):
		print('please give the value lower than: ',max(lower_diagonal_values),' and higher than: ',min(lower_diagonal_values) )
	A = A.copy()
	threshold_mask = A < edge_threshold
	A[threshold_mask] = 0

	#get the networkx graph from numpy matrix
	return nx.from_numpy_matrix(A)

