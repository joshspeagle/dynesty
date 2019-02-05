# -*- coding: utf-8 -*-

"""
Clustering Using a Similarity Measure Based
on Shared Near Neighbors
R. A. JARVIS AND EDWARD A. PATRICK

A nonparametric clustering technique incorporating the
concept of similarity based on the sharing of near neighbors is presented.
In addition to being an essentially parallel approach, the computational
elegance of the method is such that the scheme is applicable
to a wide class of practical problems involving large sample size and high
dimensionality. No attempt is made to show how a priori problem
knowledge can be introduced into the procedure.

from https://github.com/llazzaro/jarvispatrick

The MIT License (MIT)

Copyright (c) 2016 Leonardo Lazzaro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
from collections import defaultdict
import numpy
import numba

def jarvis_patrick_clustering(dists, number_of_neighbors, threshold_number_of_common_neighbors):
	"""
	:param dists: distance matrix

	:param number_of_neighbors: J, how many neighbors of a point to consider

	:param threshold_number_of_common_neighbors: K, how many neighbors two points have to have in common to be put into the same cluster.
	"""
	if threshold_number_of_common_neighbors > number_of_neighbors:
		raise ValueError('Asked for more common neighbors than number of neighbors')
	
	#print 'jarvis_patrick_clustering'
	# initially each element is a cluster of 1 element
	n = len(dists)
	cluster = dict([(i,i) for i in range(n)])
	# each row of dists, sort
	neighbors_list = numpy.argsort(dists, axis=1)[:,1:number_of_neighbors+1]
	#print 'neighbors_list:', neighbors_list.shape
	neighbors_list = [set(neighbors) for neighbors in neighbors_list]
	#print 'merging neighbors...'
	for element, neighbors in enumerate(neighbors_list):
		for other_element, other_neighbors in enumerate(neighbors_list):
			if other_element >= element: continue
			# we check both sides since the relation is not symmetric
			if element not in other_neighbors or other_element not in neighbors:
				#print 'not within each others neighbor list', element, other_neighbors, other_element, neighbors
				continue
			
			number_of_common_neighbors = len(neighbors.intersection(other_neighbors))
			#print '%d ^ %d : %d in common -> %d or %d' % (element, 
			#	other_element, number_of_common_neighbors, 
			#	cluster[element], cluster[other_element])
			if number_of_common_neighbors < threshold_number_of_common_neighbors:
				continue
			
			i, j = cluster[element], cluster[other_element]
			if i == j: 
				continue
			i, j = min(i, j), max(i, j)
			# move all from j to i
			#print 'merging cluster %d to %d' % (j, i)
			for k in cluster.keys():
				if cluster[k] == j:
					cluster[k] = i
	#print 'merging neighbors done...'
	
	result = defaultdict(list)
	for element, cluster_nro in cluster.items():
		result[cluster_nro].append(element)
	
	#print 'jarvis_patrick_clustering: %d clusters' % len(result)
	assert len(result) <= len(dists), (len(result), len(dists))

	#if fast_jarvis_patrick_clustering is not None:
	#	result2 = fast_jarvis_patrick_clustering(dists, number_of_neighbors, threshold_number_of_common_neighbors)
	#	assert result2 == result.values(), (result.values(), result2)
	#	return result2

	return result.values()

def jarvis_patrick_clustering_iterative(dists, number_of_neighbors, n_stable_iterations=2):
	"""
	Jarvis-Patrick clustering. The number of neighbors to consider 
	for merging points into the same cluster, K, is increased until the 
	result does not change any more.
	"""
	clustering_results = []
	for J in range(2, number_of_neighbors+1):
		#print 'jarvis_patrick_clustering J=%d' % J
		clustering_results.append(jarvis_patrick_clustering(dists, J, 1))
		if len(clustering_results) >= n_stable_iterations:
			first = clustering_results[-1]
			stable = True
			for other in clustering_results[-n_stable_iterations:-1]:
				if other != first:
					stable = False
					break
			if stable: 
				return first

if __name__ == '__main__':
	numpy.random.seed(1)
	import scipy.spatial

	x = numpy.random.uniform(size=(100, 5))
	for number_of_neighbors in range(1, 5):
		threshold_number_of_common_neighbors = 1
		print('JP clustering with ', number_of_neighbors, threshold_number_of_common_neighbors)
		dists = scipy.spatial.distance.cdist(x, x, metric='euclidean')
		result = jarvis_patrick_clustering(dists, number_of_neighbors, threshold_number_of_common_neighbors)
		#result2 = fast_jarvis_patrick_clustering(dists, number_of_neighbors, threshold_number_of_common_neighbors)
		#assert result2 == result, (result, result2)
	
	for i in range(4):
		d = numpy.random.uniform(size=(1000, 10))
		dists = scipy.spatial.distance.cdist(d, d, metric='euclidean')
		jarvis_patrick_clustering_iterative(dists, len(d))


