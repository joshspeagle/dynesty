from __future__ import print_function, division
"""

Neighbourhood helper functions
-------------------------------

Copyright (c) 2017 Johannes Buchner

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy
import scipy.spatial

def count_within_distance_of(members, maxdistance, us, metric='euclidean'):
	dists = scipy.spatial.distance.cdist(members, us, metric=metric)
	return (dists < maxdistance).sum(axis=0)

def any_within_distance_of(members, maxdistance, us, metric='euclidean'):
	dists = scipy.spatial.distance.cdist(members, us, metric=metric)
	return (dists < maxdistance).any(axis=0)

def is_within_distance_of(xx, maxdistance, y):
	dists = scipy.spatial.distance.cdist(xx, [y], metric='euclidean')
	return (dists < maxdistance).any()

def any_within_distance_of(xx, maxdistance, yy):
	dists = scipy.spatial.distance.cdist(xx, yy, metric='euclidean')
	counts_true = (dists < maxdistance).any(axis=0)
	return counts_true

def bootstrapped_maxdistance(u):
	nsamples, ndim = xx.shape
	chosen = numpy.zeros((nsamples, nbootstraps))
	for b in range(nbootstraps):
		chosen[numpy.random.choice(numpy.arange(nsamples), size=nsamples, replace=True),b] = 1.
	
	maxdistance = lib.bootstrapped_maxdistance(xx, nsamples, ndim, chosen, nbootstraps)
	return maxdistance

def nearest_rdistance_guess(u, metric='euclidean'):
	#if metric == 'euclidean' and most_distant_nearest_neighbor is not None:
	#	return most_distant_nearest_neighbor(u)
	n = len(u)
	distances = scipy.spatial.distance.cdist(u, u, metric=metric)
	numpy.fill_diagonal(distances, 1e300)
	nearest_neighbor_distance = numpy.min(distances, axis = 1)
	rdistance = numpy.max(nearest_neighbor_distance)
	#print 'distance to nearest:', rdistance, nearest_neighbor_distance
	return rdistance

def initial_rdistance_guess(u, metric='euclidean', k = 10):
	n = len(u)
	distances = scipy.spatial.distance.cdist(u, u, metric=metric)
	if k == 1:
	#	numpy.diag(distances)
	#	nearest = [distances[i,:])[1:k] for i in range(n)]
		distances2 = distances + numpy.diag(1e100 * numpy.ones(len(distances)))
		nearest = distances2.min(axis=0)
	else:
		assert False, k
		nearest = [numpy.sort(distances[i,:])[1:k+1] for i in range(n)]
	# compute distance maximum
	rdistance = numpy.max(nearest)
	return rdistance

def update_rdistance(u, ibootstrap, rdistance, verbose = False, metric='euclidean'):
	n, ndim = u.shape
	
	# bootstrap to find smallest rdistance which includes
	# all points
	choice = set(numpy.random.choice(numpy.arange(n), size=n))
	mask = numpy.array([c in choice for c in numpy.arange(n)])
	
	distances = scipy.spatial.distance.cdist(u[mask], u[~mask], metric=metric)
	assert distances.shape == (mask.sum(), (~mask).sum())
	nearest_distance_to_members = distances.min(axis=0)
	if verbose:
		print('nearest distances:', nearest_distance_to_members.max(), nearest_distance_to_members)
	newrdistance = max(rdistance, nearest_distance_to_members.max())
	if newrdistance > rdistance and verbose:
		print(ibootstrap, 'extending:', newrdistance)
	return newrdistance

def find_rdistance(u, verbose=False, nbootstraps=15, metric='euclidean'):
	#if metric == 'euclidean' and bootstrapped_maxdistance is not None:
	#	return bootstrapped_maxdistance(u, nbootstraps)
	# find nearest point for every point
	if verbose: print('finding nearest neighbors:')
	rdistance = 0 #initial_rdistance_guess(u)
	if verbose: print('initial:', rdistance)
	for ibootstrap in range(nbootstraps):
		rdistance = update_rdistance(u, ibootstrap, rdistance, verbose=verbose, metric=metric)
	return rdistance

