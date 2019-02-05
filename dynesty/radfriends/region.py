from __future__ import print_function, division
"""

RadFriends region with transforms
----------------------------------

Copyright (c) 2017 Johannes Buchner

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy
import scipy.spatial, scipy.cluster
from .neighbors import find_rdistance, is_within_distance_of, count_within_distance_of, any_within_distance_of
from collections import defaultdict

class ClusterResult(object):
	def __init__(self, points, clusters, metric, verbose=False):
		self.ws = points
		self.clusters = clusters
		self.metric = metric
		if verbose:
			print('CLUSTERS:')
			for cluster in clusters:
				clusterpoints = metric.untransform(points[cluster,:])
				print('CLUSTER:', clusterpoints.mean(axis=0), clusterpoints.std(axis=0))
	
	def get_cluster_id(self, point):
		w = self.metric.transform(point)
		dists = scipy.spatial.distance.cdist(self.ws, [w], metric='euclidean')
		i = numpy.argmin(dists)
		for j, cluster in enumerate(self.clusters):
			if i in cluster:
				return j
	
	def get_cluster_ids(self, points):
		ws = self.metric.transform(points)
		dists = scipy.spatial.distance.cdist(self.ws, ws, metric='euclidean')
		i = numpy.argmin(dists, axis=0)
		assert len(i) == len(points)
		results = []
		for ii in i:
			for j, cluster in enumerate(self.clusters):
				if ii in cluster:
					results.append(j)
		return results
	
	def get_n_clusters(self):
		return len(self.clusters)

class RadFriendsRegion(object):
	def __init__(self, members, maxdistance=None, metric='euclidean', nbootstraps=10, verbose=False):
		self.members = members
		assert metric == 'euclidean'
		if maxdistance is None:
			maxdistance = find_rdistance(members, nbootstraps=nbootstraps, 
				metric=metric, verbose=verbose)
			# print 'new RadFriendsRegion with r=', maxdistance
		self.maxdistance = maxdistance
		self.metric = metric
		self.verbose = verbose
		self.lo = numpy.min(self.members, axis=0) - self.maxdistance
		self.hi = numpy.max(self.members, axis=0) + self.maxdistance
	
	def add_members(self, us):
		self.members = numpy.vstack((self.members, us))
		self.lo = numpy.min(self.members, axis=0) - self.maxdistance
		self.hi = numpy.max(self.members, axis=0) + self.maxdistance
	
	def are_near_members(self, us):
		dists = scipy.spatial.distance.cdist(self.members, us, metric=self.metric)
		dist_criterion = dists < self.maxdistance
		return dist_criterion
	
	def count_nearby_members(self, us):
		return count_within_distance_of(self.members, self.maxdistance, us)
	
	def get_nearby_member_ids(self, u):
		return numpy.where(self.are_near_members([u]))[0]
	
	def is_inside(self, u):
		# is it true for at least one?
		if not ((u >= self.lo).all() and (u <= self.hi).all()):
			return False
		return is_within_distance_of(self.members, self.maxdistance, u)
		#return self.are_near_members([u]).any()
	
	def are_inside(self, us):
		# is it true for at least one?
		#return self.are_near_members(us).any(axis=0)
		return any_within_distance_of(self.members, self.maxdistance, us)
	
	def get_clusters(self):
		# agglomerate clustering of members
		dists = scipy.spatial.distance.cdist(self.members, self.members, metric=self.metric)
		connected = dists < self.maxdistance
		nmembers = len(self.members)
		cluster = dict([(i,i) for i in range(nmembers)])
		for i in range(nmembers):
			neighbors = numpy.where(connected[i,:])[0] #[i+1:]
			for j in neighbors:
				cluster[j] = cluster[i]
		result = defaultdict(list)
		for element, cluster_nro in list(cluster.items()):
			result[cluster_nro].append(element)
		#print 'RadFriends: %d clusters' % len(result)
		return result
		
	
	def generate(self, nmax=0):
		members = self.members
		maxdistance = self.maxdistance
		nmembers, ndim = numpy.shape(self.members)
		# how many points to try to generate
		# if too small, many function calls, inefficient
		# if too large, large cdist matrices, spikes in memory use
		N = 1000
		verbose = self.verbose
		nall = 0
		ntotal = 0
		#print 'draw from radfriends'
		while nmax == 0 or nall < nmax:
			#print 'drew %d/%d so far' % (N, nmax)
			# draw from box
			# this can be efficient if there are a lot of points
			ntotal = ntotal + N
			nall += N
			us = numpy.random.uniform(self.lo, self.hi, size=(N, ndim))
			mask = self.are_inside(us)
			#print 'accepted %d/%d [box draw]' % (mask.sum(), N)
			if mask.any():
				yield us[mask,:], ntotal
				#for u in us[mask,:]:
				#	#print 'box draw success:', ntotal
				#	yield u, ntotal
				ntotal = 0
			
			# draw from points
			# this can be efficient in higher dimensions
			us = members[numpy.random.randint(0, len(members), N),:]
			ntotal = ntotal + N
			nall += N
			if verbose: print('chosen point', us)
			# draw direction around it
			direction = numpy.random.normal(0, 1, size=(N, ndim))
			direction = direction / ((direction**2).sum(axis=1)**0.5).reshape((-1,1))
			if verbose: print('chosen direction', direction)
			# choose radius: volume gets larger towards the outside
			# so give the correct weight with dimensionality
			radius = maxdistance * numpy.random.uniform(0, 1, size=(N,1))**(1./ndim)
			us = us + direction * radius
			#mask = numpy.logical_and((u >= self.lo).all(axis=0), (u <= self.hi).all(axis=0))
			#if not mask.any():
			#	if verbose: print 'rejection because outside'
			#	continue
			#us = us[mask,:]
			#if verbose: print 'using point', us
			# count the number of points this is close to
			nnear = self.count_nearby_members(us)
			if verbose: print('near', nnear)
			# accept with probability 1./nnear
			coin = numpy.random.uniform(size=len(us))
			
			accept = coin < 1. / nnear
			#print 'accepted %d/%d [point draw]' % (accept.sum(), N)
			if not accept.any():
				if verbose: print('probabilistic rejection due to overlaps')
				continue
			#print '  overlaps accepted %d of %d, typically %.2f neighbours' % (accept.sum(), N, nnear.mean())
			us = us[accept,:]
			yield us, ntotal
			#for u in us:
			#	#print 'ball draw success:', ntotal
			#	yield u, ntotal
			ntotal = 0

