from __future__ import print_function, division
"""

Implementation of RadFriends
https://arxiv.org/abs/1407.5459
Uses standardised euclidean distance, which makes it fast.

Copyright (c) 2017 Johannes Buchner

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



"""

import numpy
import scipy.spatial, scipy.cluster
import matplotlib.pyplot as plt
from collections import defaultdict
#from .jarvispatrick import jarvis_patrick_clustering, jarvis_patrick_clustering_iterative
from .metric import IdentityMetric, SimpleScaling, TruncatedScaling
from .region import ClusterResult, RadFriendsRegion

class MetricLearningFriendsConstrainer(object):
	def __init__(self, metriclearner, rebuild_every = 50, metric_rebuild_every = 50, verbose = False,
			force_shrink=False):
		self.iter_since_metric_rebuild = 0
		self.ndraws_since_rebuild = 0
		self.region = None
		self.rebuild_every = int(rebuild_every)
		self.metric_rebuild_every = int(metric_rebuild_every)
		self.verbose = verbose
		self.force_shrink = force_shrink
		self.metriclearner = metriclearner
		self.metric = IdentityMetric()
		self.clusters = None
		self.direct_draws_efficient = True
		self.last_cluster_points = None
		self.prev_maxdistance = None
	
	def cluster(self, u, ndim, keepMetric=False):
		"""
		1) Splits live points into clusters using Jarvis-Patrick K=1 clustering
		2) Project new clusters onto old clusters for identification tree.
		   If new cluster encompasses more than one old cluster: 
		3) Overlay all clusters (shift by cluster mean) and compute new metric (covariance)
		4) Using original points and new metric, compute RadFriends bootstrapped distance and store
		5) In each RadFriends cluster, find points.
		6) If still mono-mode: no problem
		   If discovered new clusters in (1): store filtering function and cluster assignment
		   If no new clusters: no problem
		"""
		w = self.metric.transform(u)
		prev_region = self.region
		if keepMetric:
			self.region = RadFriendsRegion(members=w)
			if self.force_shrink and self.region.maxdistance > self.prev_maxdistance:
				self.region = RadFriendsRegion(members=w, maxdistance=self.prev_maxdistance)
			self.prev_maxdistance = self.region.maxdistance
			if self.verbose: print('keeping metric, not reclustering.')
			return
		
		metric_updated = False
		clustermetric = self.metric
		"""
		if self.verbose: print('computing distances for clustering...')
		wdists = scipy.spatial.distance.cdist(w, w, metric='euclidean')
		# apply Jarvis-Patrick clustering
		if self.verbose: print('Clustering...')
		clusters = jarvis_patrick_clustering_iterative(wdists, number_of_neighbors=len(wdists), n_stable_iterations=3)
		# Overlay all clusters (shift by cluster mean) 
		if self.verbose: print('Metric update ...')
		shifted_cluster_members = []
		for members in clusters:
			cluster_mean = numpy.mean(u[members,:], axis=0)
			shifted_cluster_members += (u[members,:] - cluster_mean).tolist()
		shifted_cluster_members = numpy.asarray(shifted_cluster_members)
		"""
		if self.verbose: print('Metric update ...')
		cluster_mean = numpy.mean(u, axis=0)
		shifted_cluster_members = u - cluster_mean

		# Using original points and new metric, compute RadFriends bootstrapped distance and store
		if self.metriclearner == 'none':
			metric = self.metric # stay with identity matrix
			metric_updated = False
		elif self.metriclearner == 'simplescaling' or (self.metriclearner == 'mahalanobis' and ndim == 1):
			metric = SimpleScaling()
			metric.fit(shifted_cluster_members)
			metric_updated = True
		elif self.metriclearner == 'truncatedscaling' or (self.metriclearner == 'truncatedmahalanobis' and ndim == 1):
			metric = TruncatedScaling()
			metric.fit(shifted_cluster_members)
			metric_updated = self.metric == IdentityMetric() or not numpy.all(self.metric.scale == metric.scale)
		elif self.metriclearner == 'mahalanobis':
			metric = MahalanobisMetric()
			metric.fit(shifted_cluster_members)
			metric_updated = True
		elif self.metriclearner == 'truncatedmahalanobis':
			metric = TruncatedMahalanobisMetric()
			metric.fit(shifted_cluster_members)
			metric_updated = self.metric == IdentityMetric() or not (numpy.all(self.metric.scale == metric.scale) and numpy.all(self.metric.cov == metric.cov))
		elif self.metriclearner == 'sdml':
			metric = SDML()
			metric.fit(shifted_cluster_members, W = numpy.ones((len(w), len(w))))
			metric_updated = True
		elif self.metriclearner == 'truncatedsdml':
			metric = TruncatedSDML()
			metric.fit(shifted_cluster_members, W = numpy.ones((len(w), len(w))))
			metric_updated = self.metric == IdentityMetric() or not (numpy.all(self.metric.scale == metric.scale) and numpy.all(self.metric.cov == metric.cov))
		else:
			assert False, self.metriclearner
		
		self.metric = metric
		
		#oldclusters = self.clusters
		#self.clusters = clusters
		
		wnew = self.metric.transform(u)
		if self.verbose: print('Region update ...')
		
		self.region = RadFriendsRegion(members=wnew) #, maxdistance=shifted_region.maxdistance)
		if not metric_updated and self.force_shrink and self.prev_maxdistance is not None:
			if self.region.maxdistance > self.prev_maxdistance:
				self.region = RadFriendsRegion(members=w, maxdistance=self.prev_maxdistance)
		self.prev_maxdistance = self.region.maxdistance
		if self.verbose: print('done.')
	
	def are_inside_cluster(self, points):
		w = self.metric.transform(points)
		return self.region.are_inside(w)
	
	def is_inside(self, point):
		if not ((point >= 0).all() and (point <= 1).all()):
			return False
		w = self.metric.transform(point)
		return self.region.is_inside(w)

	def generate(self, ndim):
		ntotal = 0
		N = 10000
		while True:
			#if numpy.random.uniform() < 0.01:
			if ndim < 40:
				# draw from radfriends directly
				for ws, n in self.region.generate(N):
					us = self.metric.untransform(ws)
					assert us.shape[1] == ndim, us.shape
					ntotal = ntotal + n
					mask = numpy.logical_and(us < 1, us > 0).all(axis=1)
					assert mask.shape == (len(us),), (mask.shape, us.shape)
					if mask.any():
						#print 'radfriends draw in unit cube:', mask.sum(), ntotal
						for u in us[mask,:]:
							assert u.shape == (us[0].shape), (u.shape, us.shape, mask.shape)
							yield u, ntotal
							ntotal = 0
					#if all([0 <= ui <= 1 for ui in u]):
					#	yield u, ntotal
					#	ntotal = 0
			if numpy.random.uniform() < 0.1:
				# draw from unit cube
				# this can be efficient if volume still large
				ntotal = ntotal + N
				us = numpy.random.uniform(size=(N, ndim))
				ws = self.metric.transform(us)
				nnear = self.region.are_inside(ws)
				#print '  %d of %d accepted' % (nnear.sum(), N)
				for u in us[nnear,:]:
					#print 'unit cube draw success:', ntotal
					yield u, ntotal
					ntotal = 0
	
	def rebuild(self, u, ndim, keepMetric=False):
		if self.last_cluster_points is not None and \
			len(self.last_cluster_points) == len(u) and \
			numpy.all(self.last_cluster_points == u):
			# do nothing if everything stayed the same
			return
		
		self.cluster(u=u, ndim=ndim, keepMetric=keepMetric)
		self.last_cluster_points = u
		
		print('maxdistance:', self.region.maxdistance)
		self.generator = self.generate(ndim)
	
	def _draw_constrained_prepare(self, Lmins, priortransform, loglikelihood, live_pointsu, ndim, **kwargs):
		rebuild = self.ndraws_since_rebuild > self.rebuild_every or self.region is None
		rebuild_metric = self.iter_since_metric_rebuild > self.metric_rebuild_every
		keepMetric = not rebuild_metric
		if rebuild:
			print('rebuild triggered at call')
			self.rebuild(numpy.asarray(live_pointsu), ndim, keepMetric=keepMetric)
			self.ndraws_since_rebuild = 0
			if rebuild_metric:
				self.iter_since_metric_rebuild = 0
		else:
			#print 'no rebuild: %d %d' % (self.iter_since_metric_rebuild, self.ndraws_since_rebuild)
			rebuild_metric = False
		assert self.generator is not None
		return rebuild, rebuild_metric
	
	def get_Lmax(self):
		return None

	def draw_constrained(self, Lmins, priortransform, loglikelihood, live_pointsu, ndim, **kwargs):
		ntoaccept = 0
		ntotalsum = 0
		self.iter_since_metric_rebuild += 1
		#print 'MLFriends trying to replace', Lmins
		rebuild, rebuild_metric = self._draw_constrained_prepare(Lmins, priortransform, loglikelihood, live_pointsu, ndim, **kwargs)
		while True:
			#print '    starting generator ...'
			for u, ntotal in self.generator:
				assert (u >= 0).all() and (u <= 1).all(), u
				ntotalsum += ntotal
				x = priortransform(u)
				L = loglikelihood(x)
				ntoaccept += 1
				self.ndraws_since_rebuild += 1

				#print 'ntotal:', ntotal
				if ntotal > 100000:
					self.direct_draws_efficient = False
				
				if numpy.any(L > Lmins):
					# yay, we win
					#print 'accept after %d tries' % ntoaccept
					return u, x, L, ntoaccept
				
				# if running very inefficient, optimize clustering 
				#     if we haven't done so at the start
				if not rebuild and self.ndraws_since_rebuild > self.rebuild_every:
					rebuild = True
					print('RadFriends rebuild triggered after %d draws' % self.ndraws_since_rebuild)
					self.rebuild(numpy.asarray(live_pointsu), ndim, keepMetric=True)
					self.ndraws_since_rebuild = 0
					break
				if not rebuild_metric and ntoaccept > 200:
					rebuild_metric = True
					print('RadFriends metric rebuild triggered after %d draws' % self.ndraws_since_rebuild)
					self.rebuild(numpy.asarray(live_pointsu), ndim, keepMetric=False)
					self.iter_since_metric_rebuild = 0
					break


