===============================
References and Acknowledgements
===============================

**The release paper describing the code can be found**
`here <https://github.com/joshspeagle/dynesty/tree/master/paper/dynesty.pdf>`_.

Code
====

``dynesty`` is the spiritual successor to Nested Sampling package `nestle 
<http://kylebarbary.com/nestle/>`_ and has benefited enormously from the work
put in by `Kyle Barbary <http://kylebarbary.com/>`_ and 
`other contributors 
<https://github.com/joshspeagle/dynesty/blob/master/AUTHORS.md>`_.

Much of the API is inspired by the ensemble MCMC package
`emcee <http://dan.iel.fm/emcee/current/>`_ as well as other work by
`Daniel Foreman-Mackey <http://dan.iel.fm/>`_.

Many of the plotting utilities draw heavily upon Daniel Foreman-Mackey's
wonderful `corner <http://corner.readthedocs.io>`_ package.

Several other plotting utilities as well as the real-time status outputs are
inspired in part by features available in the statistical modeling package
`PyMC3 <https://pymc-devs.github.io/pymc3/index.html>`_.

Papers and Texts
================

The dynamic sampling framework was entirely inspired by:

    `Higson et al. 2017b <http://adsabs.harvard.edu/abs/2017arXiv170403459H>`_.
    *Dynamic nested sampling: an improved algorithm for parameter estimation
    and evidence calculation.*
    ArXiv e-prints, 1704.03459.

Much of the nested sampling error analysis is based on:

    `Higson et al. 2017a <http://adsabs.harvard.edu/abs/2017arXiv170309701H>`_.
    *Sampling errors in nested sampling parameter estimation.*
    ArXiv e-prints, 1703.09701.

    `Chopin & Robert 2010
    <http://adsabs.harvard.edu/abs/2008arXiv0801.3887C>`_.
    *Properties of Nested Sampling.*
    Biometrika, 97, 741.

The nested sampling algorithms in
:class:`~dynesty.nestedsamplers.RadFriendsSampler` and
:class:`~dynesty.nestedsamplers.SupFriendsSampler` 
are based on:

    `Buchner 2016 <http://adsabs.harvard.edu/abs/2014arXiv1407.5459B>`_.
    *A statistical test for Nested Sampling algorithms.*
    Statistics and Computing, 26, 383.

Slice sampling and its implementations in nested sampling are based on:

    `Bloem-Reddy & Cunningham 2016
    <http://proceedings.mlr.press/v48/bloem-reddy16.html>`_.
    *Slice Sampling on Hamiltonian Trajectories.*
    PMLR, 48, 3050.

    `Handley, Hobson & Lasenby 2015b
    <http://adsabs.harvard.edu/abs/2015MNRAS.453.4384H>`_.
    *POLYCHORD: next-generation nested sampling.*
    MNRAS, 453, 4384.

    `Handley, Hobson & Lasenby 2015a
    <http://adsabs.harvard.edu/abs/2015MNRAS.450L..61H>`_.
    *POLYCHORD: nested sampling for cosmology.*
    MNRASL, 450, L61.

    `Neal 2003 <https://projecteuclid.org/euclid.aos/1056562461>`_.
    *Slice sampling.* Ann. Statist., 31, 705.

The implementation of multi-ellipsoidal decomposition are based in part on:

    `Feroz et al. 2013 <http://adsabs.harvard.edu/abs/2013arXiv1306.2144F>`_.
    *Importance Nested Sampling and the MultiNest Algorithm.*
    ArXiv e-prints, 1306.2144.

    `Feroz, Hobson & Bridges 2009
    <http://adsabs.harvard.edu/abs/2009MNRAS.398.1601F>`_.
    *MultiNest: an efficient and robust Bayesian inference tool for cosmology
    and particle physics.*
    MNRAS, 398, 1601.

Several useful reference texts include:

    `Salomone et al. 2018
    <https://arxiv.org/abs/1805.03924>`_.
    *Unbiased and Consistent Nested Sampling via Sequential Monte Carlo.*
    ArXiv e-prints, 1805.03924.

    `Walter 2015
    <https://arxiv.org/abs/1412.6368>`_.
    *Point Process-based Monte Carlo estimation.*
    ArXiv e-prints, 1412.6368.

    `Shaw, Bridges & Hobson 2007
    <http://adsabs.harvard.edu/abs/2007MNRAS.378.1365S>`_.
    *Efficient Bayesian inference for multimodal problems in cosmology.*
    MNRAS, 378, 1365.

    `Mukherjee, Parkinson & Liddle 2006
    <http://adsabs.harvard.edu/abs/2006ApJ...638L..51M>`_.
    *A Nested sampling algorithm for cosmological model selection.*
    ApJ, 638, L51.

    `Silvia & Skilling 2006
    <https://global.oup.com/academic/product/data-analysis-9780198568322>`_.
    *Data Analysis: A Bayesian Tutorial, 2nd Edition.*
    Oxford University Press.

    `Skilling 2006 <https://projecteuclid.org/euclid.ba/1340370944>`_.
    *Nested sampling for general Bayesian computation.*
    Bayesian Anal., 1, 833.

    `Skilling 2004 <http://adsabs.harvard.edu/abs/2004AIPC..735..395S>`_.
    *Nested Sampling.*
    In Maximum entropy and Bayesian methods in science and engineering
    (ed. G. Erickson, J.T. Rychert, C.R. Smith).
    AIP Conf. Proc., 735, 395.
