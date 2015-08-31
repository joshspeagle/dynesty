"""
======
Eggbox
======

A likelihood surface with multiple modes of equal height.
"""

import numpy as np
import matplotlib.pyplot as plt

import nestle


# Define the posterior density to be sampled:
tmax = 5.0 * np.pi
constant = np.log(1.0 / tmax**2)

def loglhood(x):
    t = 2.0 * tmax * x - tmax
    return (2.0 + np.cos(t[0]/2.0)*np.cos(t[1]/2.0))**5.0

def prior(x):
    return x

# plot the surface
t0, t1 = np.meshgrid(np.linspace(0., 1., 50),
                     np.linspace(0., 1., 50))
z = loglhood(np.array([t0, t1]))
plt.imshow(z, extent=(0., 1., 0., 1.), cmap='hot')
ax = plt.gca()
fig = plt.gcf()

res = nestle.sample(loglhood, prior, 2, npoints=100, method='multi')
print(res.summary())

plt.figure()
plt.scatter(x=res.samples[:, 0], y=res.samples[:, 1])
