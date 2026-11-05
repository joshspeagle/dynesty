"""
Skeleton of a restructured Sampler (design sketch, not imported anywhere).

One primitive, `Sampler.draw(loglstar)`, produces a point from the prior
with logl > loglstar. Everything else is built from it:

    initial live points        -> _initialize_live_points (kept as today)
    unit cube sampling         -> draw() while the volume is large
    bound + internal sampler   -> draw() once the volume is small
    nested sampling iterations -> sample(): kill worst, record, draw()
    final live points          -> final_points(): computed, nothing changes
    batch starting points      -> draw(logl_min) on a sampler spawned from
                                  reference points above logl_min
    batch sampling             -> sample() on a sampler spawned from them

NestedSampler and DynamicNestedSampler only hold Samplers, decide what to
run, and merge the results. They never write a Sampler's attributes.

"as today" marks code that moves over unchanged from sampler.py.
"""
import copy
import math
import numpy as np

from dynesty.bounding import UnitCube
from dynesty.internal_samplers import UnitCubeSampler
from dynesty.sampler import _get_bound, _initialize_live_points
from dynesty.utils import RunRecord, progress_integration


class Sampler:
    """
    One nested sampling run over one set of live points.

    Settings are fixed for the lifetime of the object. The state is in four
    groups (live set, integration, proposals, call accounting); each is
    listed in __init__ and changed only by the methods noted there.
    """

    def __init__(
            self,
            # --- settings: the problem
            loglikelihood,
            prior_transform,
            ndim,
            # --- settings: how new points are proposed
            sampling,  # InternalSampler: options + tuning to start from
            bounding,
            ncdim=None,
            bound_update_interval=None,  # in likelihood calls
            first_update=None,  # {'min_eff': percent}, see _cube_phase()
            bound_bootstrap=None,
            bound_enlarge=None,
            # --- settings: execution
            rstate=None,
            pool=None,
            use_pool=None,
            queue_size=None,
            blob=False,
            cite=None,
            # --- the start
            live_points=None,  # (u, v, logl[, blobs]), all above the start
            logvol_init=0.):
        # ---------------- settings (never change) ----------------
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.ndim = ndim
        self.ncdim = ncdim or ndim
        self.bounding = bounding
        self.bound_update_interval = bound_update_interval
        self.first_update = first_update or {}
        self.bound_bootstrap = bound_bootstrap
        self.bound_enlarge = bound_enlarge
        self.rstate = rstate
        self.pool = pool
        self.use_pool = use_pool or {}
        self.queue_size = queue_size
        self.blob = blob
        self.cite = cite

        # ---------------- live set: changed by sample() ----------------
        self.live_u, self.live_v, self.live_logl = live_points[:3]
        self.live_blobs = live_points[3] if blob else None
        self.nlive = len(self.live_u)
        self.live_it = np.zeros(self.nlive, dtype=int)  # birth iteration
        self.live_bound = np.zeros(self.nlive, dtype=int)  # bound drawn from

        # ---------------- integration: changed by sample() ----------------
        self.logvol_init = logvol_init
        self.logvol = logvol_init  # ln(prior volume) above the worst point
        self.logz = -1.e300
        self.h = 0.
        self.loglstar = -np.inf
        self.it = 1
        self.saved_run = RunRecord()  # dead points only
        self.plateau_counter = 0  # >0 while going through a plateau
        self.plateau_logdvol = None

        # ---------------- proposals: changed by draw() ----------------
        # this run's own copy; spawn() passes a copy with the tuning to keep
        self.internal_sampler = sampling
        self.bound = UnitCube(self.ncdim)  # the configured bound once fitted
        self.unit_cube_sampling = True
        self.queue = []

        # ---------------- call accounting: changed by draw() ----------------
        self.ncall = 0  # likelihood calls made by this sampler
        self.ncall_at_last_update = 0
        self.bound_list = [self.bound]  # bounds fitted so far (if saved)
        self.nbound = 1  # bounds fitted so far (saved or not)
        self.save_bounds = True
        self.logl_first_update = None  # record: logl at the switch

        # The phase follows from the start: a sampler that starts in a
        # small volume (e.g. a batch above logl_min) starts with a bound
        # fitted to its own live points, which it is then drawn from.
        if not self._cube_phase():
            self._switch_to_bound()
            self.live_bound[:] = self.nbound - 1

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _settings(self):
        """Constructor arguments that are settings (no start, no state)."""
        return dict(loglikelihood=self.loglikelihood,
                    prior_transform=self.prior_transform,
                    ndim=self.ndim,
                    sampling=self.internal_sampler,
                    bounding=self.bounding,
                    ncdim=self.ncdim,
                    bound_update_interval=self.bound_update_interval,
                    first_update=self.first_update,
                    bound_bootstrap=self.bound_bootstrap,
                    bound_enlarge=self.bound_enlarge,
                    rstate=self.rstate,
                    pool=self.pool,
                    use_pool=self.use_pool,
                    queue_size=self.queue_size,
                    blob=self.blob,
                    cite=self.cite)

    def spawn(self, live_points, logvol_init=0., scale=None, **settings):
        """
        A new run with the same settings (optionally overriding some, e.g.
        bound_update_interval) starting from `live_points`. It gets a copy
        of this run's internal sampler, so it starts from the same tuning
        (with `scale` replaced if given) without affecting this run.
        """
        sampling = copy.deepcopy(self.internal_sampler)
        if scale is not None:
            sampling.scale = scale
        return Sampler(live_points=live_points,
                       logvol_init=logvol_init,
                       **{
                           **self._settings(), 'sampling': sampling,
                           **settings
                       })

    # ------------------------------------------------------------------
    # proposals: the one primitive
    # ------------------------------------------------------------------
    def _cube_phase(self):
        """
        Sample the unit cube while that is efficient enough. Its acceptance
        rate is the prior volume above the threshold, exp(logvol), so the
        rule only depends on the volume. The switch happens once.
        """
        min_eff = self.first_update.get('min_eff', 3.)
        return (self.unit_cube_sampling
                and self.logvol > math.log(min_eff / 100.))

    def _switch_to_bound(self):
        self.unit_cube_sampling = False
        self.logl_first_update = self.loglstar
        self.bound = _get_bound(
            self.bounding,
            self.ndim,
            ncdim=self.ncdim,
            periodic=self.internal_sampler.sampler_kwargs.get('periodic'))
        self._update_bound()

    def _update_bound(self):
        """Refit the bound to the live points above loglstar (as today)."""
        ...
        if self.save_bounds:
            self.bound_list.append(copy.deepcopy(self.bound))
        self.nbound += 1
        self.ncall_at_last_update = self.ncall

    @property
    def scale(self):
        """Scale of the proposals being made (1 in the unit cube)."""
        return 1 if self.unit_cube_sampling else self.internal_sampler.scale

    def draw(self, loglstar):
        """
        Return a new point from the prior with logl > loglstar, together
        with the number of calls it took, its proposal statistics and the
        index of the bound it was drawn from. The live set is only used as
        a reference (bound fitting, MCMC starting points); it is not changed.
        """
        if self.unit_cube_sampling and not self._cube_phase():
            self._switch_to_bound()
        sampler = (UnitCubeSampler(ndim=self.ndim)
                   if self.unit_cube_sampling else self.internal_sampler)
        nc = 0
        while True:
            if not self.queue:
                self._fill_queue(sampler, loglstar)  # as today
            ret = self.queue.pop(0)
            nc += ret.ncalls
            self.ncall += ret.ncalls
            if ret.tuning_info is not None and not self.unit_cube_sampling:
                self.internal_sampler.tune(ret.tuning_info,
                                           update=not self.queue)
            if (not self.unit_cube_sampling and not self.queue
                    and self.ncall >= self.ncall_at_last_update +
                    self.bound_update_interval):
                self._update_bound()
            if ret.logl > loglstar:
                bound_index = 0 if self.unit_cube_sampling else self.nbound - 1
                return ret.u, ret.v, ret.logl, nc, ret.proposal_stats, \
                    bound_index

    # ------------------------------------------------------------------
    # nested sampling
    # ------------------------------------------------------------------
    def sample(self, maxiter=None, maxcall=None, dlogz=0.01, logl_max=np.inf):
        """
        Generator of dead points. Each iteration kills the worst live point,
        records it with its volume, and replaces it by draw(its logl).
        Calling it again continues the run: the integration state is in
        attributes, so there is no separate 'resume' or 'remove live' logic.
        """
        while not self._should_stop(maxiter, maxcall, dlogz, logl_max):
            worst = np.argmin(self.live_logl)
            dlv = self._volume_step(worst)  # handles plateaus (as today)
            self.logvol -= dlv
            loglstar_new = self.live_logl[worst]
            dead = self._dead_record(worst)  # copies of u, v, logl, blob...
            u, v, logl, nc, stats, bound_index = self.draw(loglstar_new)
            logwt, self.logz, self.h = progress_integration(
                self.loglstar, loglstar_new, self.logz, self.logvol, dlv,
                self.h)
            self.loglstar = loglstar_new
            dead.update(logvol=self.logvol,
                        logwt=logwt,
                        logz=self.logz,
                        h=self.h,
                        nc=nc,
                        bounditer=bound_index,
                        scale=self.scale,
                        proposal_stats=stats)
            self.saved_run.append(dead)
            self._replace(worst, u, v, logl, bound_index)  # sets live_it
            self.it += 1
            yield dead

    def final_points(self):
        """
        Records for the remaining live points, treated as dead points with
        the expected volumes (nlive + 1 - i) / (nlive + 1) of the current
        volume (as today's add_live_points, plateau included). Nothing in
        the sampler changes, so sampling can simply continue afterwards.
        """
        ...

    @property
    def results(self):
        """saved_run plus final_points(), packed into a Results object."""
        ...

    @property
    def eff(self):
        """Overall efficiency in percent, for reporting only."""
        return 100. * (self.it - 1) / max(self.ncall, 1)


# ----------------------------------------------------------------------
# supervisors
# ----------------------------------------------------------------------
class NestedSampler:
    """Holds one Sampler, started from fresh prior draws."""

    def __init__(self, loglikelihood, prior_transform, ndim, nlive=500,
                 **settings):
        live_points, logvol_init, self.init_ncall = _initialize_live_points(
            ...)
        self.sampler = Sampler(loglikelihood=loglikelihood,
                               prior_transform=prior_transform,
                               ndim=ndim,
                               live_points=live_points,
                               logvol_init=logvol_init,
                               **settings)

    def run_nested(self, print_progress=True, **stop_kwargs):
        for dead in self.sampler.sample(**stop_kwargs):
            ...  # print progress
        for dead in self.sampler.final_points():
            ...  # print progress

    @property
    def results(self):
        return self.sampler.results  # plus init_ncall in the call count


class DynamicNestedSampler:
    """
    Holds the base run Sampler and runs batch Samplers spawned from it.
    Owns the merged run, the weight/stopping functions and the batch policy.
    """

    def _run_base(self, nlive, **stop_kwargs):
        live_points, logvol_init, ncall = _initialize_live_points(...)
        self.sampler = Sampler(live_points=live_points,
                               logvol_init=logvol_init,
                               **self.settings)
        ...  # sample(), final_points(), store as the base run

    def _batch_sampler(self, nlive, logl_min, update_interval):
        """
        The sampler for one batch: nlive independent points above
        logl_min, in a run that starts from the volume at logl_min.
        Returns the sampler and the calls spent on its starting points.
        """
        if logl_min == -np.inf:
            live_points, logvol_init, ncall = _initialize_live_points(...)
            return self.sampler.spawn(
                live_points,
                logvol_init=logvol_init,
                bound_update_interval=update_interval), ncall
        # saved points above logl_min, weighted to be roughly uniform
        # (as today), and the volume at logl_min from the merged run
        ref_points, ref_scale, logvol_min = self._reference_points(
            logl_min, nlive)
        ref = self.sampler.spawn(ref_points,
                                 logvol_init=logvol_min,
                                 scale=ref_scale,
                                 bound_update_interval=update_interval)
        new = [ref.draw(logl_min) for _ in range(nlive)]
        u, v, logl, nc, stats, bound_index = map(list, zip(*new))
        # starts from ref's tuning, fits its own bound to its own points
        return ref.spawn(_pack(u, v, logl, self.blob),
                         logvol_init=logvol_min), sum(nc)

    def sample_batch(self, nlive, logl_bounds, update_interval, **stop):
        logl_min, logl_max = logl_bounds
        batch, ncall = self._batch_sampler(nlive, logl_min, update_interval)
        for dead in batch.sample(logl_max=logl_max, **stop):
            ...  # store in new_run, shifting bound indices
        for dead in batch.final_points():
            ...  # store in new_run
        ...  # merge new_run into the merged run; keep batch.bound_list


def _pack(u, v, logl, blob):
    """(u, v, logl[, blobs]) arrays from lists of drawn points."""
    blobs = [_.blob for _ in logl] if blob else None
    return np.array(u), np.array(v), np.array(logl, dtype=float), blobs
