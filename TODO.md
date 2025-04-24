Here's rough todo

* Get rid of ncdim (make it property of rwalk), main sampler doesn't need to know it

* deal with update_interval

* deal with all the sampler options, like walks passed to NestedSampler, deprecate  ?

* implelement a new sampler. Maybe 50% slice, 50%rwalk ?

* Allow saving some information from the sampler into RunRecord ?

* Think of Sampler interface, do i really need seeds ?

* what about generation of livepoints/axes by proposed_live not used by uniform sampler
do i care about that ? probably not 
