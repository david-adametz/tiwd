Translation-Invariant Wishart-Dirichlet Process (TIWD)

-----------
COMPILATION
-----------

* requires a recent version of the Armadillo linear algebra library (http://arma.sourceforge.net)
* requires a recent version of boost_random and boost_math (http://www.boost.org)

to build, simply run:

$ make

-----
USAGE
-----

to get help

$ ./tiwd --help

demo with supplied file...

$ ./tiwd -s S.csv -k 10 -d 200 -e 400 -n 5000 --verbose

... will produce a file 'output_Zflat.csv' with cluster labels (starting with '0'!)

Explanation of input flags

	-s S.csv				pos. def. matrix in comma-separated value format
	-n 5000					number of Gibbs sweeps, 10% are used for burnin
	-k 10						maximum number of clusters allowed (because of the truncated Dirichlet prior)
	-o output				prefix applied to output files, i.e. output_Zflat.csv
	-d 200					number of dimensions, is used as an annealing parameter, this is the starting value
	-e 400					this is the end value of d for annealing
	--verbose				[optional] gives feedback every 100 Gibbs sweeps

----------
David Adametz, 13th Sept 2013
david.adametz@unibas.ch
