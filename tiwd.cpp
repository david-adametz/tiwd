#include <iostream>
#include <algorithm>
#include <ctime>
#include <armadillo>
#include <boost/random.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <getopt.h>

using namespace std;
using namespace arma;
using namespace boost;

double gamrnd(double shape, double scale, mt19937& rng) {
	gamma_distribution<> gd(shape, scale);
	variate_generator<mt19937&, gamma_distribution<> > g(rng, gd);

	return g();
}

double gampdf(double x, double shape, double scale) {
	double p = 1.0 / (tgamma(shape) * pow(scale, shape)) * pow(x, shape - 1) * exp(-x / scale);
	return p;
}

double unifrnd(mt19937& rng) {
	uniform_01<> ud;
	variate_generator<mt19937&, uniform_01<> > u(rng, ud);

	return u();
}

int main(int argc, char **argv) {
	mt19937 rng;
	rng.seed(time(NULL) + getpid());

	bool verbose = false;

	const char *Sfilename = NULL;
	const char *Ofilename = NULL;

	int n_sample = -1;
	int d_init = -1;
	int d_end = -1;
	int kmax = -1;

	int input;

	opterr = 0;

	static int verbose_flag = 0;

	static struct option long_options[] = { { "verbose", no_argument, &verbose_flag, 1 } };
	int option_index = 0;

	while ((input = getopt_long(argc, argv, "s:o:n:k:d:e:", long_options, &option_index)) != -1) {
		switch (input) {
			case 0:
				if (long_options[option_index].flag != 0)
					break;
				printf("option %s", long_options[option_index].name);
				if (optarg)
					printf(" with arg %s", optarg);
				printf("\n");
				break;
			case 's':
				Sfilename = optarg;
				break;
			case 'n':
				n_sample = atoi(optarg);
				break;
			case 'k':
				kmax = atoi(optarg);
				break;
			case 'o':
				Ofilename = optarg;
				break;
			case 'd':
				d_init = atof(optarg);
			case 'e':
				d_end = atof(optarg);
				break;
			default: /* '?' */
				fprintf(stderr, "Usage: %s -s S.csv [-n 1000] [-k 10] [-d 200] [-e 400] [-o pattern{_Zflat.csv}] [--verbose]\n", argv[0]);
				return 1;
		}
	}

	if (verbose_flag == 1) {
		verbose = true;
	}

	if (Sfilename == NULL) {
		cout << "INFO: No S matrix supplied! Exiting" << endl;
		return 0;
	}
	cout << "INFO: Using S matrix from file: " << Sfilename << endl;

	if (Ofilename == NULL) {
		Ofilename = "output";
		cout << "INFO: No output file supplied, saving to file " << Ofilename << "_Zflat.csv" << endl;
	}

	if (n_sample == -1) {
		n_sample = 1000;
		cout << "INFO: n was not specified, setting to n = " << n_sample << endl;
	}

	if (d_init == -1) {
		d_init = 200;
		cout << "INFO: d_init was not specified, setting to d_init = " << d_init << endl;
	}

	if (d_end == -1) {
		d_end = 2.0 * d_init;
		cout << "INFO: d_end was not specified, setting to d_end = " << d_end << endl;
	}

	if (kmax == -1) {
		kmax = 10;
		cout << "INFO: k(max) not specified, setting to k(max) = " << kmax << endl;
	}

	mat S;
	S.load(Sfilename);

	int n = S.n_rows;
	double d = d_init;

	double Ashape = 0.25;
	double Ascale = 2.0;

	double gammashape = 2.0;

	double lambda = 0.1 / log(n);

	int burnin = min(0.1 * n_sample, double(1000));
	cout << "burnin = " << burnin << endl;
	double anneal_factor = pow(d_end / d_init, 1.0 / double(n_sample - burnin));
	double final_anneal_factor = 1.0 / 0.975;

	int k = 1;

	uvec Zflat = zeros<uvec>(n);
	vec SiZversion = -1.0 * ones<vec>(n);
	int version = 0;

	vec ZtZ = zeros<vec>(kmax);
	ZtZ(0) = n;
	vec Z0tZ0 = ZtZ;

	vec A = 0.1 * ones<vec>(kmax);
	vec B = zeros<vec>(kmax);
	B(0) = -A(0) / (A(0) * n + 1);
	vec B0 = B;

	vec lvec = zeros<vec>(kmax);

	double trS = trace(S);
	double sumS = accu(S);

	mat ZtSZ = zeros<mat>(kmax, kmax);
	ZtSZ(0, 0) = sumS;
	mat Z0tSZ0 = ZtSZ;

	mat SiZ = zeros<mat>(n, kmax);

	double sWs;
	double detWQ;
	double trWQS;

	int acceptance_A = 0;
	int acceptance_A_counter = 0;

	int k_i;

	mat Zdebug = zeros(n, kmax);
	Zdebug.col(0) = ones<vec>(n);

	int m = 0;

	// begin sampling ----------------------------------------------
	while (m < n_sample) {
		m++;
		if (m >= burnin) {
			if (m <= n_sample) {
				d = d * anneal_factor;
			} else {
				d = d * final_anneal_factor;
			}
		}

		if (verbose) {
			if (m % 100 == 0 || m == 1) {
				printf("sweep %d, d = %.3f, k = %d\n", m, d, k);
			}
		}

		for (int i = 0; i < n; ++i) {

			k_i = Zflat(i);

			// remove object i from cluster k_i
			B0 = B;
			if (ZtZ(k_i) == 1) {
				B0(k_i) = -A(k_i);
			} else {
				B0(k_i) = B0(k_i) / (1.0 + B0(k_i)); // remove 1 object from cluster k_i
			}

			Z0tZ0 = ZtZ;
			if (ZtZ(k_i) == 0) {
				cout << "error: k_i = " << k_i << endl;
				cout << "ZtZ =\n" << ZtZ << endl;
				cout << "Zflat =\n" << Zflat << endl;
				return 0;
			}

			Z0tZ0(k_i) = Z0tZ0(k_i) - 1;

			if (SiZversion(i) != version) {
				SiZ.row(i) = zeros<mat>(1, kmax);
				for (int ii = 0; ii < n; ++ii) {
					if (ii != i) {
						SiZ(i, Zflat(ii)) += S(i, ii);
					}
				}
				SiZversion(i) = version;
			}

			Z0tSZ0 = ZtSZ;
			Z0tSZ0.submat(k_i, 0, k_i, k - 1) -= SiZ.submat(i, 0, i, k - 1);
			Z0tSZ0.submat(0, k_i, k - 1, k_i) = Z0tSZ0.submat(k_i, 0, k_i, k - 1).t();
			Z0tSZ0(k_i, k_i) = ZtSZ(k_i, k_i) - 2.0 * SiZ(i, k_i) - S(i, i);

			if (Z0tZ0(k_i) == 0) {

				uvec indices = zeros<uvec>(kmax);
				int counter = 0;
				for (int ii = 0; ii < k; ++ii) {
					indices(counter) = ii;
					if (ii != k_i) {
						counter++;
					}
				}
				indices(counter) = k_i;
				counter++;
				for (int ii = k; ii < kmax; ++ii) {
					indices(counter) = ii;
					counter++;
				}

				A = A(indices);
				B0 = B0(indices);
				Z0tZ0 = Z0tZ0(indices);

				for (int ii = 0; ii < n; ++ii) {
					if (int(Zflat(ii)) > k_i) {
						Zflat(ii) = Zflat(ii) - 1;
					}
				}
				Zflat(i) = -1;

				Z0tSZ0 = Z0tSZ0.cols(indices);
				Z0tSZ0 = Z0tSZ0.rows(indices);

				mat tmp = SiZ.row(i);
				SiZ.row(i) = tmp.cols(indices);

				k_i = k;

				version++;
				k--;
			}

			// existing cluster
			for (int j = 0; j < k; ++j) {

				// add object i to cluster j
				B = B0;
				B(j) = B(j) / (1.0 - B(j)); // add 1 object to cluster j

				Zflat(i) = j;

				ZtZ = Z0tZ0;
				ZtZ(j) = ZtZ(j) + 1;

				ZtSZ = Z0tSZ0;
				ZtSZ.submat(j, 0, j, k - 1) += SiZ.submat(i, 0, i, k - 1);
				ZtSZ.submat(0, j, k - 1, j) = ZtSZ.submat(j, 0, j, k - 1).t();
				ZtSZ(j, j) = Z0tSZ0(j, j) + 2.0 * SiZ(i, j) + S(i, i);

				sWs = 1.0 / (sum(pow(ZtZ.subvec(0, k - 1), 2) % B.subvec(0, k - 1)) + n);

				detWQ = double(n) * sWs / prod(1.0 + ZtZ.subvec(0, k - 1) % A.subvec(0, k - 1));

				trWQS = trace(diagmat(B.subvec(0, k - 1)) * ZtSZ.submat(0, 0, k - 1, k - 1)) + trS
					- sWs
					* (accu(
								diagmat(ZtZ.subvec(0, k - 1)) * diagmat(B.subvec(0, k - 1)) * ZtSZ.submat(0, 0, k - 1, k - 1)
								* diagmat(B.subvec(0, k - 1)) * diagmat(ZtZ.subvec(0, k - 1)))
							+ 2.0
							* accu(
								ZtSZ.submat(0, 0, k - 1, k - 1) * diagmat(B.subvec(0, k - 1))
								* diagmat(ZtZ.subvec(0, k - 1))) + sumS);

				lvec(j) = d / 2.0 * log(detWQ) - double(n - 1) * d / 2.0 * log(trWQS);
				lvec(j) = lvec(j) + log(ZtZ(j) + lambda / double(kmax)); // m.n. prior

				if (is_finite(lvec(j)) == false) {
					cout << "error: lvec(j = " << j << ") is nan!" << endl;
					cout << "sWs = " << sWs << endl;
					cout << "detWQ = " << detWQ << endl;
					cout << "trWQS = " << trWQS << endl;

					cout << "ZtZ =\n" << ZtZ << endl;
					cout << "Zflat =\n" << Zflat << endl;
					cout << "k = " << k << endl;

					cout << "A =\n" << A << endl;

					cout << "B =\n" << B << endl;
					vec Btrue = zeros<vec>(kmax);
					for (int ii = 0; ii < kmax; ++ii) {
						Btrue(ii) = -A(ii) / (ZtZ(ii) * A(ii) + 1.0);
					}
					cout << "Btrue =\n" << Btrue << endl;

					mat Z = zeros(n, max(Zflat) + 1);
					for (int ii = 0; ii < n; ++ii) {
						Z(ii, Zflat(ii)) = 1;
					}
					cout << "should be ZtSZ =\n" << Z.t() * S * Z << endl;
					cout << "is ZtSZ =\n" << ZtSZ.submat(0, 0, k - 1, k - 1) << endl;

					return 0;
				}
			}

			// new cluster
			if (k < kmax) {
				B = B0;
				B(k) = -A(k) / (A(k) + 1.0);

				ZtZ = Z0tZ0;
				ZtZ(k) = 1;

				Zflat(i) = k;

				ZtSZ = Z0tSZ0;
				ZtSZ.submat(k, 0, k, k) = SiZ.submat(i, 0, i, k);
				ZtSZ.submat(0, k, k, k) = SiZ.submat(i, 0, i, k).t();
				ZtSZ(k, k) = S(i, i);

				sWs = 1.0 / (sum(pow(ZtZ.subvec(0, k), 2) % B.subvec(0, k)) + n);

				detWQ = double(n) * sWs / prod(1.0 + ZtZ.subvec(0, k) % A.subvec(0, k));

				trWQS = trace(diagmat(B.subvec(0, k)) * ZtSZ.submat(0, 0, k, k)) + trS
					- sWs
					* (accu(
								diagmat(ZtZ.subvec(0, k)) * diagmat(B.subvec(0, k)) * ZtSZ.submat(0, 0, k, k)
								* diagmat(B.subvec(0, k)) * diagmat(ZtZ.subvec(0, k)))
							+ 2.0 * accu(ZtSZ.submat(0, 0, k, k) * diagmat(B.subvec(0, k)) * diagmat(ZtZ.subvec(0, k))) + sumS);

				lvec(k) = d / 2.0 * log(detWQ) - double(n - 1) * d / 2.0 * log(trWQS);
				lvec(k) = lvec(k) + log(lambda * (1.0 - double(k) / double(kmax))); // m.n. prior

				if (is_finite(lvec(k)) == false) {
					cout << "error: lvec(k = " << k << ") is nan!" << endl;
					cout << "sWs = " << sWs << endl;
					cout << "detWQ = " << detWQ << endl;
					cout << "trWQS = " << trWQS << endl;
					return 0;
				}
			}

			// decide which cluster is best
			vec bins;
			if (k >= kmax) {
				bins = cumsum(exp(lvec.subvec(0, k - 1) - max(lvec.subvec(0, k - 1))));
			} else {
				bins = cumsum(exp(lvec.subvec(0, k) - max(lvec.subvec(0, k))));
			}

			double unif = unifrnd(rng) * bins(bins.n_elem - 1);

			bins(bins.n_elem - 1) = bins(bins.n_elem - 1) + 1e-6;

			int bStar = -1;
			for (int j = 0; j <= k; ++j) {
				if (bins(j) >= unif) {
					bStar = j;
					break;
				}
			}

			if (bStar < 0) {
				cout << "error!!! exiting" << endl;
				cout << "bStar = " << bStar << endl;
				cout << "bins =\n" << bins << endl;
				cout << "unif = " << unif << endl;
				cout << "lvec =\n" << lvec << endl;
				cout << "ZtZ =\n" << ZtZ << endl;
				cout << "Zflat =\n" << Zflat << endl;
				cout << "k = " << k << endl;
				return 0;
			}

			if (bStar < k) {
				B = B0;
				B(bStar) = B(bStar) / (1.0 - B(bStar)); // add 1 object to cluster j

				Zflat(i) = bStar;
				ZtZ = Z0tZ0;
				ZtZ(bStar) += 1;

				ZtSZ = Z0tSZ0;
				ZtSZ.submat(bStar, 0, bStar, k - 1) += SiZ.submat(i, 0, i, k - 1);
				ZtSZ.submat(0, bStar, k - 1, bStar) = ZtSZ.submat(bStar, 0, bStar, k - 1).t();
				ZtSZ(bStar, bStar) = Z0tSZ0(bStar, bStar) + 2.0 * SiZ(i, bStar) + S(i, i);

				if (bStar != k_i) {
					version++;
				}
			} else {
				Zflat(i) = k;

				if (k + 2 <= kmax) {
					A(k + 1) = gamrnd(Ashape, Ascale, rng);
				}

				k++;
				version++;

			}

		} // for all objects

		// metropolis hastings for A
		for (int j = 0; j < k; ++j) {
			sWs = 1.0 / (sum(pow(ZtZ.subvec(0, k - 1), 2) % B.subvec(0, k - 1)) + n);

			detWQ = n * sWs / prod(1 + ZtZ.subvec(0, k - 1) % A.subvec(0, k - 1));

			trWQS = trace(diagmat(B.subvec(0, k - 1)) * ZtSZ.submat(0, 0, k - 1, k - 1)) + trS
				- sWs
				* (accu(
							diagmat(ZtZ.subvec(0, k - 1)) * diagmat(B.subvec(0, k - 1)) * ZtSZ.submat(0, 0, k - 1, k - 1)
							* diagmat(B.subvec(0, k - 1)) * diagmat(ZtZ.subvec(0, k - 1)))
						+ 2.0
						* accu(
							ZtSZ.submat(0, 0, k - 1, k - 1) * diagmat(B.subvec(0, k - 1)) * diagmat(ZtZ.subvec(0, k - 1)))
						+ sumS);

			double ko = gammashape;
			double thetao = A(j) / ko;

			vec Ap = A;
			Ap(j) = gamrnd(ko, thetao, rng) + 1e-6; // proposal

			vec Bp = B;
			Bp(j) = -Ap(j) / (ZtZ(j) * Ap(j) + 1.0);

			double kp = gammashape;
			double thetap = Ap(j) / kp;

			sWs = 1.0 / (sum(pow(ZtZ.subvec(0, k - 1), 2) % Bp.subvec(0, k - 1)) + n);

			double detWQp = n * sWs / prod(1 + ZtZ.subvec(0, k - 1) % Ap.subvec(0, k - 1));

			double trWQSp = trace(diagmat(Bp.subvec(0, k - 1)) * ZtSZ.submat(0, 0, k - 1, k - 1)) + trS
				- sWs
				* (accu(
							diagmat(ZtZ.subvec(0, k - 1)) * diagmat(Bp.subvec(0, k - 1)) * ZtSZ.submat(0, 0, k - 1, k - 1)
							* diagmat(Bp.subvec(0, k - 1)) * diagmat(ZtZ.subvec(0, k - 1)))
						+ 2.0
						* accu(
							ZtSZ.submat(0, 0, k - 1, k - 1) * diagmat(Bp.subvec(0, k - 1))
							* diagmat(ZtZ.subvec(0, k - 1))) + sumS);

			lvec(j) = d / 2.0 * log(detWQ) - double(n - 1) * d / 2.0 * log(trWQS);

			double a1 = exp(
					(d / 2.0 * log(detWQp) - double(n - 1) * d / 2.0 * log(trWQSp))
					- (d / 2.0 * log(detWQ) - double(n - 1) * d / 2.0 * log(trWQS)));

			double a2 = gampdf(Ap(j), Ashape, Ascale) / gampdf(A(j), Ashape, Ascale);
			double a3 = gampdf(A(j), kp, thetap) / gampdf(Ap(j), ko, thetao);

			double r = min(1.0, a1 * a2 * a3);

			acceptance_A_counter = acceptance_A_counter + 1;
			if (unifrnd(rng) < r) {
				A(j) = Ap(j);
				B(j) = Bp(j);
				acceptance_A = acceptance_A + 1;
			}
		}

		if (k + 1 <= kmax) {
			A(k) = gamrnd(Ashape, Ascale, rng);
			B(k) = -A(k) / (ZtZ(k) * A(k) + 1.0);
		}

	}

	cout << "--------------- end reached -----------------" << endl;
	cout << "k = " << k << endl;

	vec acceptA = zeros<vec>(1);

	acceptA(0) = double(acceptance_A) / double(acceptance_A_counter);

	cout << "acceptance A = " << acceptA(0) << endl;

	char filename[200];

	sprintf(filename, "%s_acceptanceA.csv", Ofilename);
	acceptA.save(filename, csv_ascii);

	sprintf(filename, "%s_Zflat.csv", Ofilename);
	Zflat.save(filename, csv_ascii);

	return 0;
}
