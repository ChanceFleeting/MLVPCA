#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <armadillo>
#include <cmath>
namespace py = pybind11;

/*
Adapted from the Cpp - R helper code. Converted for python use. Manually converting types to ensure control.
*/

//////////////////////////////////////////////////////////////////////////////////////////
//                                                                                      //
//                                   fun_GetCov_c                                       //
//                                                                                      //
//////////////////////////////////////////////////////////////////////////////////////////

namespace semi {
	arma::mat cross_semivar(arma::mat Y, int j, int u, int L2) {
	    int n = (int)Y.n_rows;
	    int L1 = n / L2;
	    arma::uvec rowloc_j = arma::regspace<arma::uvec>(j, L2, n - 1); // seq(from=j, to=n, by=L2)
	    arma::uvec rowloc_u = arma::regspace<arma::uvec>(u, L2, n - 1); // seq(from=u, to=n, by=L2)
	    arma::mat Y_diff = Y.rows(rowloc_j) - Y.rows(rowloc_u);
	    arma::mat cross_semivar = Y_diff.t() * Y_diff / (2 * L1);
	    return cross_semivar;
	}
}

py::dict MultilevelS_c(arma::mat Y, int L1, int L2, py::dict option) {

    // Added: Convert the numpy array to an Armadillo matrix
    //py::scoped_interpreter guard{}
    //py::buffer_info buf_info = Y_array.request();
    //arma::mat Y(reinterpret_cast<double*>(buf_info.ptr), buf_info.shape[0], buf_info.shape[1]);

    std::string model = option["model"].cast<std::string>();
    double corr_uncorrpct = option["corr_uncorrpct"].cast<double>();
    py::object corr_rho0_obj = option["corr_rho"];

    int n = (int)Y.n_rows;
    int p = (int)Y.n_cols;

    py::dict S_list;

    if (model == "1Way") {
        arma::mat xcov = Y.t() * Y / n;
        arma::mat G_z = xcov;
        arma::mat G_w = arma::zeros(p, p);
        S_list["G_w"] = G_w;
        S_list["G_z"] = G_z;
        S_list["corr_rho"] = py::none();
        S_list["c"] = py::none();

        return S_list;

    } else if (model == "2WayNested") {

        arma::mat corr_rho;
        arma::mat h_w_sum = arma::zeros(L2, L2);

        // 1.Get correlation matrix and c
        if (corr_rho0_obj.is_none()) {
            // get sum of cross.semivar for each of the L2xL2 cell
            for (int j = 0; j < L2; ++j) {
                for (int u = 0; u < L2; ++u) {
                    arma::mat h_w_ju = semi::cross_semivar(Y, j, u, L2);
                    h_w_ju.diag().zeros();  // remove the diagonal
                    h_w_sum(j, u) = accu(h_w_ju);  // sum of h_w_ju
                }
            }

            //calculate h_w_sum_far
            arma::vec h_w_sum_vec = vectorise(h_w_sum);
            arma::vec upperv = quantile(h_w_sum_vec.elem(find(h_w_sum_vec > 0)), arma::vec {1 - corr_uncorrpct});
            double upper = upperv(0);  //this 2 steps take upper corr.uncorrpct% quantile. Note that the quantile can be a little different from in R because of algorithm difference.
            double h_w_sum_far = mean(h_w_sum.elem(find(h_w_sum > upper)));

            //correlation between different electrode effects
            corr_rho = (h_w_sum_far - h_w_sum) / h_w_sum_far;
            corr_rho.elem(find(h_w_sum > upper)).zeros();  // far pairs are forced to be 0.

        } else {
            corr_rho = py::cast<arma::mat>(corr_rho0_obj);
        }

        double c = (L2 - accu(corr_rho) / L2) / (L2 - 1);

        //2. Get level-specific covariance matrix
        arma::mat B = kron(arma::eye(L1, L1), L2 * arma::eye(L2, L2));
        arma::mat E = kron(arma::eye(L1, L1), arma::ones(L2).t());

        arma::mat Hw = (Y.t() * (B - E.t() * E) * Y) * 2 / (n * L2 - n);
        arma::mat Hz = (Y.t() * (n * arma::eye(n, n) - arma::ones(n) * arma::ones(n).t() - B + E.t() * E) * Y) * 2 / (n * n - n * L2);

        arma::mat G_w = Hw / (2 * c);
        arma::mat G_z = Hz / 2 - Hw / (2 * c);

        S_list["G_w"] = G_w;
        S_list["G_z"] = G_z;
        S_list["corr_rho"] = corr_rho;
        S_list["c"] = c;
        S_list["h_w_sum"] = h_w_sum;

        return S_list;
    }

    return S_list;
}

////////////////////
//     deflate    //
////////////////////

//[[Rcpp::export]]

arma::mat deflate_c(arma::mat S,
              py::object PrevPi = py::none()){

  if (!PrevPi.is_none()){
    arma::mat PrevPi0 = PrevPi.cast<arma::mat>();
    int p = (int) S.n_rows;
    S = (arma::eye(p,p)-PrevPi0)*S*(arma::eye(p,p)-PrevPi0);
  }
  return S;
}

////////////////////////////////////////
//                                    //
//     First step in ADMM: get h      //
//                                    //
////////////////////////////////////////

double GetTheta_c(arma::vec v, int ndim) {

    if ((v(ndim-1) - v(ndim)) >= 1) {
        double theta = v(ndim-1) - 1;
        return theta;

    } else {
        int p = (int) v.n_elem; // p = length(v)
        arma::vec v1 = arma::linspace(1, p+1, p+1);
        v1.subvec(0, p-1) = v;
        v1.subvec(p, p) = v(p-1) - 1.0 * ndim / p;

        double theta = 0;
        double fnew = 0.0;
        int ddnew = 0;
        int dnew = (int) max(arma::vec{(ndim - 2) * 1.0, 0.0});
        double f = fnew;
        int dd = ddnew;
        int d = dnew;

        while (fnew < ndim) {
            f = fnew;
            dd = ddnew;
            d = dnew;
            dnew += 1; // dnew = dnew + 1
            theta = v1(dnew - 1); // Don't re-define double theta, to avoid changing the outside variable
            arma::uvec loc1 = arma::find((v1 - theta) < 1);
            ddnew = (int) loc1(0) + 1;
            fnew = (ddnew - 1) * 1.0 + arma::sum(v1.subvec(ddnew - 1, dnew - 1)) - (dnew - ddnew + 1) * theta;
        }

        if (fnew == ndim) {
            return theta;
        } else {
            theta = v1(d - 1);
            double m0 = min(arma::vec{1 - (v1(dd - 1) - theta), theta - v1(d)});
            while ((f + (d - dd + 1) * m0) < ndim) {
                f = f + (d - dd + 1) * m0;
                dd += 1;
                theta -= m0;
                m0 = min(arma::vec{1 - (v1(dd - 1) - theta), theta - v1(d)});
            }
            theta = theta - (ndim - f) / (d - dd + 1);
            return theta;
        }
    }
}

arma::mat FantopeProj_c(arma::mat mat1, int ndim, int d, py::object mat0 = py::none()) {
    // If we have a previous projection matrix pi, find its orthogonal complement U and multiply U to mat1
    arma::mat U;
    if (!mat0.is_none()) {
        arma::mat mat00 = py::cast<arma::mat>(mat0);
        int p = (int) mat00.n_rows;
        arma::vec D;
        eig_sym(D, U, arma::eye(p, p) - mat00);
        U = reverse(U, 1);
        U = U.cols(0, p - d - 1);
        mat1 = U.t() * mat1 * U;
    }

    // Decompose mat1 via spectral decomposition, form eigenvalue, and rebuild mat1
    mat1 = (mat1 + mat1.t()) / 2; // Real Symmetric matric
    arma::mat V; arma::vec D;
    eig_sym(D, V, mat1);
    V = reverse(V, 1); D = reverse(D);

    // Get the theta value
    double theta = GetTheta_c(D, ndim);
    arma::vec newvalues = arma::min(arma::max(D - theta, arma::zeros(D.n_elem)), arma::ones(D.n_elem));
    arma::mat newmat = V * diagmat(newvalues) * V.t();

    // If mat0 was provided, adjust the projection matrix with U
    if (!mat0.is_none()) {
        newmat = U * newmat * U.t();
    }

    return newmat;
}


////////////////////////////////////////
//                                    //
//    Second step in ADMM: get y      //
//                                    //
////////////////////////////////////////


arma::mat SoftThreshold_c(arma::mat x, double lambda){
  int n = (int)x.n_rows;
  arma::mat newvalue = sign(x)%max(abs(x)-lambda,arma::zeros(n,n));
  return newvalue;
}

////////////////////////////////////////
//                                    //
// Optimization using ADMM:           //
// Iteratively solve for projection   //
// matrix H.                          //
//                                    //
////////////////////////////////////////

arma::mat seqADMM_c(arma::mat& S, int ndim, int PrevPi_d, double alpha, double lambda, py::dict option,
                    py::object PrevPi = py::none(), bool verbose = false){
    int p = (int) S.n_cols;
	int m = option.contains("m") ? option["m"].cast<int>() : 0;  // ADDED INTERNAL DEFAULTS DUE TO TYPE ERROR
	int maxiter = option.contains("maxiter") ? option["maxiter"].cast<int>() : 100; 
	double eps = option.contains("eps") ? option["eps"].cast<double>() : 1e-6;

    if ((alpha == 0) && (lambda == 0)) {
        S = deflate_c(S, PrevPi);
        S = (S + S.t()) / 2;
        arma::vec D; arma::mat V;
        eig_sym(D, V, S);
        V = reverse(V, 1);
        V = V.cols(0, ndim - 1);
        arma::mat projH = V * V.t();
        return projH;

    } else {
        //starting value
        eps = ndim * eps;
        double tau = 0.1 * arma::abs(S).max();
        int tauStep = 2;

        arma::mat y0 = arma::zeros(p, p);
        arma::mat w0 = arma::zeros(p, p);

        int niter = 0;
        double maxnorm = eps + 1;

        while (niter < maxiter && maxnorm > eps) {
            //update h
            arma::mat h = FantopeProj_c(y0 - w0 + S / tau, ndim, PrevPi_d, PrevPi);

            //update y
            arma::mat y1 = arma::zeros(p, p);
            for (int rowi = 0; rowi < m; ++rowi) {
                for (int colj = 0; colj < m; ++colj) {
                    arma::uvec locRow = arma::regspace<arma::uvec>(rowi * (p / m), (rowi + 1) * (p / m) - 1);
                    arma::uvec locCol = arma::regspace<arma::uvec>(colj * (p / m), (colj + 1) * (p / m) - 1);
                    arma::mat y_ij = SoftThreshold_c(h(locRow, locCol) + w0(locRow, locCol), lambda / tau);
                    double normy_ij = norm(y_ij, "fro");
                    if (normy_ij > alpha * (p / m) / tau) {
                        y1(locRow, locCol) = (normy_ij - alpha * (p / m) / tau) * y_ij / normy_ij;
                    }
                }
            }

            //update w
            arma::mat w1 = w0 + h - y1;

            //stop criterion
            double normr1 = std::pow(norm(h - y1, "fro"), 2);
            double norms1 = std::pow(norm(tau * (y0 - y1), "fro"), 2);
            maxnorm = std::max(normr1, norms1);
            niter += 1;

            //update
            y0 = y1;
            if (normr1 > 100 * norms1) {
                tau *= tauStep;
                w0 = w1 / tauStep;
            } else if (norms1 > 100 * normr1) {
                tau /= tauStep;
                w0 = w1 * tauStep;
            } else {
                w0 = w1;
            }
        }


	    if (verbose) {
	    	py::module warnings = py::module::import("warnings");
	        if (niter < maxiter) {
	            warnings.attr("warn")("Warning: seqADMM has converged after ", niter, " iterations.");
	        } else {
	            warnings.attr("warn")("Warning: seqADMM could not converge.");
	        }
	    }

        arma::mat projH = y0;
        return projH;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
//                                                                                      //
//                             fun_SelectTuningPar_c                                    //
//                                                                                      //
//////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////
//                                          //
//   Select gamma through cross-validation  //
//                                          //
//////////////////////////////////////////////

py::dict CV_Gamma_c(const arma::mat& X, py::dict option) {

    py::list gammaSeq_list = option["gammaSeq_list"];
    int ngamma = py::cast<int>(option["nsol"]);
    int L1 = py::cast<int>(option["L1"]);
    int L2 = py::cast<int>(option["L2"]);
    int nfold = py::cast<int>(option["nfold"]);
    arma::mat SmoothD = py::cast<arma::mat>(option["SmoothD"]); // This casting may be a stretch. Need to check if it works... Documentation is unclear.

    //starting value
    arma::vec V_w = arma::zeros(ngamma);
    arma::vec V_z = arma::zeros(ngamma);
    int i1_w = 0;
    int i1_z = 0;
    arma::vec gamma_w_vec = py::cast<arma::vec>(gammaSeq_list[0]);  // gammaSeq_w
    arma::vec gamma_z_vec = py::cast<arma::vec>(gammaSeq_list[1]);  // gammaSeq_z
    int nsplit = (int) floor(L1 / nfold);

    //for each of the 10 candidate gamma perform CV and get CV <S,H>;
    for (int i = 0; i < ngamma; ++i) {
        double gamma_w = gamma_w_vec(i);
        double gamma_z = gamma_z_vec(i);
	    // For each rho, do CV nfold times, and aggregate test tr(t(H)S) to get V. Then compare V for every rho
	    // Split data by L1 (subject), if has more levels, split the data by the upper level to preserve data structure
        for (int ifold = 0; ifold < nfold; ++ifold) {
            arma::uvec idtest = arma::regspace<arma::uvec>(ifold * nsplit * L2, (ifold + 1) * nsplit * L2 - 1);
            arma::mat Xtest = X.rows(idtest);
            arma::mat Xtrain = X;
            Xtrain.shed_rows(idtest);
            py::dict G_train = MultilevelS_c(Xtrain, L1 - nsplit, L2, option);
            py::dict G_test = MultilevelS_c(Xtest, nsplit, L2, option);

            //for within subject
            arma::mat G_train_w = py::cast<arma::mat>(G_train["G_w"]);
            arma::mat K_train_w = G_train_w - gamma_w * SmoothD;
            arma::mat projH_w = seqADMM_c(K_train_w, 1, 0, 0, 0, option);
            arma::mat K_test_w = py::cast<arma::mat>(G_test["G_w"]);
            V_w(i) += accu(projH_w % K_test_w);

            //for between subject
            arma::mat G_train_z = py::cast<arma::mat>(G_train["G_z"]);
            arma::mat K_train_z = G_train_z - gamma_z * SmoothD;
            arma::mat projH_z = seqADMM_c(K_train_z, 1, 0, 0, 0, option);
            arma::mat K_test_z = py::cast<arma::mat>(G_test["G_z"]);
            V_z(i) += accu(projH_z % K_test_z);
        }
    }

    //select best gamma based on largest CV <S, H>
    i1_w = (int) V_w.index_max();
    i1_z = (int) V_z.index_max();

    double gamma_w_best = (i1_w < (ngamma - 1)) ? gamma_w_vec(i1_w + 1) : gamma_w_vec(i1_w);
    double gamma_z_best = (i1_z < (ngamma - 1)) ? gamma_z_vec(i1_z + 1) : gamma_z_vec(i1_z);

    //return the results as a Python dictionary
    py::dict gamma_list;
    gamma_list["gamma_w"] = gamma_w_best;
    gamma_list["gamma_z"] = gamma_z_best;
    return gamma_list;
}

//////////////////////////////////////////////////////
//                                                  //
// Select alpha and lambda through cross-validation //
//                                                  //
//////////////////////////////////////////////////////

namespace cv {
	arma::vec each(arma::mat x_c, double alpha_w, double lambda_w, double alpha_z, double lambda_z,
	               int cont_w, int cont_z, int Fantope_d, int PrevPi_d, py::dict option,
	               py::object PrevPi_w = py::none(), py::object PrevPi_z = py::none()) {
	    int L1 = py::cast<int>(option["L1"]);
	    int L2 = py::cast<int>(option["L2"]);
	    int nfold = py::cast<int>(option["nfold"]);
	    int nsplit = (int)std::floor(L1 / nfold);
	    py::list gamma_list = option["gamma_list"];
	    double gamma_w = py::cast<double>(gamma_list[0]);
	    double gamma_z = py::cast<double>(gamma_list[1]);
	    arma::mat SmoothD = py::cast<arma::mat>(option["SmoothD"]);

	    double V_w = 0;
	    double V_z = 0;
	    double Vsplit1_w = 0;
	    double Vsplit1_z = 0;

	    for (int ifold = 0; ifold < nfold; ++ifold) {
	        arma::uvec idtest = arma::regspace<arma::uvec>(ifold * nsplit * L2, (ifold + 1) * nsplit * L2 - 1);
	        arma::mat Xtest = x_c.rows(idtest);
	        arma::mat Xtrain = x_c; Xtrain.shed_rows(idtest);
	        py::dict G_train = MultilevelS_c(Xtrain, L1 - nsplit, L2, option);
	        py::dict G_test = MultilevelS_c(Xtest, nsplit, L2, option);

	        // within subject
	        if (cont_w > 0) {
	            arma::mat G_train_w = py::cast<arma::mat>(G_train["G_w"]);
	            arma::mat K_train_w = G_train_w - gamma_w * SmoothD;
	            arma::mat projH_w = seqADMM_c(K_train_w, Fantope_d, PrevPi_d, alpha_w, lambda_w, option, PrevPi_w);
	            arma::mat K_test_w = py::cast<arma::mat>(G_test["G_w"]);
	            V_w += accu(projH_w % K_test_w);
	        }

	        // between subject
	        if (cont_z > 0) {
	            arma::mat G_train_z = py::cast<arma::mat>(G_train["G_z"]);
	            arma::mat K_train_z = G_train_z - gamma_z * SmoothD;
	            arma::mat projH_z = seqADMM_c(K_train_z, Fantope_d, PrevPi_d, alpha_z, lambda_z, option, PrevPi_z);
	            arma::mat K_test_z = py::cast<arma::mat>(G_test["G_z"]);
	            V_z += accu(projH_z % K_test_z);
	        }
	    }

	    Vsplit1_w = V_w / nfold;
	    Vsplit1_z = V_z / nfold;
	    arma::vec Vsplit = {Vsplit1_z, Vsplit1_w};
	    return Vsplit;
	}
}

py::dict CV_AlphaLambda_c(arma::mat x_c, arma::mat alphaSeq_w, arma::mat lambdaSeq_w, arma::mat alphaSeq_z, 
						arma::mat lambdaSeq_z, int Fantope_d, int PrevPi_d, py::dict option, py::object 
						PrevPi_w = py::none(), py::object PrevPi_z = py::none()) {

    int nalpha = (int) alphaSeq_z.n_elem;
    int nlambda = (int) lambdaSeq_z.n_elem;
    double alpha1_w, lambda1_w, alpha1_z, lambda1_z;

    py::object alpha_list = option["alpha_list"];
    py::object lambda_list = option["lambda_list"];

	//////////////////////////////////////////////////
	// If both alpha and lambda need to be selected //
	//////////////////////////////////////////////////

    if (alpha_list.is_none() && lambda_list.is_none()) {
    	// start value
        double alpha0_w = alphaSeq_w(0);
        double lambda0_w = 0;
        double alpha0_z = alphaSeq_z(0);
        double lambda0_z = 0;

        int cont_w = 1;
        int cont_z = 1;
        int maxiter_cv = py::cast<int>(option["maxiter_cv"]);
        int niter = 1;

        while (((cont_w > 0) | (cont_z > 0)) & (niter < maxiter_cv)) {
            if (niter % 2 == 1) {
			    /////////////////////////
			    // first select lambda //
			    /////////////////////////

			    // cross-validation
                arma::mat Vsplit1 = arma::zeros(2, nlambda);
                for (int i = 0; i < nlambda; ++i) {
			        Vsplit1.col(i) = cv::each(x_c, alpha0_w, lambdaSeq_w(i), alpha0_z, lambdaSeq_z(i), cont_w, cont_z,
			        Fantope_d, PrevPi_d, option, PrevPi_w, PrevPi_z);
                }

                // choose the best lambda.z
                int I_z = (int) Vsplit1.row(0).index_max();
                if (cont_z > 0) {
                    lambda1_z = lambdaSeq_z[I_z];
                } else {
                    lambda1_z = lambda0_z;
                }

                // choose the best lambda.w
                int I_w = (int) Vsplit1.row(1).index_max();
                if (cont_w > 0) {
                    lambda1_w = lambdaSeq_w[I_w];
                } else {
                    lambda1_w = lambda0_w;
                }

                // stop criteria
                if ((lambda1_w == lambda0_w) && (niter > 1) && (cont_w > 0)) {
                    cont_w = 0;
                }
                if ((lambda1_z == lambda0_z) && (niter > 1) && (cont_z > 0)) {
                    cont_z = 0;
                }

                // update lambda0
                lambda0_w = lambda1_w;
                lambda0_z = lambda1_z;
                niter += 1;

            } else {

		        /////////////////////////
		        //  then select alpha  //
		        /////////////////////////

        		// cross-validation
                arma::mat Vsplit2 = arma::zeros(2, nalpha);
                for (int j = 0; j < nalpha; ++j) {
          			  Vsplit2.col(j) = cv::each(x_c, alphaSeq_w(j), lambda0_w, alphaSeq_z(j), lambda0_z, cont_w, cont_z,
                      Fantope_d, PrevPi_d, option, PrevPi_w, PrevPi_z);
                }

                // choose the best alpha.z
                int J_z = (int) Vsplit2.row(0).index_max();
                if (cont_z > 0) {
                    alpha1_z = alphaSeq_z[J_z];
                } else {
                    alpha1_z = alpha0_z;
                }

                // choose the best alpha.w
                int J_w = (int) Vsplit2.row(1).index_max();
                if (cont_w > 0) {
                    alpha1_w = alphaSeq_w[J_w];
                } else {
                    alpha1_w = alpha0_w;
                }

                // stop criteria
                if ((alpha1_w == alpha0_w) && (niter > 1) && (cont_w > 0)) {
                    cont_w = 0;
                }
                if ((alpha1_z == alpha0_z) && (niter > 1) && (cont_z > 0)) {
                    cont_z = 0;
                }

                // update alpha0
                alpha0_w = alpha1_w;
                alpha0_z = alpha1_z;
                niter += 1;
            }
        }

    } else if (!alpha_list.is_none() && lambda_list.is_none()) {
	    ////////////////////////////
	    // If only select lambda  //
	    ////////////////////////////

    	py::object k = option["k"];
        int ki = PrevPi_d + 1;
        py::dict alpha0_list = py::cast<py::dict>(alpha_list);
        double alpha0_w = py::cast<arma::vec>(alpha0_list["alpha_w"])(ki - 1);
        double alpha0_z = py::cast<arma::vec>(alpha0_list["alpha_z"])(ki - 1);
	    if (k.is_none()){
	      alpha0_w = py::cast<double>(alpha0_list["alpha_w"]);
	      alpha0_z = py::cast<double>(alpha0_list["alpha_z"]);
	    } else{
		  arma::vec alpha0_w_vec = py::cast<arma::vec>(alpha0_list["alpha_w"]);
	      alpha0_w = alpha0_w_vec(ki-1);
	      arma::vec alpha0_z_vec = py::cast<arma::vec>(alpha0_list["alpha_z"]);
	      alpha0_z = alpha0_z_vec(ki-1);
	    }
        int cont_w = 1;
        int cont_z = 1;

        // cross-validation
        arma::mat Vsplit1 = arma::zeros(2, nlambda);
        for (int i = 0; i < nlambda; ++i) {
	      Vsplit1.col(i) = cv::each(x_c, alpha0_w, lambdaSeq_w(i), alpha0_z, lambdaSeq_z(i), cont_w, cont_z,
	                  Fantope_d, PrevPi_d, option, PrevPi_w, PrevPi_z);
        }

        int I_z = (int) Vsplit1.row(0).index_max();
        lambda1_z = lambdaSeq_z[I_z];

        int I_w = (int) Vsplit1.row(1).index_max();
        lambda1_w = lambdaSeq_w[I_w];

        alpha1_z = alpha0_z;
        alpha1_w = alpha0_w;

    } else if (alpha_list.is_none() && !lambda_list.is_none()) {

	    ////////////////////////////
	    // If only select alpha   //
	    ////////////////////////////

    	py::object k = option["k"];
        int ki = PrevPi_d + 1;
        py::dict lambda0_list = py::cast<py::dict>(lambda_list);
        double lambda0_w = py::cast<arma::vec>(lambda0_list["lambda_w"])(ki - 1);
        double lambda0_z = py::cast<arma::vec>(lambda0_list["lambda_z"])(ki - 1);
	    if (k.is_none()){
	      lambda0_w = py::cast<double>(lambda0_list["lambda_w"]);
	      lambda0_z = py::cast<double>(lambda0_list["lambda_z"]);
	    } else{
		  arma::vec lambda0_w_vec = py::cast<arma::vec>(lambda0_list["lambda_w"]);
	      lambda0_w = lambda0_w_vec(ki-1);
	      arma::vec lambda0_z_vec = py::cast<arma::vec>(lambda0_list["lambda_z"]);
	      lambda0_z = lambda0_z_vec(ki-1);
	    }
        int cont_w = 1;
        int cont_z = 1;

    	// cross-validation
        arma::mat Vsplit2 = arma::zeros(2, nalpha);
        for (int j = 0; j < nalpha; ++j) {
            Vsplit2.col(j) = cv::each(x_c, alphaSeq_w(j), lambda0_w, alphaSeq_z(j), lambda0_z, cont_w, cont_z,
                                       Fantope_d, PrevPi_d, option, PrevPi_w, PrevPi_z);
        }

        int J_z = (int) Vsplit2.row(0).index_max();
        alpha1_z = alphaSeq_z[J_z];

        int J_w = (int) Vsplit2.row(1).index_max();
        alpha1_w = alphaSeq_w[J_w];

        lambda1_z = lambda0_z;
        lambda1_w = lambda0_w;
    }

    py::dict AlphaLambda_list;
    AlphaLambda_list["alpha1_w"] = alpha1_w;
    AlphaLambda_list["alpha1_z"] = alpha1_z;
    AlphaLambda_list["lambda1_w"] = lambda1_w;
    AlphaLambda_list["lambda1_z"] = lambda1_z;

    return AlphaLambda_list;
}

//////////////////////////////////////////////////////
//                                                  //
//    Select alpha and lambda through FVE method    //
//                                                  //
//////////////////////////////////////////////////////

py::dict FVE_AlphaLambda_c(arma::mat K, arma::mat G, arma::vec alphaSeq, arma::vec lambdaSeq, 
                            double totV, int Fantope_d, int PrevPi_d, py::dict option, std::string select, 
                            py::object PrevPi = py::none()) {

    py::object alpha_list = option["alpha_list"];
    py::object lambda_list = option["lambda_list"];
    py::object k = option["k"];

    double rFVEproportion = option["rFVEproportion"].cast<double>();
    int nalpha = (int) alphaSeq.n_elem;
    int nlambda = (int) lambdaSeq.n_elem;
    double alpha1, lambda1;

    if (alpha_list.is_none() && lambda_list.is_none()) {
	
	    /////////////////////////////////////
	    // If select both alpha and lambda //
	    /////////////////////////////////////

        arma::mat FVE = arma::zeros(nalpha, nlambda);
        for (int i = 0; i < nalpha; ++i) {
            for (int j = 0; j < nlambda; ++j) {
                double alpha = alphaSeq(i);
                double lambda = lambdaSeq(j);
                arma::mat projH = seqADMM_c(K, Fantope_d, PrevPi_d, alpha, lambda, option, PrevPi);
                arma::vec D; arma::mat eigV;
                eig_sym(D, eigV, (projH + projH.t()) / 2);
                eigV = reverse(eigV, 1);
                arma::vec eigV1 = eigV.col(0);
                FVE(i, j) = accu(eigV1.t() * G * eigV1) / totV; //use raw cov instead of smoothed S as in Chen2015
            }
        }

        arma::mat prop = FVE / FVE(0, 0);
        arma::uvec eligible_vec = find(prop >= rFVEproportion);
        //If multiple largest combo of rank lambda+alpha (e.g. 2+4, 4+2), choose the one with largest alpha
   		arma::mat eligible_mat = arma::zeros(eligible_vec.n_elem, 2);
        for (int i = 0; i < eligible_vec.n_elem; ++i) {
            int loc = (int) eligible_vec(i);
            eligible_mat(i, 0) = loc % nalpha;
            eligible_mat(i, 1) = ceil(loc / nalpha);
        }

        arma::mat eligible1 = eligible_mat.rows(find(sum(eligible_mat, 1) == sum(eligible_mat, 1).max()));
        arma::rowvec I;
        if (eligible1.n_rows == 1) {
            I = eligible1;
        } else {
            I = eligible1.row(eligible1.col(0).index_max());
        }

        alpha1 = (double)alphaSeq((int)I(0));
        lambda1 = (double)lambdaSeq((int)I(1));

    } else if (!alpha_list.is_none() && lambda_list.is_none()) {

	    /////////////////////////////////////
	    //       If only select lambda     //
	    /////////////////////////////////////

        py::dict alpha0_list = py::cast<py::dict>(alpha_list);

        int ki = PrevPi_d + 1;
        double alpha;
        if (select == "w") {
            if (!k.is_none()) {
                arma::vec alpha_w = py::cast<arma::vec>(alpha0_list["alpha_w"]);
                alpha = alpha_w(ki - 1);
            } else {
                alpha = py::cast<double>(alpha0_list["alpha_w"]);
            }
        } else if (select == "z") {
            if (!k.is_none()) {
                arma::vec alpha_z = py::cast<arma::vec>(alpha0_list["alpha_z"]);
                alpha = alpha_z(ki - 1);
            } else {
                alpha = py::cast<double>(alpha0_list["alpha_z"]);
            }
        }

        arma::vec FVE = arma::zeros(nlambda);
        for (int i = 0; i < nlambda; ++i) {
            double lambda = lambdaSeq(i);
            arma::mat projH = seqADMM_c(K, Fantope_d, PrevPi_d, alpha, lambda, option, PrevPi);
            arma::vec D; arma::mat eigV;
            eig_sym(D, eigV, (projH + projH.t()) / 2);
            eigV = reverse(eigV, 1);
            arma::vec eigV1 = eigV.col(0);
            FVE(i) = accu(eigV1.t() * G * eigV1) / totV;
        }

        arma::vec prop = FVE / FVE(0);
        int I = (int) max(find(prop >= rFVEproportion));
        lambda1 = lambdaSeq(I);
        alpha1 = alpha;

    } else if (alpha_list.is_none() && !lambda_list.is_none()) {
	
	    /////////////////////////////////////
	    //       If only select alpha      //
	    /////////////////////////////////////

        py::dict lambda0_list = py::cast<py::dict>(lambda_list);
        int ki = PrevPi_d + 1;
        double lambda;
        if (select == "w") {
            if (!k.is_none()) {
                arma::vec lambda_w = py::cast<arma::vec>(lambda0_list["lambda_w"]);
                lambda = lambda_w(ki - 1);
            } else {
                lambda = py::cast<double>(lambda0_list["lambda_w"]);
            }
        } else if (select == "z") {
            if (!k.is_none()) {
                arma::vec lambda_z = py::cast<arma::vec>(lambda0_list["lambda_z"]);
                lambda = lambda_z(ki - 1);
            } else {
                lambda = py::cast<double>(lambda0_list["lambda_z"]);
            }
        }

        arma::vec FVE = arma::zeros(nalpha);
        for (int i = 0; i < nalpha; ++i) {
            double alpha = alphaSeq(i);
            arma::mat projH = seqADMM_c(K, Fantope_d, PrevPi_d, alpha, lambda, option, PrevPi);
            arma::vec D; arma::mat eigV;
            eig_sym(D, eigV, (projH + projH.t()) / 2);
            eigV = reverse(eigV, 1);
            arma::vec eigV1 = eigV.col(0);
            FVE(i) = accu(eigV1.t() * G * eigV1) / totV;
        }

        arma::vec prop = FVE / FVE(0);
        int I = (int) max(find(prop >= rFVEproportion));
        alpha1 = alphaSeq(I);
        lambda1 = lambda;
    }

    py::dict AlphaLambda_dict;
    AlphaLambda_dict["alpha1"] = alpha1;
    AlphaLambda_dict["lambda1"] = lambda1;

    return AlphaLambda_dict;
}

//////////////////////////////////////////////
//                                          //
//  	 ADDED: PYBIND11 Wrapper Syntax 	//
//                                          //
//////////////////////////////////////////////

PYBIND11_MODULE(fun_GetCov_SeqADMM_SelectTuningPar, m) {
    m.def("MultilevelS_c", &MultilevelS_c, "Calculate multilevel structure with given parameters", py::arg("Y"), py::arg("L1"), py::arg("L2"), py::arg("option"));
    m.def("deflate_c", &deflate_c, "Deflate matrix by a provided Pi matrix", py::arg("S"), py::arg("PrevPi") = py::none());
    m.def("GetTheta_c", &GetTheta_c, "Get theta for ADMM step", py::arg("v"), py::arg("ndim"));
    m.def("FantopeProj_c", &FantopeProj_c, "Perform Fantope projection", py::arg("mat1"), py::arg("ndim"), py::arg("d"), py::arg("mat0") = py::none());
    m.def("SoftThreshold_c", &SoftThreshold_c, "Second step in ADMM: get y", py::arg("x"), py::arg("lambda"));
    m.def("seqADMM_c", &seqADMM_c, "ADMM optimization to solve for projection matrix H", py::arg("S"), py::arg("ndim"), py::arg("PrevPi_d"), py::arg("alpha"), py::arg("lambda"), py::arg("option"), py::arg("PrevPi") = py::none(), py::arg("verbose") = false);
    m.def("CV_Gamma_c", &CV_Gamma_c, "Select gamma using cross-validation", py::arg("X"), py::arg("option"));
    m.def("CV_AlphaLambda_c", &CV_AlphaLambda_c, "Select alpha and lambda through cross-validation", py::arg("x_c"), py::arg("alphaSeq_w"), py::arg("lambdaSeq_w"), py::arg("alphaSeq_z"), py::arg("lambdaSeq_z"), py::arg("Fantope_d"), py::arg("PrevPi_d"), py::arg("option"), py::arg("PrevPi_w") = py::none(), py::arg("PrevPi_z") = py::none());
    m.def("FVE_AlphaLambda_c", &FVE_AlphaLambda_c, "Select alpha and lambda through FVE method", py::arg("K"), py::arg("G"), py::arg("alphaSeq"), py::arg("lambdaSeq"), py::arg("totV"), py::arg("Fantope_d"), py::arg("PrevPi_d"), py::arg("option"), py::arg("select"), py::arg("PrevPi") = py::none());
}