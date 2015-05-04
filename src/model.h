/*
 * Copyright (C) 2007 by
 * 
 * 	Xuan-Hieu Phan
 *	hieuxuan@ecei.tohoku.ac.jp or pxhieu@gmail.com
 * 	Graduate School of Information Sciences
 * 	Tohoku University
 *
 * GibbsLDA++ is a free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 *
 * GibbsLDA++ is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GibbsLDA++; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
 */

/* 
 * References:
 * + The Java code of Gregor Heinrich (gregor@arbylon.net)
 *   http://www.arbylon.net/projects/LdaGibbsSampler.java
 * + "Parameter estimation for text analysis" by Gregor Heinrich
 *   http://www.arbylon.net/publications/text-est.pdf
 */

#ifndef	_MODEL_H
#define	_MODEL_H

#include "constants.h"
#include "dataset.h"

using namespace std;

// LDA model
class model {
public:
    // fixed options
    string wordmapfile;		// file that contains word map [string -> integer id]
    string trainlogfile;	// training log file
    string tassign_suffix;	// suffix for topic assignment file
    string theta_suffix;	// suffix for theta file
    string phi_suffix;		// suffix for phi file
    string others_suffix;	// suffix for file containing other parameters
    string twords_suffix;	// suffix for file containing words-per-topics

    string dir;			// model directory
    string dfile;		// data file    
    string ffile;       // data file    
    string model_name;		// model name
    int model_status;		// model status:
				// MODEL_STATUS_UNKNOWN: unknown status
				// MODEL_STATUS_EST: estimating from scratch
				// MODEL_STATUS_EST_LDA: estimating from scratch, for the FLDA model

    dataset * ptrndata;	// pointer to corpus/tweets training dataset object
    dataset * pfrnddata; // pointer to friend training dataset object
    dataset * pnewdata; // pointer to new dataset object

    mapid2word id2word; // word map [int => string]
    mapid2word id2user; // friend map [int => user string]
    
    // --- model parameters and variables ---    
    int M; // dataset size (i.e., number of users/user documents)
    int V; // vocabulary size
    int K; // number of topics
    double alpha, beta; // LDA hyperparameters 
    double epsilon, gamma, rho0, rho1; // FLDA hyperparameters
    int niters; // number of Gibbs sampling iterations
    int liter; // the iteration at which the model was saved
    int savestep; // saving period
    int twords; // print out top words per each topic

    double * p; // temp variable for sampling
    double * q; // temp variable for sampling
    double * r; // temp variable for sampling
    int ** z; // topic assignments for words, size M x doc.size()
    int ** nw; // cwt[i][j]: number of instances of word/term i assigned to topic j, over all users, size V x K
    int ** nd; // na[i][j]: number of words in document i assigned to topic j, over all words, size M x K
    int * nwsum; // nwsum[j]: total number of words assigned to topic j, size K
    int * ndsum; // nasum[i]: total number of words in document i, size M
    double ** theta; // theta: document-topic distributions, size M x K
    double ** phi; // phi: topic-word distributions, size K x V

    // --- microblog FLDA network variables ---
    // model parameters
    int L; // user size (ie number of followers for a given user(??))
    int O; // total number of friends

    // sampling variables
    // Documents are replaced by users in FLDA!
    int ** x; // topic assignments for users, size L x K
    int ** y; // binary array for whether user is being followed for content or non-content reasons, size L
    int ** nl; // d_z, m, *, *: number of words in user m's document assigned to topic z, for all links, for any reason, size M x K
    int * nlsum; // nlsum[i]: total number of friends following person i, size M
    // int ** A; // c_x, m, *: topic x assigned to user m, size M x K
    // int ** B; // d_x, m, *, *: topic x assigned to user m(????), size M x K
    int * C0; // d_*, m, *, 0: size M
    int * C1; // d_*, m, *, 1: size M
    int * D0; // d_*, *, e, 0: 
    int D0sum; // d_*, *, *, 0:
    int ** D1; // d_x, *, e, 1:
    int * D1sum; // d_x, *, *, 1:
    double ** mu; // mu: , size M x 2
    double ** sigma; // sigma: topic-follower distributions, size K x O
    double * pi; // pi: , size O

    string sigma_suffix;   // suffix for sigma file
    string pi_suffix;      // suffix for pi file
    string mu_suffix;      // suffix for mu file

    int tusers; // print out top users per each topic
    string tusers_suffix;   // suffix for file containing words-per-topics
    string friendmapfile;   // file that contains friend map [string -> integer id]
    string twitteridmapfile;   // file that contains all Twitter IDs and their mapped usernames/handles [integer id -> string]
    mapid2word twitterid2user; // friend map [int => user string]
    
    // --------------------------------------
    
    model() {
    	set_default_values();
    }
          
    ~model();
    
    // set default values for variables
    void set_default_values();   

    // parse command line to get options
    int parse_args(int argc, char ** argv);
    
    // initialize the model
    int init(int argc, char ** argv);
    
    // save LDA model to files
    // model_name.tassign: topic assignments for words in docs
    // model_name.theta: document-topic distributions
    // model_name.phi: topic-word distributions
    // model_name.others: containing other parameters of the model (alpha, beta, M, V, K)
    int save_model(string model_name);
    int save_model_tassign(string filename);
    int save_model_theta(string filename);
    int save_model_phi(string filename);
    int save_model_others(string filename);
    int save_model_twords(string filename);
    
    // init for estimation
    int init_est();
	
    // estimate LDA model using Gibbs sampling
    void estimate();
    void compute_theta();
    void compute_phi();
    int sampling(int m, int n);

    // FLDA
    int init_est_flda();
    void estimate_flda();
    int sampling_flda_eq1(int m, int n);
    int sampling_flda_eqs23(int m, int l);
    void flda_compute_theta();
    void flda_compute_phi();
    void flda_compute_sigma();
    void flda_compute_mu();
    void flda_compute_pi();

    int save_model_sigma(string filename);
    int save_model_pi(string filename);
    int save_model_mu(string filename);

    int save_model_tusers(string filename);
};

#endif

