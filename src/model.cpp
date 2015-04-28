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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "constants.h"
#include "strtokenizer.h"
#include "utils.h"
#include "dataset.h"
#include "model.h"

using namespace std;

model::~model() {
    if (p) {
	delete p;
    }

    if (ptrndata) {
	delete ptrndata;
    }
    
    if (pnewdata) {
	delete pnewdata;
    }

    if (z) {
	for (int m = 0; m < M; m++) {
	    if (z[m]) {
		delete z[m];
	    }
	}
    }
    
    if (nw) {
	for (int w = 0; w < V; w++) {
	    if (nw[w]) {
		delete nw[w];
	    }
	}
    }

    if (nd) {
	for (int m = 0; m < M; m++) {
	    if (nd[m]) {
		delete nd[m];
	    }
	}
    } 
    
    if (nwsum) {
	delete nwsum;
    }   
    
    if (ndsum) {
	delete ndsum;
    }
    
    if (theta) {
	for (int m = 0; m < M; m++) {
	    if (theta[m]) {
		delete theta[m];
	    }
	}
    }
    
    if (phi) {
	for (int k = 0; k < K; k++) {
	    if (phi[k]) {
		delete phi[k];
	    }
	}
    }

}

void model::set_default_values() {
    wordmapfile = "wordmap.txt";
    friendmapfile = "friendmap.txt";
    trainlogfile = "trainlog.txt";
    tassign_suffix = ".tassign";
    theta_suffix = ".theta";
    phi_suffix = ".phi";
    others_suffix = ".others";
    twords_suffix = ".twords";
    
    dir = "./";
    dfile = "trndocs.dat";
    ffile = "frnddocs.dat";
    model_name = "model-final";    
    model_status = MODEL_STATUS_UNKNOWN;
    
    ptrndata = NULL;
    pnewdata = NULL;
    
    M = 0;
    V = 0;
    K = 100;
    alpha = 50.0 / K;
    epsilon = 0.1;
    beta = 0.1;
    gamma = 0.1;
    rho0 = 50.0 / K;
    rho1 = 50.0 / K;
    niters = 2000;
    liter = 0;
    savestep = 200;    
    twords = 0;
    
    p = NULL;
    z = NULL;
    nw = NULL;
    nd = NULL;
    nwsum = NULL;
    ndsum = NULL;
    theta = NULL;
    phi = NULL;

    // FLDA variables
    // model parameters
    L = 0;

    // sampling variables
    x = NULL; 
    y = NULL;
    nl = NULL;
    C0 = NULL;
    C1 = NULL;
    D0 = NULL;
    D0sum = 0;
    D1 = NULL;
    D1sum = NULL;
}

int model::parse_args(int argc, char ** argv) {
    return utils::parse_args(argc, argv, this);
}

int model::init(int argc, char ** argv) {
    // call parse_args
    if (parse_args(argc, argv)) {
    	return 1;
    }

    if (model_status == MODEL_STATUS_EST) {
	// estimating the model from scratch
    	if (init_est()) {
    	    return 1;
    	}
    } else if (model_status == MODEL_STATUS_EST_FLDA) {
        if (init_est_flda()) {
            return 1;
        }
    }
    
    return 0;
}

int model::save_model(string model_name) {
    if (save_model_tassign(dir + model_name + tassign_suffix)) {
        return 1;
    }
    
    if (save_model_others(dir + model_name + others_suffix)) {
        return 1;
    }
    
    if (save_model_theta(dir + model_name + theta_suffix)) {
        return 1;
    }
    
    if (save_model_phi(dir + model_name + phi_suffix)) {
        return 1;
    }
    
    if (twords > 0) {
        if (save_model_twords(dir + model_name + twords_suffix)) {
            return 1;
        }
    }
    
    return 0;
}

int model::save_model_tassign(string filename) {
    int i, j;
    
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
        printf("Cannot open file %s to save!\n", filename.c_str());
        return 1;
    }

    // wirte docs with topic assignments for words
    for (i = 0; i < ptrndata->M; i++) {    
        for (j = 0; j < ptrndata->docs[i]->length; j++) {
            fprintf(fout, "%d:%d ", ptrndata->docs[i]->words[j], z[i][j]);
        }
        fprintf(fout, "\n");
    }

    fclose(fout);
    
    return 0;
}

int model::save_model_theta(string filename) {
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
        printf("Cannot open file %s to save!\n", filename.c_str());
        return 1;
    }
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            fprintf(fout, "%f ", theta[i][j]);
        }
        fprintf(fout, "\n");
    }
    
    fclose(fout);
    
    return 0;
}

int model::save_model_phi(string filename) {
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
        printf("Cannot open file %s to save!\n", filename.c_str());
        return 1;
    }
    
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < V; j++) {
            fprintf(fout, "%f ", phi[i][j]);
        }
        fprintf(fout, "\n");
    }
    
    fclose(fout);    
    
    return 0;
}

int model::save_model_others(string filename) {
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
        printf("Cannot open file %s to save!\n", filename.c_str());
        return 1;
    }

    fprintf(fout, "alpha=%f\n", alpha);
    fprintf(fout, "beta=%f\n", beta);
    fprintf(fout, "ntopics=%d\n", K);
    fprintf(fout, "ndocs=%d\n", M);
    fprintf(fout, "nwords=%d\n", V);
    fprintf(fout, "liter=%d\n", liter);
    
    fclose(fout);    
    
    return 0;
}

int model::save_model_twords(string filename) {
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
        printf("Cannot open file %s to save!\n", filename.c_str());
        return 1;
    }
    
    if (twords > V) {
        twords = V;
    }
    mapid2word::iterator it;
    
    for (int k = 0; k < K; k++) {
        vector<pair<int, double> > words_probs;
        pair<int, double> word_prob;
        for (int w = 0; w < V; w++) {
            word_prob.first = w;
            word_prob.second = phi[k][w];
            words_probs.push_back(word_prob);
        }
        
        // quick sort to sort word-topic probability
        utils::quicksort(words_probs, 0, words_probs.size() - 1);
        
        fprintf(fout, "Topic %dth:\n", k);
        for (int i = 0; i < twords; i++) {
            it = id2word.find(words_probs[i].first);
            if (it != id2word.end()) {
                fprintf(fout, "\t%s   %f\n", (it->second).c_str(), words_probs[i].second);
            }
        }
    }
    
    fclose(fout);    
    
    return 0;    
}

int model::init_est() {
    int m, n, w, k;

    p = new double[K];

    // + read training data
    ptrndata = new dataset;
    if (ptrndata->read_trndata(dir + dfile, dir + wordmapfile)) {
        printf("Fail to read training data!\n");
        return 1;
    }

    // + allocate memory and assign values for variables
    M = ptrndata->M;
    V = ptrndata->V;
    // K: from command line or default value
    // alpha, beta: from command line or default values
    // niters, savestep: from command line or default values

    nw = new int*[V];
    for (w = 0; w < V; w++) {
        nw[w] = new int[K];
        for (k = 0; k < K; k++) {
    	    nw[w][k] = 0;
        }
    }
	
    nd = new int*[M];
    for (m = 0; m < M; m++) {
        nd[m] = new int[K];
        for (k = 0; k < K; k++) {
    	    nd[m][k] = 0;
        }
    }
	
    nwsum = new int[K];
    for (k = 0; k < K; k++) {
	   nwsum[k] = 0;
    }
    
    ndsum = new int[M];
    for (m = 0; m < M; m++) {
	   ndsum[m] = 0;
    }

    srandom(time(0)); // initialize for random number generation
    z = new int*[M];
    for (m = 0; m < ptrndata->M; m++) {
    	int N = ptrndata->docs[m]->length;
    	z[m] = new int[N];
    	
        // initialize for z
        for (n = 0; n < N; n++) {
    	    int topic = (int)(((double)random() / RAND_MAX) * K);
    	    z[m][n] = topic;
    	    
    	    // number of instances of word i assigned to topic j
    	    nw[ptrndata->docs[m]->words[n]][topic] += 1;
    	    // number of words in document i assigned to topic j
    	    nd[m][topic] += 1;
    	    // total number of words assigned to topic j
    	    nwsum[topic] += 1;
        } 
        // total number of words in document i
        ndsum[m] = N;      
    }
    
    theta = new double*[M];
    for (m = 0; m < M; m++) {
        theta[m] = new double[K];
    }
	
    phi = new double*[K];
    for (k = 0; k < K; k++) {
        phi[k] = new double[V];
    }    
    
    return 0;
}

void model::estimate() {
    if (twords > 0) {
    	// print out top words per topic
    	dataset::read_wordmap(dir + wordmapfile, &id2word);
    }

    printf("Sampling %d iterations!\n", niters);

    int last_iter = liter;
    for (liter = last_iter + 1; liter <= niters + last_iter; liter++) {
    	printf("Iteration %d ...\n", liter);
    	
    	// for all z_i
    	for (int m = 0; m < M; m++) {
            // LDA portion of sampling
    	    for (int n = 0; n < ptrndata->docs[m]->length; n++) {
        		// (z_i = z[m][n])
        		// sample from p(z_i|z_-i, w)
        		int topic = sampling(m, n);
        		z[m][n] = topic;
    	    }

            // FLDA portion of network analysis
            // for () {
            // }
    	}
    	
    	if (savestep > 0) {
    	    if (liter % savestep == 0) {
        		// saving the model
        		printf("Saving the model at iteration %d ...\n", liter);
        		compute_theta();
        		compute_phi();
        		save_model(utils::generate_model_name(liter));
    	    }
    	}
    }
    
    printf("Gibbs sampling completed!\n");
    printf("Saving the final model!\n");
    compute_theta();
    compute_phi();
    liter--;
    save_model(utils::generate_model_name(-1));
}

int model::sampling(int m, int n) {
    // remove z_i from the count variables
    int topic = z[m][n];
    int w = ptrndata->docs[m]->words[n];
    nw[w][topic] -= 1;
    nd[m][topic] -= 1;
    nwsum[topic] -= 1;
    ndsum[m] -= 1;

    double Vbeta = V * beta;
    double Kalpha = K * alpha;    
    // do multinomial sampling via cumulative method
    for (int k = 0; k < K; k++) {
        p[k] = (nw[w][k] + beta) / (nwsum[k] + Vbeta) *
                (nd[m][k] + alpha) / (ndsum[m] + Kalpha);
    }

    // Why do you add these all up? It becomes a cumulative up-to-k array
    // cumulate multinomial parameters
    for (int k = 1; k < K; k++) {
       p[k] += p[k - 1];
    }

    // Creates a random number that's smaller than all of the numbers together
    // scaled sample because of unnormalized p[]
    double u = ((double)random() / RAND_MAX) * p[K - 1];
    
    // The topic with the highest probability will have the largest range
    // The random u from above will be most likely to fall under this topic
    for (topic = 0; topic < K; topic++) {
        if (p[topic] > u) {
            break;
        }
    }
    
    // add newly estimated z_i to count variables
    nw[w][topic] += 1;
    nd[m][topic] += 1;
    nwsum[topic] += 1;
    ndsum[m] += 1;    
    
    // Returns topic(index) that broke on the loop above
    return topic;
}

void model::compute_theta() {
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            theta[m][k] = (nd[m][k] + alpha) / (ndsum[m] + K * alpha);
        }
    }
}

void model::compute_phi() {
    for (int k = 0; k < K; k++) {
        for (int w = 0; w < V; w++) {
            phi[k][w] = (nw[w][k] + beta) / (nwsum[k] + V * beta);
        }
    }
}

int model::init_est_flda() {
    // printf("Begin the FLDA initialization\n");
    int m, n, w, l, k, f, o;

    p = new double[K];

    // + read training data
    ptrndata = new dataset;
    if (ptrndata->read_trndata(dir + dfile, dir + wordmapfile)) {
        printf("Fail to read tweet training data!\n");
        return 1;
    }

    // read friend network data
    pfrnddata = new dataset;
    if (pfrnddata->read_trndata(dir + ffile, dir + friendmapfile)) {
        printf("Fail to read friend training data!\n");
        return 1;
    }

    // + allocate memory and assign values for variables
    M = ptrndata->M;
    V = ptrndata->V;
    L = pfrnddata->M;
    O = pfrnddata->V;
    // K: from command line or default value
    // alpha, beta: from command line or default values
    // niters, savestep: from command line or default values

    nw = new int*[V];
    for (w = 0; w < V; w++) {
        nw[w] = new int[K];
        for (k = 0; k < K; k++) {
            nw[w][k] = 0;
        }
    }
    
    nd = new int*[M];
    for (m = 0; m < M; m++) {
        nd[m] = new int[K];
        for (k = 0; k < K; k++) {
            nd[m][k] = 0;
        }
    }
    
    nwsum = new int[K];
    for (k = 0; k < K; k++) {
       nwsum[k] = 0;
    }

    //------- FLDA TESTING ---------
    ndsum = new int[M];
    for (m = 0; m < M; m++) {
       ndsum[m] = 0;
    }

    nlsum = new int[M];
    for (l = 0; l < L; l++) {
       nlsum[m] = 0;
    }
    // MAY BE REMOVED
    
    // FLDA matrices
    // Eq 1
    nl = new int*[M];
    for (m = 0; m < M; m++) {
        nl[m] = new int[K];
        for (k = 0; k < K; k++) {
            nl[m][k] = 0;
        }
    }

    // Eq 2 and 3
    C0 = new int[M];
    for (m = 0; m < M; m++) {
        C0[m] = 0;
    }

    C1 = new int[M];
    for (m = 0; m < M; m++) {
        C1[m] = 0;
    }

    D0 = new int[O];
    for (o = 0; o < O; o++) {
        D0[o] = 0;
    }

    D1sum = 0;

    D1 = new int*[O];
    for (o = 0; o < O; o++) {
        D1[o] = new int[K];
        for (k = 0; k < K; k++) {
            D1[o][k] = 0;
        }
    }

    D1sum = new int[K];
    for (k = 0; k < K; k++) {
        D1sum[k] = 0;
    }

    srandom(time(0)); // initialize for random number generation
    z = new int*[M];

    // FLDA
    x = new int*[M];
    y = new int*[M];
    for (m = 0; m < ptrndata->M; m++) {
        int N = ptrndata->docs[m]->length;
        z[m] = new int[N];

        // FLDA
        int F = pfrnddata->docs[m]->length;
        x[m] = new int[F];
        y[m] = new int[F];
        
        // initialize for z
        for (n = 0; n < N; n++) {
            int topic = (int)(((double)random() / RAND_MAX) * K);
            z[m][n] = topic;
            
            // number of instances of word i assigned to topic j
            nw[ptrndata->docs[m]->words[n]][topic] += 1;
            // number of words in document i assigned to topic j
            nd[m][topic] += 1;
            // total number of words assigned to topic j
            nwsum[topic] += 1;

            // FLDA
            // Eq 1
            nl[m][topic] += 1;
        } 
        // total number of words in document i
        ndsum[m] = N;      

        for (f = 0; f < F; f++) {
            int topic = (int)(((double)random() / RAND_MAX) * K);
            int indicator = 0;
            if (((double)random() / RAND_MAX) > 0.5) {
                indicator = 1;
            } 

            x[m][f] = topic;
            y[m][f] = indicator;

            // FLDA
            // Eq 2 and 3
            nl[m][topic] += 1;
            nd[m][topic] += 1;

            C0[m] += 1;
            C1[m] += 1;

            D0[pfrnddata->docs[m]->words[f]] += 1;
            D0sum += 1;

            D1[pfrnddata->docs[m]->words[f]][topic] += 1;
            D1sum[topic] += 1;
        }
        // total number of friends following person m
        nlsum[m] = F;
    }
    
    theta = new double*[M];
    for (m = 0; m < M; m++) {
        theta[m] = new double[K];
    }
    
    phi = new double*[K];
    for (k = 0; k < K; k++) {
        phi[k] = new double[V];
    }    

    mu = new double*[M];
    for (m = 0; m < M; m++) {
        mu[m] = new double[2];
    }    

    sigma = new double*[K];
    for (k = 0; k < K; k++) {
        sigma[k] = new double[O];
    }    

    pi = new double[O];

    return 0;
}

void model::estimate_flda() {
    if (twords > 0) {
        // print out top words per topic
        dataset::read_wordmap(dir + wordmapfile, &id2word);
    }

    printf("Sampling %d iterations!\n", niters);

    int last_iter = liter;
    for (liter = last_iter + 1; liter <= niters + last_iter; liter++) {
        printf("Iteration %d ...\n", liter);
        
        // for all z_i
        for (int m = 0; m < M; m++) {
            // LDA portion of sampling
            for (int n = 0; n < ptrndata->docs[m]->length; n++) {
                // (z_i = z[m][n])
                // sample from p(z_i|z_-i, w)
                int topic = sampling_flda_eq1(m, n);
                z[m][n] = topic;
            }

            // FLDA portion of network analysis
            for (int l = 0; l < pfrnddata->docs[m]->length; l++) {
                if (y[m][l]) {
                    // y = 1
                    int topic = sampling_flda_eq3(m, l);
                    x[m][l] = topic;
                } else {
                    // y = 0
                    int topic = sampling_flda_eq2(m, l);
                    x[m][l] = topic;
                }
                // ..how do I change the initial values of y[m][l]?
            }
        }
        
        if (savestep > 0) {
            if (liter % savestep == 0) {
                // saving the model
                printf("Saving the model at iteration %d ...\n", liter);
                flda_compute_theta();
                flda_compute_phi();
                save_model(utils::generate_model_name(liter));
            }
        }
    }
    
    printf("Gibbs sampling completed!\n");
    printf("Saving the final model!\n");
    flda_compute_theta();
    // compute_theta();
    flda_compute_phi();
    liter--;
    save_model(utils::generate_model_name(-1));
}

int model::sampling_flda_eq1(int m, int n) {
    // remove z_i from the count variables
    int topic = z[m][n];
    int w = ptrndata->docs[m]->words[n];
    nw[w][topic] -= 1;
    nd[m][topic] -= 1;
    nwsum[topic] -= 1;

    double Vbeta = V * beta;
    // do multinomial sampling via cumulative method

    // Equation 1
    for (int k = 0; k < K; k++) {
        p[k] = ((nd[m][k] + nl[m][k] + alpha) * (nw[w][k] + beta)) /
                (nwsum[topic] + Vbeta);
                // ((ndsum[m] + nlsum[m] + K * alpha) * (nwsum[topic] + Vbeta));
    }

    // Why do you add these all up? It becomes a cumulative up-to-k array
    // cumulate multinomial parameters
    for (int k = 1; k < K; k++) {
       p[k] += p[k - 1];
    }

    // Creates a random number that's smaller than all of the numbers together
    // scaled sample because of unnormalized p[]
    double u = ((double)random() / RAND_MAX) * p[K - 1];
    
    // The topic with the highest probability will have the largest range
    // The random u from above will be most likely to fall under this topic
    for (topic = 0; topic < K; topic++) {
        if (p[topic] > u) {
            break;
        }
    }
    
    // add newly estimated z_i to count variables
    nw[w][topic] += 1;
    nd[m][topic] += 1;
    nwsum[topic] += 1;
    
    // Returns topic(index) that broke on the loop above
    return topic;
}

// pair<int, int> model::sampling_flda_eq2(int m, int l) {
int model::sampling_flda_eq2(int m, int l) {
    // remove x_i and y_i from the count variables
    int topic = x[m][l];
    // int indicator = y[m][l];
    int e = pfrnddata->docs[m]->words[l];

    nl[m][topic] -= 1;
    C0[m] -= 1;
    D0[e] -= 1;
    D0sum -= 1;

    double Mepsilon = M * epsilon;

    // do multinomial sampling via cumulative method
    for (int k = 0; k < K; k++) {
        p[k] = ((nd[m][topic] + nl[m][topic] + alpha) * (C0[m] + rho0) * 
                (D0[e] + epsilon)) / (D0sum + Mepsilon);
    }

    // Why do you add these all up? It becomes a cumulative up-to-k array
    // cumulate multinomial parameters
    for (int k = 1; k < K; k++) {
       p[k] += p[k - 1];
    }

    // Creates a random number that's smaller than all of the numbers together
    // scaled sample because of unnormalized p[]
    double u = ((double)random() / RAND_MAX) * p[K - 1];
    
    // The topic with the highest probability will have the largest range
    // The random u from above will be most likely to fall under this topic
    for (topic = 0; topic < K; topic++) {
        if (p[topic] > u) {
            break;
        }
    }
    
    // add newly estimated z_i to count variables
    nl[m][topic] += 1;
    C0[m] += 1;
    D0[e] += 1;
    D0sum += 1;
    
    // Returns topic(index) that broke on the loop above
    // pair<int, int> final = make_pair(topic, 0);
    return topic;
}

int model::sampling_flda_eq3(int m, int l) {
    // remove x_i and y_i from the count variables
    int topic = x[m][l];
    // int indicator = y[m][l];
    int e = pfrnddata->docs[m]->words[l];

    nl[m][topic] -= 1;
    C1[m] -= 1;
    D1[e][topic] -= 1;
    D1sum[topic] -= 1;

    double Mgamma = M * gamma;

    // do multinomial sampling via cumulative method
    for (int k = 0; k < K; k++) {
        p[k] = ((nd[m][topic] + nl[m][topic] + alpha) * (C1[m] + rho1) * 
                (D1[e][topic] + gamma)) / (D1sum[topic] + Mgamma);
    }

    // Why do you add these all up? It becomes a cumulative up-to-k array
    // cumulate multinomial parameters
    for (int k = 1; k < K; k++) {
       p[k] += p[k - 1];
    }

    // Creates a random number that's smaller than all of the numbers together
    // scaled sample because of unnormalized p[]
    double u = ((double)random() / RAND_MAX) * p[K - 1];
    
    // The topic with the highest probability will have the largest range
    // The random u from above will be most likely to fall under this topic
    for (topic = 0; topic < K; topic++) {
        if (p[topic] > u) {
            break;
        }
    }
    
    // add newly estimated z_i to count variables
    nl[m][topic] += 1;
    C1[m] += 1;
    D1[e][topic] += 1;
    D1sum[topic] += 1;
    
    // Returns topic(index) that broke on the loop above
    // pair<int, int> final = make_pair(topic, 0);
    return topic;
}

void model::flda_compute_theta() {
    ndsum = new int[M];
    nlsum = new int[M];
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            ndsum[m] += nd[m][k];
            nlsum[m] += nl[m][k];
        }
    }

    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            theta[m][k] = (nd[m][k] + nl[m][k] + alpha) / 
                            (ndsum[m] + nlsum[m] + K * alpha);
        }
    }
}

void model::flda_compute_phi() {
    compute_phi();
}

void model::flda_compute_mu() {
    for (int m = 0; m < M; m++) {
        mu[m][0] = (C0[m] + rho0) / 
                    (C0[m] + C1[m] + rho0 + rho1);
        mu[m][1] = (C1[m] + rho1) / 
                    (C0[m] + C1[m] + rho0 + rho1);
    }
}

void model::flda_compute_sigma() {
    for (int k = 0; k < K; k++) {
        for (int o = 0; o < O; o++) {
            sigma[k][o] = (D1[o][k] + gamma) / 
                            (D1sum[k] + M * gamma);
        }
    }
}

void model::flda_compute_pi() {
    for (int o = 0; o < O; o++) {
        pi[o] = (D0[o] + epsilon) /
                    (D0sum + M * epsilon);
    }
}
