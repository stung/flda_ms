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

#include <stdio.h>
#include <string>
#include <map>
#include "strtokenizer.h"
#include "utils.h"
#include "model.h"

using namespace std;

int utils::parse_args(int argc, char ** argv, model * pmodel) {
    int model_status = MODEL_STATUS_UNKNOWN;
    string dir = "";
    string model_name = "";
    string dfile = "";
    string ffile = "";
    double alpha = -1.0;
    double beta = -1.0;
    int K = 0;
    int niters = 0;
    int savestep = 0;
    int twords = 0;
    int tusers = 0;
    int withrawdata = 0;

    int i = 0; 
    while (i < argc) {
		string arg = argv[i];
		
		if (arg == "-est") {
		    model_status = MODEL_STATUS_EST;
		    
		} else if (arg == "-est_flda") {
		    model_status = MODEL_STATUS_EST_FLDA;
		    
		} else if (arg == "-dir") {
		    dir = argv[++i];	    
		    
		} else if (arg == "-dfile") {
		    dfile = argv[++i];	    
		    
		}  else if (arg == "-ffile") {
			ffile = argv[++i];

		} else if (arg == "-model") {
		    model_name = argv[++i];	    	    
		    
		} else if (arg == "-alpha") {
		    alpha = stof(argv[++i]);	    
		    
		} else if (arg == "-beta") {
		    beta = stof(argv[++i]);	    
		    
		} else if (arg == "-ntopics") {
		    K = stoi(argv[++i]);	    
		    
		} else if (arg == "-niters") {
		    niters = stoi(argv[++i]);	    
		    
		} else if (arg == "-savestep") {
		    savestep = stoi(argv[++i]);
		    
		} else if (arg == "-twords") {
		    twords = stoi(argv[++i]);
		    
		} else if (arg == "-tusers") {
		    tusers = stoi(argv[++i]);

		} else if (arg == "-withrawdata") {
		    withrawdata = 1;
		
		} else {
		    // any more?
		}	
			
		i++;
    }
    
    if (model_status == MODEL_STATUS_EST) {
		if (dfile == "") {
		    printf("Please specify the input data file for model estimation!\n");
		    return 1;
		}
		
		pmodel->model_status = model_status;
		
		if (K > 0) {
		    pmodel->K = K;
		}
		
		if (alpha >= 0.0) {
		    pmodel->alpha = alpha;
		} else {
		    // default value for alpha
		    pmodel->alpha = 50.0 / pmodel->K;
		}
		
		if (beta >= 0.0) {
		    pmodel->beta = beta;
		}
		
		if (niters > 0) {
		    pmodel->niters = niters;
		}
		
		if (savestep > 0) {
		    pmodel->savestep = savestep;
		}
		
		if (twords > 0) {
		    pmodel->twords = twords;
		}
		
		pmodel->dfile = dfile;
		
		string::size_type idx = dfile.find_last_of("/");			
		if (idx == string::npos) {
		    pmodel->dir = "./";
		} else {
		    pmodel->dir = dfile.substr(0, idx + 1);
		    pmodel->dfile = dfile.substr(idx + 1, dfile.size() - pmodel->dir.size());
		    printf("dir = %s\n", pmodel->dir.c_str());
		    printf("dfile = %s\n", pmodel->dfile.c_str());
		}
    } 

    if (model_status == MODEL_STATUS_EST_FLDA) {
		if (dfile == "") {
		    printf("Please specify the input tweets file for model estimation!\n");
		    return 1;
		}

		if (ffile == "") {
		    printf("Please specify the input friend file for model estimation!\n");
		    return 1;
		}
		
		pmodel->model_status = model_status;
		
		if (K > 0) {
		    pmodel->K = K;
		}
		
		if (alpha >= 0.0) {
		    pmodel->alpha = alpha;
		} else {
		    // default value for alpha
		    pmodel->alpha = 50.0 / pmodel->K;
		}
		
		if (beta >= 0.0) {
		    pmodel->beta = beta;
		}
		
		if (niters > 0) {
		    pmodel->niters = niters;
		}
		
		if (savestep > 0) {
		    pmodel->savestep = savestep;
		}
		
		if (twords > 0) {
		    pmodel->twords = twords;
		}

		if (tusers > 0) {
		    pmodel->tusers = tusers;
		}
		
		pmodel->dfile = dfile;
		pmodel->ffile = ffile;
		
		string::size_type idx = dfile.find_last_of("/");			
		if (idx == string::npos) {
		    pmodel->dir = "./";
		} else {
		    pmodel->dir = dfile.substr(0, idx + 1);
		    pmodel->dfile = dfile.substr(idx + 1, dfile.size() - pmodel->dir.size());
		    printf("dir = %s\n", pmodel->dir.c_str());
		    printf("dfile = %s\n", pmodel->dfile.c_str());
		}
    } 
    
    if (model_status == MODEL_STATUS_UNKNOWN) {
		printf("Please specify the task you would like to perform (-est/-est_flda)!\n");
		return 1;
    }
    
    return 0;
}

int utils::read_and_parse(string filename, model * pmodel) {
    // open file <model>.others to read:
    // alpha=?
    // beta=?
    // ntopics=?
    // ndocs=?
    // nwords=?
    // citer=? // current iteration (when the model was saved)
    
    FILE * fin = fopen(filename.c_str(), "r");
    if (!fin) {
	printf("Cannot open file: %s\n", filename.c_str());
	return 1;
    }
    
    char buff[BUFF_SIZE_SHORT];
    string line;
    
    while (fgets(buff, BUFF_SIZE_SHORT - 1, fin)) {
		line = buff;
		strtokenizer strtok(line, "= \t\r\n");
		int count = strtok.count_tokens();
		
		if (count != 2) {
		    // invalid, ignore this line
		    continue;
		}

		string optstr = strtok.token(0);
		string optval = strtok.token(1);
		
		if (optstr == "alpha") {
		    pmodel->alpha = stof(optval.c_str());
		    
		} else if (optstr == "beta") {	    
		    pmodel->beta = stof(optval.c_str());
		
		} else if (optstr == "ntopics") {
		    pmodel->K = stoi(optval.c_str());
		
		} else if (optstr == "ndocs") {	   
		    pmodel->M = stoi(optval.c_str());
		 
		} else if (optstr == "nwords") {
		    pmodel->V = stoi(optval.c_str());
		
		} else if (optstr == "liter") {
		    pmodel->liter = stoi(optval.c_str());
		
		} else {
		    // any more?
		}
    }
    
    fclose(fin);
    
    return 0;
}

string utils::generate_model_name(int iter) {
    string model_name = "model-";

    char buff[BUFF_SIZE_SHORT];
    
    if (0 <= iter && iter < 10) {
	sprintf(buff, "0000%d", iter);
    } else if (10 <= iter && iter < 100) {
	sprintf(buff, "000%d", iter);
    } else if (100 <= iter && iter < 1000) {
	sprintf(buff, "00%d", iter);
    } else if (1000 <= iter && iter < 10000) {
	sprintf(buff, "0%d", iter);
    } else {
	sprintf(buff, "%d", iter);
    }
    
    if (iter >= 0) {
	model_name += buff;
    } else {
	model_name += "final";
    }
    
    return model_name;
}

void utils::sort(vector<double> & probs, vector<int> & words) {
    for (int i = 0; i < probs.size() - 1; i++) {
	for (int j = i + 1; j < probs.size(); j++) {
	    if (probs[i] < probs[j]) {
		double tempprob = probs[i];
		int tempword = words[i];
		probs[i] = probs[j];
		words[i] = words[j];
		probs[j] = tempprob;
		words[j] = tempword;
	    }
	}
    }
}

void utils::quicksort(vector<pair<int, double> > & vect, int left, int right) {
    int l_hold, r_hold;
    pair<int, double> pivot;
    
    l_hold = left;
    r_hold = right;    
    int pivotidx = left;
    pivot = vect[pivotidx];

    while (left < right) {
		while (vect[right].second <= pivot.second && left < right) {
		    right--;
		}
		if (left != right) {
		    vect[left] = vect[right];
		    left++;
		}
		while (vect[left].second >= pivot.second && left < right) {
		    left++;
		}
		if (left != right) {
		    vect[right] = vect[left];
		    right--;
		}
    }

    vect[left] = pivot;
    pivotidx = left;
    left = l_hold;
    right = r_hold;
    
    if (left < pivotidx) {
	quicksort(vect, left, pivotidx - 1);
    }
    if (right > pivotidx) {
	quicksort(vect, pivotidx + 1, right);
    }    
}

bool utils::prob_vector_comp(const pair<int, double> & i, const pair<int, double> & j) {
    return (i.second > j.second);
}
