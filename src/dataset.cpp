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
#include <stdlib.h>
#include "constants.h"
#include "strtokenizer.h"
#include "dataset.h"
#include <iostream>

using namespace std;

int dataset::write_wordmap(string wordmapfile, mapword2id * pword2id) {
    FILE * fout = fopen(wordmapfile.c_str(), "w");
    if (!fout) {
	printf("Cannot open file %s to write!\n", wordmapfile.c_str());
	return 1;
    }    
    
    mapword2id::iterator it;
    fprintf(fout, "%d\n", pword2id->size());
    for (it = pword2id->begin(); it != pword2id->end(); it++) {
	fprintf(fout, "%s %d\n", (it->first).c_str(), it->second);
    }
    
    fclose(fout);
    
    return 0;
}

int dataset::read_wordmap(string wordmapfile, mapword2id * pword2id) {
    pword2id->clear();
    
    FILE * fin = fopen(wordmapfile.c_str(), "r");
    if (!fin) {
	printf("Cannot open file %s to read!\n", wordmapfile.c_str());
	return 1;
    }    
    
    char buff[BUFF_SIZE_SHORT];
    string line;
    
    fgets(buff, BUFF_SIZE_SHORT - 1, fin);
    int nwords = atoi(buff);
    
    for (int i = 0; i < nwords; i++) {
	fgets(buff, BUFF_SIZE_SHORT - 1, fin);
	line = buff;
	
	strtokenizer strtok(line, " \t\r\n");
	if (strtok.count_tokens() != 2) {
	    continue;
	}
	
	pword2id->insert(pair<string, int>(strtok.token(0), atoi(strtok.token(1).c_str())));
    }
    
    fclose(fin);
    
    return 0;
}

int dataset::read_wordmap(string wordmapfile, mapid2word * pid2word) {
    pid2word->clear();
    
    FILE * fin = fopen(wordmapfile.c_str(), "r");
    if (!fin) {
	printf("Cannot open file %s to read!\n", wordmapfile.c_str());
	return 1;
    }    
    
    char buff[BUFF_SIZE_SHORT];
    string line;
    
    fgets(buff, BUFF_SIZE_SHORT - 1, fin);
    int nwords = atoi(buff);
    
    for (int i = 0; i < nwords; i++) {
	fgets(buff, BUFF_SIZE_SHORT - 1, fin);
	line = buff;
	
	strtokenizer strtok(line, " \t\r\n");
	if (strtok.count_tokens() != 2) {
	    continue;
	}
	
	pid2word->insert(pair<int, string>(atoi(strtok.token(1).c_str()), strtok.token(0)));
    }
    
    fclose(fin);
    
    return 0;
}

int dataset::read_trndata(string dfile, string wordmapfile) {
    mapword2id word2id;
    
    FILE * fin = fopen(dfile.c_str(), "r");
    if (!fin) {
		printf("Cannot open corpus file %s to read!\n", dfile.c_str());
		return 1;
    }   
    
    mapword2id::iterator it;    
    char buff[BUFF_SIZE_LONG];
    string line;
    
    // get the number of documents
    fgets(buff, BUFF_SIZE_LONG - 1, fin);
    M = atoi(buff);
    if (M <= 0) {
		printf("No document available!\n");
		return 1;
    }
    
    // allocate memory for corpus
    if (docs) {
		deallocate();
    } else {
		docs = new document*[M];
    }
    
    // set number of words to zero
    V = 0;
    
    for (int i = 0; i < M; i++) {
		fgets(buff, BUFF_SIZE_LONG - 1, fin);
		line = buff;
		strtokenizer strtok(line, " \t\r\n");
		int length = strtok.count_tokens();
		// cout << i << endl;

		if (length < 0) {
		    printf("Invalid (empty) document!\n");
		    deallocate();
		    M = V = 0;
		    return 1;
		}
		
		// allocate new document
		document * pdoc = new document(length);
		
		for (int j = 0; j < length; j++) {
		    it = word2id.find(strtok.token(j));
		    if (it == word2id.end()) {
				// word not found, i.e., new word
				pdoc->words[j] = word2id.size();
				word2id.insert(pair<string, int>(strtok.token(j), word2id.size()));
		    } else {
				pdoc->words[j] = it->second;
		    }
		}
		
		// add new doc to the corpus
		add_doc(pdoc, i);
    }
    
    fclose(fin);
    
    // write word map to file
    if (write_wordmap(wordmapfile, &word2id)) {
		return 1;
    }
    
    // update number of words
    V = word2id.size();
    
    return 0;
}

int dataset::read_frnddata(string dfile, string friendmapfile) {
	mapword2id word2id;
    
    FILE * fin = fopen(dfile.c_str(), "r");
    if (!fin) {
		printf("Cannot open friend file %s to read!\n", dfile.c_str());
		return 1;
    }   
    
    mapword2id::iterator it;    
    char buff[BUFF_SIZE_LONG];
    string line;
    
    // get the number of documents
    fgets(buff, BUFF_SIZE_LONG - 1, fin);
    M = atoi(buff);
    if (M <= 0) {
		printf("No document available!\n");
		return 1;
    }
    
    // allocate memory for corpus
    if (docs) {
		deallocate();
    } else {
		docs = new document*[M];
    }
    
    // set number of words to zero
    V = 0;
    
    for (int i = 0; i < M; i++) {
		fgets(buff, BUFF_SIZE_LONG - 1, fin);
		line = buff;
		strtokenizer strtok(line, " \t\r\n");
		int length = strtok.count_tokens();
		// cout << i << endl;

		if (length < 0) {
		    printf("Invalid (empty) document!\n");
		    deallocate();
		    M = V = 0;
		    return 1;
		}
		
		// allocate new document
		document * pdoc = new document(length);
		
		for (int j = 0; j < length; j++) {
		    it = word2id.find(strtok.token(j));
		    if (it == word2id.end()) {
				// word not found, i.e., new word
				pdoc->words[j] = word2id.size();
				word2id.insert(pair<string, int>(strtok.token(j), word2id.size()));
		    } else {
				pdoc->words[j] = it->second;
		    }
		}
		
		// add new doc to the corpus
		add_doc(pdoc, i);
    }
    
    fclose(fin);
    
    // write word map to file
    if (write_wordmap(friendmapfile, &word2id)) {
		return 1;
    }
    
    // update number of words
    V = word2id.size();
    
    return 0;
}
