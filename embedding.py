# -*- coding: utf-8 -*-
#
# This file is part of Istex_Mental_Rotation.
# Copyright (C) 2017 SSbE ERIC Laboratory.
#
# This is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

# Load and transform ISTEX and wiki articles into bag_of_words decomposed by SVD.

# co-author : Lucie Martinet <lucie.martinet@univ-lorraine.fr>
# co-author : Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr.>
# Affiliation: University of Lyon, ERIC Laboratory, Lyon2, University of Lorraine

# Thanks to ISTEX project for the funding


# bow_svd.py --istex_dir data
import os, argparse, pickle, json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from utils import Paragraphs, Lemmatizer, avg_inner_sim, n_neg_sampling_avg_inner_sim
import IPython

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_dir", default="sample_data/wiki", type=str)  # contains wikipedia text files
    parser.add_argument("--istex_dir", default='sample_data/ISTEX/', type=str)  # contains .json files
    parser.add_argument("--ucbl_file", default='sample_data/sportArticlesAsIstex_UniqID_182.json',
                        type=str)  # is a .json file
    parser.add_argument("--istex_mr_file", default='sample_data/MentalRotationInMetaDataIstexWithoutAnnotated.json',
                        type=str)  # is a .json file
    parser.add_argument("--max_nb_wiki", default=100000, type=int)  # maximum number of Wikipedia paragraphs to use
    parser.add_argument("--paragraphs_per_article", default=2,
                        type=int)  # maximum number of paragraphs to load per article
    parser.add_argument("--vectorizer_type", default="tfidf",
                        type=str)  # possible values: "tfidf" and "count", futurework: "doc2vec"
    parser.add_argument("--lemmatizer", default=0, type=int)  # for using lemmatization_tokenizer
    parser.add_argument("--mx_ngram", default=2, type=int)  # the upper bound of the ngram range
    parser.add_argument("--mn_ngram", default=1, type=int)  # the lower bound of the ngram range
    parser.add_argument("--stop_words", default=1, type=int)  # filtering out English stop-words
    parser.add_argument("--vec_size", default=150, type=int)  # the size of the vector in the semantics space
    parser.add_argument("--min_count", default=20,
                        type=int)  # minimum frequency of the token to be included in the vocabulary
    parser.add_argument("--max_df", default=0.95,
                        type=float)  # how much vocabulary percent to keep at max based on frequency
    parser.add_argument("--debug", default=0, type=int)  # embed IPython to use the decomposed matrix while running
    parser.add_argument("--nb_neg_samplings", default=100,
                        type=int)  # number of negative samplings times used for evaluation
    parser.add_argument("--compress", default="pickle", type=str)  # for dumping resulted files
    parser.add_argument("--out_dir", default="results", type=str)  # name of the output directory

    args = parser.parse_args()
    istex = args.istex_dir
    if istex == "None":
        istex = None
    ucbl = args.ucbl_file
    istex_mr_file = args.istex_mr_file
    if istex_mr_file == "None":
        istex_mr_file = None
    wiki_dir = args.wiki_dir
    if wiki_dir == "None":
        wiki_dir = None
    max_nb_wiki = args.max_nb_wiki
    paragraphs_per_article = args.paragraphs_per_article
    vectorizer_type = args.vectorizer_type
    lemmatizer = args.lemmatizer
    if lemmatizer:
        lemmatizer = Lemmatizer()
    else:
        lemmatizer = None
    mx_ngram = args.mx_ngram
    mn_ngram = args.mn_ngram
    stop_words = args.stop_words
    if stop_words:
        stop_words = 'english'
    else:
        stop_words = None
    n_components = args.vec_size
    min_count = args.min_count
    max_df = args.max_df
    debug = args.debug
    nb_neg_samplings = args.nb_neg_samplings
    compress = args.compress
    out_dir = args.out_dir

    paragraphs = Paragraphs(istex=istex, ucbl=ucbl, istex_mr=istex_mr_file, wiki=wiki_dir,
                            max_nb_wiki_paragraphs=max_nb_wiki, paragraphs_per_article=paragraphs_per_article)
    print(paragraphs.shape)
    pickle.dump(paragraphs, open('para_pickle.pickle', 'wb'))
