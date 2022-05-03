#!/bin/bash

STANFORD_DATSETS_DIR="dataset/stanfordTreeBank"
mkdir -p $STANFORD_DATSETS_DIR

wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip -P $STANFORD_DATSETS_DIR
unzip $STANFORD_DATSETS_DIR/stanfordSentimentTreebank.zip -d $STANFORD_DATSETS_DIR
rm $STANFORD_DATSETS_DIR/stanfordSentimentTreebank.zip
mv $STANFORD_DATSETS_DIR/stanfordSentimentTreebank/datasetSentences.txt $STANFORD_DATSETS_DIR/datasetSentences.txt
rm -r $STANFORD_DATSETS_DIR/stanfordSentimentTreebank
rm -r $STANFORD_DATSETS_DIR/__MACOSX

WIKI_DATASET_DIR="dataset/wikiBillionChars"
mkdir -p $WIKI_DATASET_DIR

wget -c http://mattmahoney.net/dc/enwik9.zip -P $WIKI_DATASET_DIR
unzip $WIKI_DATASET_DIR/enwik9.zip -d $WIKI_DATASET_DIR
rm $WIKI_DATASET_DIR/enwik9.zip
perl wikiclean.pl $WIKI_DATASET_DIR/enwik9 > $WIKI_DATASET_DIR/enwik9_cleaned
python3 processFile.py