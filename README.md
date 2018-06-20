# bprForBli

The code for the paper: Learning Translations via Matrix Completion, Wijaya et al., EMNLP 2017

REQUIREMENTS:

java

maven

tensorflow

numpy

math

sklearn

INSTRUCTIONS:
(1) Download bprForBli.tar.xz and unzip it

(2) cd to librec directory

(3) install happy coding:
mvn install:install-file -Dfile=lib/happy.coding.utils-1.2.5.jar -DgroupId=happy.coding -DartifactId=utils -Dversion=1.2.5 -Dpackaging=jar

(4) mvn clean install

(5) Gather the required files:
-- en.vec: english word embeddings, space separated, first line is the length and dimension, first column are words, all lower-cased
-- fo.vec: foreign word embeddings, space separated, first line is the length and dimension, first column are words, all lower-cased
-- fo.json: bilingual dictionary, in JSON format (see demo/Datasets/dictionary/ for example file) — all lower cased
-- fo.words: list of foreign words to be translated, one word per line, all lower-cased

To generate the .vec files yourself for a new language, you can use gensim:

import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = gensim.models.word2vec.LineSentence(filename)
model = gensim.models.Word2Vec(sentences, iter=15, negative=15)
model.wv.save_word2vec_format(fileoutname)

where filename is the file containing tokenized and lower-cased (English/Foreign) text

(6) Run bprForBli using run.sh with 5 arguments: (a) 2-letter language code of the foreign language, (b) en.vec, (c) fo.vec, (d) fo.json, (e) fo.words e.g., ./run.sh sw en.vec sw.vec sw.json sw.words

TROUBLESHOOTING:
If out-of-memory, modify this line in run.sh with higher memory requirement: e.g., -Xmx200G
MAVEN_OPTS="-Xmx150G" mvn exec:java -Dexec.mainClass=librec.main.LibRec -Dexec.args="-c multi/config/BPR-$lang.conf"

If you encounter problem, try running the commands in run.sh line by line to see where the errors are. This code has been tested on linux machine. 
