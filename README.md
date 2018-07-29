# bprForBli
## Bayesian Personalized Ranking for Bilingual Lexicon Induction:

The code for the paper: [Learning Translations via Matrix Completion](https://www.seas.upenn.edu/~derry/bpr.pdf), Wijaya et al., EMNLP 2017

Citation:

```
@inproceedings{wijaya2017learning,
  title={Learning Translations via Matrix Completion},
  author={Wijaya, Derry Tanti and Callahan, Brendan and Hewitt, John and Gao, Jie and Ling, Xiao and Apidianaki, Marianna and Callison-Burch, Chris},
  booktitle={Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  pages={1452--1463},
  year={2017}
}
```

## Requirements:

python 2.7, java, maven, tensorflow, numpy, math, sklearn, sed

## Instructions:
1. cd into the `code/` subdirectory
2. Install `happy coding` using Maven:
```
mvn install:install-file -Dfile=lib/happy.coding.utils-1.2.5.jar -DgroupId=happy.coding -DartifactId=utils -Dversion=1.2.5 -Dpackaging=jar
```
3. Do a clean install, then clean and compile: 
```
mvn clean install
mvn clean
mvn compile
```
4. run `javac -cp lib/json-simple-1.1.1.jar:lib/happy.coding.utils-1.2.5.jar *java`
5. Install the python dependencies 
```
python -m virtualenv env
source env/bin/activate
pip install --upgrade pip
pip install numpy
pip install scipy
pip install sklearn
pip install tensorflow
pip install pymagnitude
pip install pandas
pip install pathlib
pip install lz4
pip install xxhash
pip install annoy
```
6. Gather the required files:
* English word embeddings, space separated, first line is the length and dimension, first column are words, all lower-cased
* Foreign word embeddings, space separated, first line is the length and dimension, first column are words, all lower-cased
* Bilingual dictionary, in JSON format, all lower cased in our Json format e.g., [ko.json](https://www.seas.upenn.edu/~derry/ko.json)
* List of foreign words to be translated, one word per line, all lower-cased 
* For BPR we also use several files derived from Wikipedia 
You can grab a [tarball of necessary data files here](https://cis.upenn.edu/~ccb/data/emnlp-2017/data.tar.gz). Then run
```
wget https://cis.upenn.edu/~ccb/data/emnlp-2017/data.tar.gz
tar xfz data.tar.gz
mv data/* .
```
In addition the tarball contains the following files:
* englishwordslemma.txt - lemma of english words (used in the evaluation of our paper)
* interlanguage.txt - the interlingual links from Wikipedia that is used to populate the matrix (used in the matrix completion using Wikipedia phase)
* languagesdata - the foreign-english translations from our MTurk dictionary (used in the matrix completion using third languages phase that is linked to interlingual links from Wikipedia)
* namedentitiesfromwiki-uniq.txt - list of named entities from Wikipedia (used in finding similarity of strings to get better named entities translation -- added after the paper, only works for Roman script languages)
* wiki.en.top.words - top 100K english words from Wikipedia based on frequency (used to filter target English words -- this is the complete set of target English words -- if your target vocabulary is more than these top 100K, you can replace this file with your total English vocabulary file -- note the format it prepends `row-` and replaces `,` with `_`)


To generate the embedding files (\*`.vec`) for a new language, you can use `gensim`: 
```
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = gensim.models.word2vec.LineSentence(filename)
model = gensim.models.Word2Vec(sentences, iter=15, negative=15)
model.wv.save_word2vec_format(fileoutname)
```
where `filename` is the file containing `sentence-separated`, `tokenized`, and `lower-cased` text (one sentence = one line). For example, we `sentence-split`, `tokenized`, and `lower-cased` Wikipedia English text to generate `en.vec` and Wikipedia Korean text to generate `ko.vec` (see [here](Wikipedia/README.md) for instructions to get text from Wikipedia and to sentence-split, tokenize, and lower-case it)

7. Activate your virtual environment if not already activated 
``
source env/bin/activate
``

8. Execute `run.sh` with 6 arguments: 
* 2-letter language code of the foreign language
* English word embeddings file e.g., `en.vec`
* Foreign word embeddings file e.g., `ko.vec`
* Bilingual dictionary file e.g., `ko.json`
* List of foreign language words to be translated e.g., `ko.words`
* The path to where your `code/` directory where run.sh is located.
```
./run.sh ko en.vec ko.vec ko.json ko.words ./
```
NOTE: If your bilingual dictionary is not on json format, you can run `runTab.sh` in the `code` directory instead with the same parameters. It takes bilingual dictionary that is in tab-separated format i.e., each line is `foreignWord` tab `englishWord`

## Translate words using Magnitude's Approximate kNN:

If you want to produce translations fast, run `runWithMagnitude.sh` in the `code` directory instead of `run.sh` (similarly, if your bilingual dictionary is not of JSON format but is tab-separated, run `runTabWithMagnitude.sh` in the `code` directory instead). Note the extra parameter for these scripts:

To produce translations with the whole BPR pipeline using Magnitude's fast approximate kNN, run:
```
./runWithMagnitude.sh ko en.vec ko.vec ko.json ko.words ./ BPR
```

To produce translations with only our non-linear mapping between the language spaces and Magnitude's fast approximate kNN, run:
```
./runWithMagnitude.sh ko en.vec ko.vec ko.json ko.words ./ monolingual
```

The details of what `runWithMagnitude.sh` contains are below:

1. In the language's directory, create a sub-directory called 'matrices' and move all matrices of bilingual embeddings to this directory. Please find files with the following names: 'wikiPMatrix.txt', 'wikiQMatrix.txt', 'thirdTPMatrix.txt', 'thirdTQMatrix.txt', 'EMatrix.txt', 'extendedMatrix.txt'. Note that some languages won't have all six files.

2. Create a directory that will later store embeddings in Magnitude format

3. Run the following command to convert matrices to a format that Magnitude can process:
```bash
python3 convertMatrix.py <language's dir> <Magnitude's dir>
e.g python3 convertMatrix.py demo/Results/ko/ demo/Results/ko/magnitudes/
```
4. Install gensim and [magnitude](https://github.com/plasticityai/magnitude) -- for both python2 and python3 if not yet installed:
```
pip install gensim
pip install pymagnitude
python3 -m pip install pymagnitude
```
5. All converted matrices are stored in a sub-directory called 'processed'. Convert each matrix to Magnitude by running:
```bash
python -m pymagnitude.converter -i <input path> -o <output path> -a 
```
6. Produce translation outputs by running:
```bash
python translate.py <language's dir> <Magnitude's dir>
```

## Pre-computed Translations
Available in our [website](https://www.seas.upenn.edu/~derry/translations.html) 

## Troubleshooting
1. If out-of-memory, modify this line in `run.sh` with higher memory requirement: e.g., -Xmx200G
```
MAVEN_OPTS="-Xmx200G" mvn exec:java -Dexec.mainClass=librec.main.LibRec -Dexec.args="-c multi/config/BPR-$lang.conf"
```
2. If you encounter problem, try running the commands in `run.sh` line by line and debugging the errors. This code has been tested on a Linux machine, but running it on other machines may cause some of the commands (e.g., `sed`, `cut`) in the script to run differently. If running `sed` gives an error "sed: RE error: illegal byte sequence" modify `run.sh` to include these commands instead:
```
export OLDLANG=$LANG
export LANG=C
tail -n +2 $english | sed 's/^/row-/g' | sed 's/,/_/g' > user.en.vec
tail -n +2 $foreign | sed 's/^/column-/g' | sed 's/,/_/g' > user.$lang.vec
export LANG=$OLDLANG
unset OLDLANG
```
`cut` in Mac differs from Linux, in Mac you should modify the `cut --complement` commands in `run.sh` from
```
cut -f1 -d' ' $lang-en.vector.extended --complement > $lang-en.vector.extended.vector
```
to
```
cut -f2- -d' ' $lang-en.vector.extended > $lang-en.vector.extended.vector
```

3. Java often has different default encoding in different environment. This code assumes UTF-8 encoding. To find out your machine's default Java encoding, you can run this command inside a Java program:
```
import java.nio.charset.Charset;
System.out.println(Charset.defaultCharset().name());
```
If your default Java encoding is not UTF-8, append this to all commands involving `java` in `run.sh`:
```
java -Dfile.encoding=UTF-8 
```
