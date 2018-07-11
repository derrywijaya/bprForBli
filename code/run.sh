#!/bin/sh
# Take arguments: 2-letter language code of the foreign language, english embeddings, foreign embeddings, json bilingual dictionary, 
# and the list of foreign words to be translated (one word per line)
#  e.g., ./run.sh sw en.vec sw.vec output/dictionary/sw.json sw.words

lang=$1
english=$2
foreign=$3
dictionary=$4
foreignwords=$5
path_to_code=$6

# process the embedding and word files to be in the format we require
export LC_ALL=C
tail -n +2 $english | sed 's/^/row-/g' | sed 's/,/_/g' > user.en.vec
tail -n +2 $foreign | sed 's/^/column-/g' | sed 's/,/_/g' > user.$lang.vec
export LC_ALL=""

# assemble the train and test files for learning the mapping between the embedding spaces from the bilingual dictionary
java -Dfile.encoding=UTF-8 -cp .:$path_to_code:$path_to_code/lib/json-simple-1.1.1.jar readDictionary $dictionary user.en.vec user.$lang.vec train-$lang.txt test-$lang.txt > context-$lang.txt
grep '^row-' context-$lang.txt > context-$lang-en.txt 
grep '^column-' context-$lang.txt > context-$lang-$lang.txt 
grep ' column-' train-$lang.txt > $lang-dict-mturk.txt
grep ' column-' test-$lang.txt >> $lang-dict-mturk.txt

# assemble input files for BPR
java -Dfile.encoding=UTF-8 -cp $path_to_code createDataForBPR $foreignwords $foreign $lang $lang-en.train $lang-en.vector.en $lang-en.vector.extended $lang-en.words

# learn mapping between embedding spaces
# first scale the input embeddings 
source $path_to_code/env/bin/activate
python $path_to_code/scale.py --input $lang-en.vector.en --output $lang-en.vector.en.scaled.vector
cut -f1 -d' ' $lang-en.vector.en > $lang-en.vector.en.words
cut -f2- -d' ' $lang-en.vector.extended > $lang-en.vector.extended.vector
cut -f1 -d' ' $lang-en.vector.extended > $lang-en.vector.extended.words
paste -d' ' $lang-en.vector.en.words $lang-en.vector.en.scaled.vector > $lang-en.vector.en.scaled
# then, assemble the input and output embeddings for learning the mapping
java -Dfile.encoding=UTF-8 -cp $path_to_code createTrainAndDevMapping $lang-en.vector.extended.words $lang-en.vector.extended.vector $lang-en.vector.en.words $lang-en.vector.en.scaled.vector $lang-dict-mturk.txt $lang 
# finally, learn the mapping and project the target embeddings to the shared space
python $path_to_code/nnlinreg4xv.py --traininput $lang-input-train.txt --trainoutput $lang-output-train.txt --testinput $lang-input-test.txt --testoutput $lang-output-test.txt --verbose True --num_epochs 10000 --num_hidden 10000 --project $lang-en.vector.en.scaled --towrite $lang-en.vector.projected.vector

# assemble input files for BPR 
java -Dfile.encoding=UTF-8 -cp $path_to_code createDataForBPRFinal $lang
sed 's/,/_/g' $lang.totranslate > $lang.totranslate.norm
sed 's/,/_/g' $lang.extended > $lang.extended.norm
sed 's/,/_/g' $lang.train > $lang.train.norm
sed 's/,/_/g' $lang.train.vec > $lang.train.vec.norm
sed 's/,/_/g' $lang.translated > $lang.translated.norm
paste -d' ' $lang-en.vector.en.words $lang-en.vector.projected.vector > $lang-en.vector.projected
sed 's/,/_/g' $lang-en.vector.projected > $lang-en.vector.projected.norm
cat $lang-en.vector.projected.norm $lang.train.vec.norm > $lang-en.context

# move all the data to the language specific folder in output/
mkdir -p output/$lang/
mv user.en.vec output/$lang/
mv user.$lang.vec output/$lang/
mv train-$lang.txt output/$lang/
mv test-$lang.txt output/$lang/
mv context-$lang.txt output/$lang/
mv context-$lang-en.txt output/$lang/ 
mv context-$lang-$lang.txt output/$lang/
mv $lang-dict-mturk.txt output/$lang/
mv $lang-en.train output/$lang/
mv $lang-en.vector.en output/$lang/
mv $lang-en.vector.extended output/$lang/
mv $lang-en.words output/$lang/
mv $lang-en.vector.en.scaled.vector output/$lang/
mv $lang-en.vector.en.words output/$lang/
mv $lang-en.vector.extended.vector output/$lang/
mv $lang-en.vector.extended.words output/$lang/
mv $lang-en.vector.en.scaled output/$lang/
mv $lang-input-t* output/$lang/
mv $lang-output-t* output/$lang/
mv $lang-en.vector.projected.vector output/$lang/
mv $lang.totranslate* output/$lang/ 
mv $lang.extended* output/$lang/
mv $lang.train* output/$lang/
mv $lang.translated* output/$lang/
mv $lang-en.vector.projected* output/$lang/
mv $lang-en.context output/$lang/

# create config file for BPR

echo "dataset.ratings.lins=./output/$lang/$lang.train.norm" > BPR-$lang.conf
echo "dataset.context.lins=./output/$lang/$lang-en.context" >> BPR-$lang.conf
echo "dataset.translation.lins=./englishwordslemma.txt" >> BPR-$lang.conf
echo "dataset.extended=./output/$lang/$lang.extended.norm" >> BPR-$lang.conf
echo "dataset.tocompute=./output/$lang/$lang.totranslate.norm" >> BPR-$lang.conf
echo "ratings.setup=-columns 0 1 -threshold 0" >> BPR-$lang.conf
echo "mainlanguage=$lang" >> BPR-$lang.conf
echo "recommender=BPRWEExtended" >> BPR-$lang.conf
echo "evaluation.setup=test-set -f ./output/$lang/test-$lang.txt -p on --rand-seed 1 --test-view all" >> BPR-$lang.conf
echo "item.ranking=on -topN -1 -ignore -1" >> BPR-$lang.conf
echo "num.factors=20" >> BPR-$lang.conf
echo "num.max.iter=100" >> BPR-$lang.conf
echo "learn.rate=0.01 -max -1 -bold-driver" >> BPR-$lang.conf
echo "reg.lambda=0.1 -u 0.1 -i 0.1 -b 0.1 -c 0.1" >> BPR-$lang.conf

# the output of BPR will be stored in /demo/Results/$lang/translations.txt
mkdir -p demo/Results/$lang/
echo "output.setup=on -dir ./demo/Results/$lang/" >> BPR-$lang.conf

# run BPR using the config file just created
MAVEN_OPTS="-Xmx150G" mvn exec:java -Dfile.encoding=UTF-8 -Dexec.mainClass=librec.main.LibRec -Dexec.args="-c BPR-$lang.conf"

# write the output to the file /demo/Results/$lang/translations.txt
java -Dfile.encoding=UTF-8 -cp $path_to_code writeOutput demo/Results/$lang/BPRWEExtended-top-10-items-extended.txt output/$lang/$lang.translated.norm > demo/Results/$lang/translations.txt
