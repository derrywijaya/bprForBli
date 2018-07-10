## Getting and processing text from Wikipedia

1. Download your target language text from Wikipedia dump: https://dumps.wikimedia.org/backup-index.html. For example, to get Wikipedia Korean dumps:
```
wget https://dumps.wikimedia.org/kowiki/20180620/kowiki-20180620-pages-articles-multistream.xml.bz2
```
2. Strip the HTML tags and obtain the text from the dump. This will create the text files of 1GB each containing the text. 
```
git clone https://github.com/attardi/wikiextractor
cd wikiextractor
bzcat wikidumpfile.bz2 | WikiExtractor.py -b 1G -o directorywhereyouwanttoputthetext/ -
```
3. Combine the output text files into one text file and run your favorite sentence-splitter and tokenizer to sentence-split, tokenize, and lower case the text. For example, using Stanford CoreNLP:
```
// creates a StanfordCoreNLP object
Properties props = new Properties();
props.put("annotators", "tokenize, ssplit");
StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

// read some text in the text variable
String text = ... // Add your text here!

// create an empty Annotation just with the given text
Annotation document = new Annotation(text);

// run all Annotators on this text
pipeline.annotate(document);

// these are all the sentences in this document
// a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
List<CoreMap> sentences = document.get(SentencesAnnotation.class);

for(CoreMap sentence: sentences) {
  // traversing the words in the current sentence
  // a CoreLabel is a CoreMap with additional token-specific methods
  String sent = "";
  for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
    // this is the text of the token
    String word = token.get(TextAnnotation.class);
    word = word.toLowerCase();
    sent = sent + word + " ";
  }
  // Write each tokenized, lower cased sentence out (one sentence = one line)
  System.out.println(sent.trim());
}
```




