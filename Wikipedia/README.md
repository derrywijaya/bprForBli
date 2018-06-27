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



