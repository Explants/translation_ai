import re
import string
from unicodedata import normalize
from numpy import array
from pickle import dump
with open('C:\Users\kenco\OneDrive\Desktop\AI Stuff\translation_ai\eng-german_example data.txt', 'rt') as file:
    text = file.read()

lines = text.strip().split('\n')
pairs = [line.split('\t') for line in lines]

cleaned = []
re_print = re.compile('[^%S]' % re.escape(string.printable()))
table = str.maketrans('','', string.punctuation)
for pair in lines:
    clean_pair = []
    for set in pair:
        line = normalize(('NFD', line).encode('ascii', 'ignore') )
        line = line.decode('utf-8')
        line = line.split()
        line = [word.lower() for word in line]
        line = [word.translate(table) for word in line]
        line = [re_print.sub('', w) for w in line]
        line = [word for word in line if word.isalpha()]
        clean_pair.append(''.join(line))
    cleaned.append(clean_pair)
    cleaned = array(cleaned)

dump(clean_pair, open('english-german.pkl', 'wb'))