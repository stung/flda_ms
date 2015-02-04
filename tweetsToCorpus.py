import os
import string
import gzip
import sys
try:
	import xml.etree.cElementTree as ET
except ImportError:
	import xml.etree.ElementTree as ET

# Gunzips and parses one XML Twitter user tweet file
# Returns one line with all of the tweets
def gzParse(xmlPath):
	# reads gzip file
	xmlFile = gzip.open(xmlPath)
	fileContent = xmlFile.read()

	# Some XML files have an extra georss namespace
	# Removing the namespace if it exists
	modFileContent = string.replace(fileContent, 'georss:', '')

	# Converting contents into an XML Element Tree
	tweets = ""
	root = ET.fromstring(modFileContent)
	for title in root.iter('title'):
		tweets += " " + title.text
	return tweets

# Collecting all of the friends and tweets in each run
corpus = []
for root, dirs, files in os.walk('1'):
	for name in files:
		if 'tweets.rss.gz' in name:
			filePath = os.path.join(root, name)

			# reads gzipped xml files and extracts tweets
			userTweets = gzParse(filePath)
			corpus.append(userTweets)

# Writing the corpus into a file
corpusFile = open('corpus.txt', 'w')
corpusString = '\n'.join(corpus)

# Ran into some encoding issues, needed to encode everything into utf-8 before writing
corpusString = corpusString.encode('utf-8').strip()
corpusFile.write(corpusString)
corpusFile.close()
