import os
import string
import gzip
import sys
import re
from nltk.corpus import stopwords

#import nltk
#nltk.download('stopwords')
cachedStopWords = stopwords.words("english")

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
	userIdx = None
	user = None
	root = ET.fromstring(modFileContent)

	for title in root.iter('title'):
		rawTweet = title.text
		if userIdx is None:
			userIdx = rawTweet.index(':')
			user = rawTweet[:userIdx].lower()
		tweet = cleanTweet(rawTweet[userIdx + 1:])
		tweets += " " + tweet
	return user, tweets

def cleanTweet(dirtyTweet):
	# Lowercase all of tweet
	dirtyTweet = dirtyTweet.lower()

	# Remove urls
	dirtyTweet = re.sub(r'(https?://|www\.)\S*', '', dirtyTweet)

	# Remove punctuation
	if type(dirtyTweet) is unicode:
		translate_table = dict((ord(char), u' ') for char in string.punctuation)
		dirtyTweet = dirtyTweet.translate(translate_table)
	else:
		replacePunctuation = string.maketrans(
			string.punctuation, ' ' * len(string.punctuation))
		dirtyTweet = dirtyTweet.translate(replacePunctuation)

	words = dirtyTweet.split()

	# Remove stopwords
	words = [word for word in words if word not in cachedStopWords]

	return ' '.join(words)

# Collecting all of the friends and tweets in each run
corpus = []
users = []
for root, dirs, files in os.walk('1'):
	for name in files:
		if 'tweets.rss.gz' in name:
			filePath = os.path.join(root, name)

			# reads gzipped xml files and extracts tweets
			user, tweets = gzParse(filePath)
			if user is not None:
				users.append(user)
				corpus.append(tweets)


# Writing the corpus into a file
corpusFile = open('corpus.txt', 'w')
corpusString = '\n'.join(corpus)

# Ran into some encoding issues, needed to encode everything into utf-8 before writing
corpusString = corpusString.encode('utf-8').strip()
corpusFile.write(corpusString)
corpusFile.close()

# Writing the users into a file
usersString = '\n'.join(users)
with open('users.txt', 'w') as usersFile:
	usersFile.write(usersString)