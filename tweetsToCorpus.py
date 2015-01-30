import os
import string
import gzip
import sys
import re

#import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords
#cachedStopWords = stopwords.words("english")

# Hardcode the stopwords from nltk to avoid module dependency
cachedStopWords = [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours',
                   u'ourselves', u'you', u'your', u'yours', u'yourself',
                   u'yourselves', u'he', u'him', u'his', u'himself', u'she',
                   u'her', u'hers', u'herself', u'it', u'its', u'itself',
                   u'they', u'them', u'their', u'theirs', u'themselves',
                   u'what', u'which', u'who', u'whom', u'this', u'that',
                   u'these', u'those', u'am', u'is', u'are', u'was', u'were',
                   u'be', u'been', u'being', u'have', u'has', u'had', u'having',
                   u'do', u'does', u'did', u'doing', u'a', u'an', u'the',
                   u'and', u'but', u'if', u'or', u'because', u'as', u'until',
                   u'while', u'of', u'at', u'by', u'for', u'with', u'about',
                   u'against', u'between', u'into', u'through', u'during',
                   u'before', u'after', u'above', u'below', u'to', u'from',
                   u'up', u'down', u'in', u'out', u'on', u'off', u'over',
                   u'under', u'again', u'further', u'then', u'once', u'here',
                   u'there', u'when', u'where', u'why', u'how', u'all', u'any',
                   u'both', u'each', u'few', u'more', u'most', u'other',
                   u'some', u'such', u'no', u'nor', u'not', u'only', u'own',
                   u'same', u'so', u'than', u'too', u'very', u's', u't', u'can',
                   u'will', u'just', u'don', u'should', u'now']


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