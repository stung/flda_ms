import os
import string
import gzip
import sys
import re
import gc
import random

#import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords
#cachedStopWords = stopwords.words("english")

# Hardcode the stopwords from nltk to avoid module dependency
cachedStopWords = [u'me', u'my', u'myself', u'we', u'our', u'ours',
                   u'ourselves', u'you', u'your', u'yours', u'yourself',
                   u'yourselves', u'he', u'him', u'his', u'himself', u'she',
                   u'her', u'hers', u'herself', u'it', u'its', u'itself',
                   u'they', u'them', u'their', u'theirs', u'themselves',
                   u'what', u'which', u'who', u'whom', u'this', u'that',
                   u'these', u'those', u'am', u'is', u'are', u'was', u'were',
                   u'be', u'been', u'being', u'have', u'has', u'had', u'having',
                   u'do', u'does', u'did', u'doing', u'an', u'the',
                   u'and', u'but', u'if', u'or', u'because', u'as', u'until',
                   u'while', u'of', u'at', u'by', u'for', u'with', u'about',
                   u'against', u'between', u'into', u'through', u'during',
                   u'before', u'after', u'above', u'below', u'to', u'from',
                   u'up', u'down', u'in', u'out', u'on', u'off', u'over',
                   u'under', u'again', u'further', u'then', u'once', u'here',
                   u'there', u'when', u'where', u'why', u'how', u'all', u'any',
                   u'both', u'each', u'few', u'more', u'most', u'other',
                   u'some', u'such', u'no', u'nor', u'not', u'only', u'own',
                   u'same', u'so', u'than', u'too', u'very', u'can',
                   u'will', u'just', u'don', u'should', u'now']

try:
  import xml.etree.cElementTree as ET
except ImportError:
  import xml.etree.ElementTree as ET

# Gunzips and parses one XML Twitter user tweet file
# Returns one line with all of the tweets
def gzParseTweets(xmlPath):
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

# Gunzips and parses one XML Twitter user follower file
# Returns list of all the user IDs that the current user is following
def gzParseFollowers(xmlPath):
  # reads gzip file
  xmlFile = gzip.open(xmlPath)
  fileContent = xmlFile.read()

  # Converting contents into an XML Element Tree
  root = ET.fromstring(fileContent)
  userids = ""

  for uid in root.iter('id'):
    userids += " " + uid.text
  return userids

def cleanTweet(dirtyTweet):
  # Lowercase all of tweet
  dirtyTweet = dirtyTweet.lower()

  # Remove urls
  dirtyTweet = re.sub(r'(https?://|www\.)\S*', '', dirtyTweet)

  # Remove punctuation, except for '@'
  if type(dirtyTweet) is unicode:
    translate_table = dict((ord(char), u' ') for char in string.punctuation)
    translate_table[ord('@')] = u'@'
    dirtyTweet = dirtyTweet.translate(translate_table)
  else:
    noAtPunct = string.punctuation.replace("@", "")
    replacePunctuation = string.maketrans(
      noAtPunct, ' ' * len(noAtPunct))
    dirtyTweet = dirtyTweet.translate(replacePunctuation)

  words = dirtyTweet.split()

  # Remove stopwords
  words = [word for word in words if word not in cachedStopWords]

  # Remove single/double letter words
  words = [word for word in words if len(word) > 2]

  # Remove usernames
  words = [word for word in words if word[0] != '@']

  return ' '.join(words)

# Need to ensure that each person both tweets and follows people
tweeters = set()
frienders = set()
legitUsers = set()

tweetFile = 'tweets.rss.gz'
friendFile = 'friends.rss.gz'

print("Finding all tweeters and frienders")

for root, dirs, files in os.walk('.', topdown=False):
  for name in files:
    if tweetFile in name:
      filePath = os.path.join(root, name)

      tweeters.add(filePath.rpartition(tweetFile)[0])

    if friendFile in name:
      filePath = os.path.join(root, name)
      
      frienders.add(filePath.rpartition(friendFile)[0])

# Extracting the overlap between the two lists
legitUsers = tweeters & frienders

print("Deleting the tweeters and frienders")
del tweeters
del frienders
gc.collect()

# print("Randomly selecting 50 percent of users")
# random.seed()
# fractionOfUsers = []
# for user in legitUsers:
#   if (random.random() < 0.50):
#     fractionOfUsers.append(user)

print("Processing the legit users")

# Collecting all of the friends and tweets in each run
corpus = ""
users = ""
followers = ""

for userPath in legitUsers:
  # Process tweets
  filePath = os.path.join(userPath, tweetFile)
  # reads gzipped xml files and extracts tweets
  user, tweets = gzParseTweets(filePath)
  if user is None:
    user = "N/A"
  userIDPair = user + ":" + filePath.split('/')[-2]
  users += userIDPair + '\n'
  corpus += tweets + '\n'

  # Process friends
  filePath = os.path.join(userPath, friendFile)

  followerList = gzParseFollowers(filePath)
  followers += followerList + '\n'

print("Writing the corpus")

# Writing the corpus into a file
corpusFile = open('corpus.txt', 'w')
# corpusString = '\n'.join(corpus)

# Ran into some encoding issues, needed to encode everything into utf-8 before writing
corpus = corpus.encode('utf-8').strip()
corpusFile.write(corpus)
corpusFile.close()
del corpus
gc.collect()

print("Writing the users")

# Writing the users into a file
# usersString = '\n'.join(users)
with open('users.txt', 'w') as usersFile:
  usersFile.write(users)
usersFile.close()
del users
gc.collect()

print("Writing the followers")

# Writing the follower list into a file
followerFile = open('followers.txt', 'w')
# followerString = '\n'.join(followers)
followerFile.write(followers)
followerFile.close()
del followers
gc.collect()
