import os
import string
import gzip
import sys
import re

# Collecting all followers to swap
userDict = {}
userSorted = []

followerFile = open('followers.txt', 'r')
userFile = open('users.txt', 'r')
for line in userFile:
	userID = line.split(':')[-1].replace('\n', '')
	userDict[userID] = []
	userSorted.append(userID)

# Keeping track of which user we are currently on
userIndex = 0

for line in followerFile:
	line = line.replace('\n', '')
	# print line

	# Add current user to all followers in the list
	followList = line.split(' ')
	# print followList
	for person in followList:
		if person not in userDict:
			userDict[person] = []
		userDict[person].append(userSorted[userIndex])
		# print userDict[person]

	userIndex += 1
	# print userIndex

followees = []

for user in userSorted:
	friendString = ' '.join(userDict[user])
	followees.append(friendString)

finalString = '\n'.join(followees)
followeeFile = open('backFollowers.txt', 'w')
followeeFile.write(finalString)

followeeFile.close()
userFile.close()
followerFile.close()


# for root, dirs, files in os.walk('1'):
#   for name in files:
#     if 'friends.rss.gz' in name:
#       filePath = os.path.join(root, name)

#       # reads gzipped xml files and extracts followees
#       followees = gzParse(filePath)
#       users[dirs] = followees

# # Writing the corpus into a file
# followeeFile = open('followees.txt', 'w')
# for key, value in users.iteritems():

# followeeFile.close()