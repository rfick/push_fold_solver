import torch
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from push_fold_models import Pusher, Caller

def load_dictionary(filename):
	with open(filename + '.pkl', 'rb') as f:
		return pickle.load(f)

def oneHotCard(card, ranks):
	oneHot = np.zeros((1, 13))
	oneHot[0, ranks.index(card[0])] = 1
	return oneHot

# 's' = suited, 'u' = offsuit
def oneHotSuited(suited):
	oneHot = np.zeros((1, 2))
	if(suited == 's'):
		oneHot[0, 0] = 1
	else:
		oneHot[0, 1] = 1
	return oneHot

# Cards should be rank followed by suit ('As', '7c', etc)
# Puts cards in rank order and tells if they are suited ('AAu', '98s', etc)
def processHand(card1, card2, ranks):
	hand = ''
	if(ranks.index(card1[0]) < ranks.index(card2[0])):
		hand = hand + card1[0] + card2[0]
	else:
		hand = hand + card2[0] + card1[0]
	if(card1[1] == card2[1]):
		hand = hand + 's'
	else:
		hand = hand + 'u'
	return hand

def pusherLoss(pushPerc, callPerc, stackSize, equity):
	loss = np.mean((-1)*(-0.5*(1-pushPerc) + pushPerc*(1.5*(1-callPerc) + (2*stackSize*equity - (stackSize - 0.5))*callPerc)))
	return loss

def callerLoss(pushPerc, callPerc, stackSize, equity):
	loss = np.mean((-1)*(1.5*(1-pushPerc) + pushPerc*(-1*(1-callPerc) + (2*stackSize*(1-equity) - (stackSize - 1))*callPerc)))
	return loss

# Hand is 'AKs', '87o', etc
def holdemResourcesPusher(hand, ranks):
	pushPerc = 0
	if(hand[0] == 'A'):
		pushPerc = 1
	elif(hand[0] == 'K'):
		if(hand[2] == 's'):
			# 4 or better
			if(ranks.index(hand[1]) < 11):
				pushPerc = 1
		else:
			# 9 or better
			if(ranks.index(hand[1]) < 6):
				pushPerc = 1
	elif(hand[0] == 'Q'):
		if(hand[2] == 's'):
			# 5 or better
			if(ranks.index(hand[1]) < 10):
				pushPerc = 1
		else:
			# 9 or better
			if(ranks.index(hand[1]) < 6):
				pushPerc = 1
	elif(hand[0] == 'J'):
		if(hand[2] == 's'):
			# 7 or better
			if(ranks.index(hand[1]) < 8):
				pushPerc = 1
		else:
			# 9 or better
			if(ranks.index(hand[1]) < 6):
				pushPerc = 1
	elif(hand[0] == 'T'):
		if(hand[2] == 's'):
			# 6 or better
			if(ranks.index(hand[1]) < 9):
				pushPerc = 1
		else:
			# 9 or better
			if(ranks.index(hand[1]) < 6):
				pushPerc = 1
	elif(hand[0] == '9'):
		if(hand[2] == 's'):
			# 6 or better
			if(ranks.index(hand[1]) < 9):
				pushPerc = 1
		else:
			# 8 or better
			if(ranks.index(hand[1]) < 7):
				pushPerc = 1
	elif(hand[0] == '8'):
		if(hand[2] == 's'):
			# 6 or better
			if(ranks.index(hand[1]) < 9):
				pushPerc = 1
		else:
			# 8 or better
			if(ranks.index(hand[1]) < 7):
				pushPerc = 1
	elif(hand[0] == '7'):
		if(hand[2] == 's'):
			# 5 or better
			if(ranks.index(hand[1]) < 10):
				pushPerc = 1
		else:
			# 7 or better
			if(ranks.index(hand[1]) < 8):
				pushPerc = 1
	elif(hand[0] == '6'):
		if(hand[2] == 's'):
			# 5 or better
			if(ranks.index(hand[1]) < 10):
				pushPerc = 1
		else:
			# 6 or better
			if(ranks.index(hand[1]) < 9):
				pushPerc = 1
	elif(hand[0] == '5'):
		if(hand[2] == 's'):
			# 4 or better
			if(ranks.index(hand[1]) < 11):
				pushPerc = 1
		else:
			# 5 or better
			if(ranks.index(hand[1]) < 10):
				pushPerc = 1
	elif(hand[0] == '4'):
		if(hand[2] == 'u'):
			# 4 or better
			if(ranks.index(hand[1]) < 11):
				pushPerc = 1
	elif(hand[0] == '3'):
		if(hand[2] == 'u'):
			# 3 or better
			if(ranks.index(hand[1]) < 12):
				pushPerc = 1
	elif(hand[0] == '2'):
		if(hand[2] == 'u'):
			# 2 or better
			if(ranks.index(hand[1]) < 13):
				pushPerc = 1
	return pushPerc

# Hand is 'AKs', '87o', etc
def holdemResourcesCaller(hand, ranks):
	callPerc = 0
	if(hand[0] == 'A'):
		if(hand[2] == 's'):
			# all suited aces
			callPerc = 1
		else:
			# 5 or better
			if(ranks.index(hand[1]) < 10):
				callPerc = 1
	elif(hand[0] == 'K'):
		if(hand[2] == 's'):
			# 9 or better
			if(ranks.index(hand[1]) < 6):
				callPerc = 1
		else:
			# T or better
			if(ranks.index(hand[1]) < 5):
				callPerc = 1
	elif(hand[0] == 'Q'):
		if(hand[2] == 's'):
			# T or better
			if(ranks.index(hand[1]) < 5):
				callPerc = 1
		else:
			# Q or better
			if(ranks.index(hand[1]) < 3):
				callPerc = 1
	elif(hand[0] == 'J'):
		if(hand[1] == 'J'):
			callPerc = 1
	elif(hand[0] == 'T'):
		if(hand[1] == 'T'):
			callPerc = 1
	elif(hand[0] == '9'):
		if(hand[1] == '9'):
			callPerc = 1
	elif(hand[0] == '8'):
		if(hand[1] == '8'):
			callPerc = 1
	elif(hand[0] == '7'):
		if(hand[1] == '7'):
			callPerc = 1
	elif(hand[0] == '6'):
		if(hand[1] == '6'):
			callPerc = 1
	elif(hand[0] == '5'):
		if(hand[1] == '5'):
			callPerc = 1
	elif(hand[0] == '4'):
		if(hand[1] == '4'):
			callPerc = 1
	elif(hand[0] == '3'):
		if(hand[1] == '3'):
			callPerc = 1
	return callPerc

pusher = Pusher()
pusher.load_state_dict(torch.load('pusher.pt'))
pusher.eval()

caller = Caller()
caller.load_state_dict(torch.load('caller.pt'))
caller.eval()

# Load equities dictionary
filename = 'equities'
equities = load_dictionary(filename)

ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
suits = ['s', 'h', 'd', 'c']

deck = []
for suit in suits:
	for rank in ranks:
		deck.append(rank + suit)

stackSize = 20

stackSizeTensor = torch.Tensor(1, 1)
stackSizeTensor[0, 0] = stackSize
stackSizeTensor = stackSizeTensor.type(torch.FloatTensor)

pushPercs = np.zeros((13, 13))
callPercs = np.zeros((13, 13))

for firstrank in range(len(ranks)):
	for secondrank in range(firstrank, len(ranks)):
		# You can't have a suited pair
		if(firstrank != secondrank):
			suitcombos = ['s', 'u']
		else:
			suitcombos = ['u']
		for suitcombo in range(len(suitcombos)):
			card1 = oneHotCard(ranks[firstrank], ranks)
			card2 = oneHotCard(ranks[secondrank], ranks)
			suited = oneHotSuited(suitcombos[suitcombo])

			card1 = torch.Tensor(card1)
			card1 = card1.type(torch.FloatTensor)
			card2 = torch.Tensor(card2)
			card2 = card2.type(torch.FloatTensor)
			suited = torch.Tensor(suited)
			suited = suited.type(torch.FloatTensor)

			pushPerc = pusher(card1, card2, suited, stackSizeTensor)
			callPerc = caller(card1, card2, suited, stackSizeTensor)

			# Place suited results in upper triangle, offsuit in lower triangle
			if(suitcombos[suitcombo] == 's'):
				index1 = ranks.index(ranks[firstrank])
				index2 = ranks.index(ranks[secondrank])
			else:
				index1 = ranks.index(ranks[secondrank])
				index2 = ranks.index(ranks[firstrank])

			pushPercs[index1, index2] = pushPerc
			callPercs[index1, index2] = callPerc

plt.figure(dpi=300)
plt.imshow(pushPercs, interpolation='none', vmin=0, vmax=1, aspect='equal')
ax = plt.gca()
ax.set_xticks(np.arange(-0.5, 13, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 13, 1), minor=True)
ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
plt.xticks(range(0, 13), ranks)
plt.yticks(range(0, 13), ranks)
plt.colorbar()
plt.title('20 bb Pusher')
plt.savefig('pusher{}.png'.format(stackSize))

plt.figure(dpi=300)
plt.imshow(callPercs, interpolation='none', vmin=0, vmax=1, aspect='equal')
ax = plt.gca()
ax.set_xticks(np.arange(-0.5, 13, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 13, 1), minor=True)
ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
plt.xticks(range(0, 13), ranks)
plt.yticks(range(0, 13), ranks)
plt.colorbar()
plt.title('20 bb Caller')
plt.savefig('caller{}.png'.format(stackSize))

numSims = 100000

hmrPushLoss = 0
hmrCallLoss = 0

myPushLoss = 0
myCallLoss = 0

myPushvsHmrCallLoss = 0
myCallvsHmrPushLoss = 0

stackSizeTensor = torch.Tensor(1, 1)
stackSizeTensor[0, 0] = stackSize
stackSizeTensor = stackSizeTensor.type(torch.FloatTensor)

for i in range(numSims):
	# Draw 4 cards
	cards = random.sample(deck, 4)
	hand1 = processHand(cards[0], cards[1], ranks)
	hand2 = processHand(cards[2], cards[3], ranks)

	handEquity = equities[hand1 + ' ' + hand2]

	hmrPushPerc = holdemResourcesPusher(hand1, ranks)
	hmrCallPerc = holdemResourcesCaller(hand2, ranks)

	card1 = oneHotCard(hand1[0], ranks)
	card2 = oneHotCard(hand1[1], ranks)
	suited = oneHotSuited(hand1[2])

	card1 = torch.Tensor(card1)
	card1 = card1.type(torch.FloatTensor)
	card2 = torch.Tensor(card2)
	card2 = card2.type(torch.FloatTensor)
	suited = torch.Tensor(suited)
	suited = suited.type(torch.FloatTensor)

	myPushPerc = pusher(card1, card2, suited, stackSizeTensor)

	card1 = oneHotCard(hand2[0], ranks)
	card2 = oneHotCard(hand2[1], ranks)
	suited = oneHotSuited(hand2[2])

	card1 = torch.Tensor(card1)
	card1 = card1.type(torch.FloatTensor)
	card2 = torch.Tensor(card2)
	card2 = card2.type(torch.FloatTensor)
	suited = torch.Tensor(suited)
	suited = suited.type(torch.FloatTensor)

	myCallPerc = caller(card1, card2, suited, stackSizeTensor)

	hmrPushLoss = hmrPushLoss + pusherLoss(hmrPushPerc, hmrCallPerc, stackSize, handEquity)
	hmrCallLoss = hmrCallLoss + callerLoss(hmrPushPerc, hmrCallPerc, stackSize, handEquity)

	myPushLoss = myPushLoss + pusherLoss(myPushPerc.detach().numpy(), myCallPerc.detach().numpy(), stackSize, handEquity)
	myCallLoss = myCallLoss + callerLoss(myPushPerc.detach().numpy(), myCallPerc.detach().numpy(), stackSize, handEquity)

	myPushvsHmrCallLoss = myPushvsHmrCallLoss + pusherLoss(myPushPerc.detach().numpy(), hmrCallPerc, stackSize, handEquity)
	myCallvsHmrPushLoss = myCallvsHmrPushLoss + callerLoss(hmrPushPerc, myCallPerc.detach().numpy(), stackSize, handEquity)

hmrPushLoss = hmrPushLoss/numSims
hmrCallLoss = hmrCallLoss/numSims

myPushLoss = myPushLoss/numSims
myCallLoss = myCallLoss/numSims

myPushvsHmrCallLoss = myPushvsHmrCallLoss/numSims
myCallvsHmrPushLoss = myCallvsHmrPushLoss/numSims

print('Holdem Resources pusher loss: {}'.format(hmrPushLoss))
print('Holdem Resources caller loss: {}'.format(hmrCallLoss))

print('My pusher loss: {}'.format(myPushLoss))
print('My caller loss: {}'.format(myCallLoss))

print('My pusher vs HMR caller loss: {}'.format(myPushvsHmrCallLoss))
print('My caller vs HMR pusher loss: {}'.format(myCallvsHmrPushLoss))