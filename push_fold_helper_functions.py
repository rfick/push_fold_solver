import numpy as np
import torch
import pickle

def load_dictionary(filename):
	with open(filename + '.pkl', 'rb') as f:
		return pickle.load(f)

def pusherLoss(pushPerc, callPerc, stackSize, equity):
	loss = torch.mean((-1)*(-0.5*(1-pushPerc) + pushPerc*(1*(1-callPerc) + (2*stackSize*equity - stackSize)*callPerc)))
	return loss

def callerLoss(pushPerc, callPerc, stackSize, equity):
	loss = torch.mean((-1)*(0.5*(1-pushPerc) + pushPerc*(-1*(1-callPerc) + (2*stackSize*(1-equity) - stackSize)*callPerc)))
	return loss

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