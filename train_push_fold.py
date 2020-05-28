import numpy as np
import random
import torch
import torch.optim as optim
from push_fold_models import Pusher, Caller
from push_fold_helper_functions import *

# Load equities dictionary
filename = 'equities'
equities = load_dictionary(filename)

ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
suits = ['s', 'h', 'd', 'c']

deck = []
for suit in suits:
	for rank in ranks:
		deck.append(rank + suit)

numEpochs = 500000
handsPerEpoch = 100
learning_rate = 1e-4
optim_betas = (0.9, 0.999)
smallestStackSize = 20
largestStackSize = 20

pusher = Pusher()
pusher.cuda()
pusherOptimizer = optim.Adam(pusher.parameters(), lr=learning_rate, betas=optim_betas)
caller = Caller()
caller.cuda()
callerOptimizer = optim.Adam(caller.parameters(), lr=learning_rate, betas=optim_betas)

for i in range(numEpochs):
	# Zero gradients
	pusher.zero_grad()
	caller.zero_grad()

	player1Card1 = torch.zeros(handsPerEpoch, 13)
	player1Card2 = torch.zeros(handsPerEpoch, 13)
	player1Suited = torch.zeros(handsPerEpoch, 3)
	player2Card1 = torch.zeros(handsPerEpoch, 13)
	player2Card2 = torch.zeros(handsPerEpoch, 13)
	player2Suited = torch.zeros(handsPerEpoch, 3)
	stackSizes = torch.zeros(handsPerEpoch, 1)
	handEquities = torch.zeros(handsPerEpoch, 1)

	for j in range(handsPerEpoch):
		# Draw 4 cards
		cards = random.sample(deck, 4)
		stackSize = random.uniform(smallestStackSize, largestStackSize)

		player1Hand = processHand(cards[0], cards[1], ranks)
		player2Hand = processHand(cards[2], cards[3], ranks)

		handEquity = equities[player1Hand + ' ' + player2Hand]

		player1Card1[j, :] = torch.Tensor(oneHotCard(player1Hand[0], ranks))
		player1Card2[j, :] = torch.Tensor(oneHotCard(player1Hand[1], ranks))
		player1Suited[j, :] = torch.Tensor(oneHotSuited(player1Hand[2], player1Hand[0], player1Hand[1]))
		player2Card1[j, :] = torch.Tensor(oneHotCard(player2Hand[0], ranks))
		player2Card2[j, :] = torch.Tensor(oneHotCard(player2Hand[1], ranks))
		player2Suited[j, :] = torch.Tensor(oneHotSuited(player2Hand[2], player1Hand[0], player1Hand[1]))
		stackSizes[j, 0] = stackSize
		handEquities[j, 0] = handEquity

	player1Card1 = player1Card1.type(torch.FloatTensor)
	player1Card1 = player1Card1.cuda()
	player1Card2 = player1Card2.type(torch.FloatTensor)
	player1Card2 = player1Card2.cuda()
	player1Suited = player1Suited.type(torch.FloatTensor)
	player1Suited = player1Suited.cuda()
	stackSizes = stackSizes.type(torch.FloatTensor)
	stackSizes = stackSizes.cuda()

	pushPerc = pusher(player1Card1, player1Card2, player1Suited, stackSizes)

	player2Card1 = player2Card1.type(torch.FloatTensor)
	player2Card1 = player2Card1.cuda()
	player2Card2 = player2Card2.type(torch.FloatTensor)
	player2Card2 = player2Card2.cuda()
	player2Suited = player2Suited.type(torch.FloatTensor)
	player2Suited = player2Suited.cuda()
	handEquities = handEquities.type(torch.FloatTensor)
	handEquities = handEquities.cuda()

	callPerc = caller(player2Card1, player2Card2, player2Suited, stackSizes)

	# Pusher doesn't update caller's gradients
	pushError = pusherLoss(pushPerc, callPerc.detach(), stackSizes, handEquities)

	# Caller doesn't update pusher's gradients
	callError = callerLoss(pushPerc.detach(), callPerc, stackSizes, handEquities)

	pushError.backward()
	callError.backward()

	pusherOptimizer.step()
	callerOptimizer.step()

	if(i % 100 == 0):
		print('Epoch: {} Pusher Error: {} Caller Error: {}'.format(i, pushError.item(), callError.item()))
		print('Push Perc: {} Call Perc: {}'.format(np.mean(pushPerc.cpu().detach().numpy()), np.mean(callPerc.cpu().detach().numpy())))

torch.save(pusher.state_dict(), 'pusher.pt')
torch.save(caller.state_dict(), 'caller.pt')