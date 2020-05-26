import numpy as np
import holdem_calc
import pickle

def saveDict(dictionary, filename):
	with open(filename + '.pkl', 'wb') as f:
		pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)

ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']

suits = ['s', 'h', 'd', 'c']

count = 0

equities = {}

for firstrank_player1 in range(len(ranks)):
	for secondrank_player1 in range(firstrank_player1, len(ranks)):
		for firstrank_player2 in range(len(ranks)):
			for secondrank_player2 in range(firstrank_player2, len(ranks)):
				# You can't have a suited pair
				if(firstrank_player1 != secondrank_player1):
					player1_suitcombos = ['s', 'u']
				else:
					player1_suitcombos = ['u']
				if(firstrank_player2 != secondrank_player2):
					player2_suitcombos = ['s', 'u']
				else:
					player2_suitcombos = ['u']

				for player1_suitcombo in range(len(player1_suitcombos)):
					for player2_suitcombo in range(len(player2_suitcombos)):
						if(player1_suitcombos[player1_suitcombo] == 's'):
							firstsuit_player1 = 's'
							secondsuit_player1 = 's'
						else:
							firstsuit_player1 = 's'
							secondsuit_player1 = 'h'
						if(player2_suitcombos[player2_suitcombo] == 's'):
							firstsuit_player2 = 'd'
							secondsuit_player2 = 'd'
						else:
							firstsuit_player2 = 'd'
							secondsuit_player2 = 'c'
						#print(ranks[firstrank_player1] + firstsuit_player1 + ranks[secondrank_player1] + secondsuit_player1 + ' ' + ranks[firstrank_player2] + firstsuit_player2 + ranks[secondrank_player2] + secondsuit_player2)
						count = count + 1
						probs = holdem_calc.calculate(None, False, 10000, None, [ranks[firstrank_player1]+firstsuit_player1, ranks[secondrank_player1]+secondsuit_player1, ranks[firstrank_player2]+firstsuit_player2, ranks[secondrank_player2] + secondsuit_player2], False)
						equities[ranks[firstrank_player1] + ranks[secondrank_player1] + player1_suitcombos[player1_suitcombo] + ' ' + ranks[firstrank_player2] + ranks[secondrank_player2] + player2_suitcombos[player2_suitcombo]] = ((probs[0]/2) + probs[1])
						if(count % 100 == 0):
							print('{} combos processed'.format(count))

saveDict(equities, 'equities')