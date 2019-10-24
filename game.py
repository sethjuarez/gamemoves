import os
import json
import random
import numpy as np

# available moves
moves = ['rock', 'paper', 'scissors', 'lizard', 'spock']
# adjacency matrix for winner
game = [[ 0, -1,  1,  1, -1],
        [ 1,  0, -1, -1,  1],
        [-1,  1,  0,  1, -1],
        [-1,  1, -1,  0,  1],
        [ 1, -1,  1, -1,  0]]

# model data
model = {
    'hist': 5,
    'play': [],
    'pred': []
}

def init():
    for i in range(model['hist']):
        if i == 0:
            model['pred'].append({
                'count': 0,
                'probs': { }
            })
        else:
            model['pred'].append({ })

def predict():
    history = model['play']
    pred = model['pred']
    guesses = np.zeros((model['hist'], len(moves)))

    for i in range(model['hist']):
        record = pred[i]

        # guessing based on user distribution
        if i == 0:
            for k in record['probs'].keys():
                idx = moves.index(k)
                guesses[i][idx] = record['probs'][k] / record['count']
        # guess on conditional distribution
        else:
            m = history[-i:]
            # not enough history
            if len(m) < i:
                break

            item = '_'.join(m)
            if item in record:
                for k in record[item]['probs'].keys():
                    idx = moves.index(k)
                    guesses[i][idx] = record[item]['probs'][k] / record[item]['count']


    # nothing to go on so just guess
    if np.sum(guesses) == 0:
        return random.choice(moves)
    else:
        # get the highest probability slots
        probabilities = np.max(guesses, axis=0)
        max_p = np.max(probabilities)
        # may have more than one max_p
        their_moves = [idx for idx, val in enumerate(probabilities) if val == max_p]
        
        # fill out their choices
        choices = []
        for guess in their_moves:
            # find out moves that beats projected move
            choices.extend([moves[idx] for idx, val in enumerate(game[guess]) if val == -1])

        # choose one randomly
        return random.choice(choices)

def update(move):
    history = model['play']
    predictions = model['pred']

    # add to history
    history.append(move)

    # update model
    for i in range(model['hist']):
        # last moves i moves
        m = history[-(i+1):]

        # if history is insufficient
        if len(m) == i:
            break

        # get coniditionals
        key = m[:i]
        # get vals
        val = m[-1]

        pred = predictions[i]

        # update predictions
        if i == 0:
            predictions[i]['count'] += 1
            if val in pred['probs']:
                pred['probs'][val] += 1
            else:
                pred['probs'][val] = 1
        else:
            skey = '_'.join(key)
            if skey in pred:
                pred[skey]['count'] += 1
                if val in pred[skey]['probs']:
                    pred[skey]['probs'][val] += 1
                else:
                    pred[skey]['probs'][val] = 1
            else:
                pred[skey] = {
                    'count': 1,
                    'probs': { val : 1 }
                }

def winner(player1, player2):
    print(f'{player1} vs. {player2}')
    idx1, idx2 = moves.index(player1), moves.index(player2)
    w = game[idx1][idx2]
    return w

if __name__ == "__main__":
    # init model - called
    init()

    plays = ['rock', 'paper', 'scissors', 'rock', 'paper', 'scissors', 'lizard', 'rock', 'spock', 'rock', 'paper']

    # fill model
    for item in plays:
        update(item)

    print(f'Model:\n{json.dumps(model, sort_keys=True,indent=3)}\n')

    for i in range(len(plays)):
        print(f'\nRound {i+1}')
        # computer guess
        computer = predict()
        # human move
        human = plays[i]
        # who won??
        w = winner(human, computer)
        # update model
        update(human)
        print(f'{["you lose", "a tie", "you win!"][w+1]}')