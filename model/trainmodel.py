import random
import time

import numpy as np
from keras.optimizers import Adam

from game.game import ACTIONS
from model.model_params import FINAL_EPS, OBSERVATION, LEARNING_RATE, STARTING_EPS, EXPLORE, BATCH, GAMMA
from paths import basepath, scores_file_path, actions_file_path
from utils import load_pickle, save_pickle, actions_df, scores_df


def trainNetwork(model, game_state, maxgames, observe=False, verbose=False):
    last_time = time.time()
    # store the previous observations in replay memory
    D = load_pickle("D")  # load from file system
    # get the first state by doing nothing
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1  # 0 => do nothing,
    # 1=> jump

    x_t, r_0, terminal = game_state.get_state(do_nothing, scores_df)  # get next step after performing the action

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # stack 4 images to create placeholder input

    s_t = s_t.reshape((1, s_t.shape[0], s_t.shape[1], s_t.shape[2]))  # 1*20*40*4

    initial_state = s_t

    if observe:
        OBSERVE = np.inf  # We keep observing and never train
        epsilon = FINAL_EPS
    else:  # Training Mode
        OBSERVE = OBSERVATION
        epsilon = load_pickle("epsilon")

    print("Loading weights...")
    model.load_weights(basepath + "/model/model.h5")
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    print("Weights loaded successfully")

    t = load_pickle("time")  # resume from the previous time step stored in file system

    games = 0  # how many games to play
    while games < maxgames:  # endless running

        loss = 0
        a_t = np.zeros([ACTIONS])  # action at t

        # choose an action epsilon greedy
        if random.random() <= epsilon:  # randomly explore an action
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:  # predict the output
            q = model.predict(s_t)  # input a stack of 4 images, get the prediction
            max_Q = np.argmax(q)  # choose index with maximum q value
            action_index = max_Q
            a_t[action_index] = 1  # 0=> do nothing, 1=> jump

        # We reduce the epsilon (exploration parameter) gradually
        if epsilon > FINAL_EPS and t > OBSERVE:
            epsilon -= (STARTING_EPS - FINAL_EPS) / EXPLORE

        # run the selected action and observed next state and reward
        x_t1, r_t, terminal = game_state.get_state(a_t)
        if verbose:
            print(f'FPS {1 / (time.time() - last_time)}')  # MEASURE FRAME RATE
        last_time = time.time()
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1 x 20 x 40 x 1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)  # remove oldest image and append new image to the input stack

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        # only train if done observing
        if t > OBSERVE:

            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32 x 20 x 40 x 4
            targets = np.zeros((inputs.shape[0], ACTIONS))  # 32 x 2

            # Now we do the experience replay
            for i in range(len(minibatch)):
                state_t = minibatch[i][0]  # 4D stack of images
                action_t = minibatch[i][1]  # This is action index
                reward_t = minibatch[i][2]  # reward at state_t due to action_t
                state_t1 = minibatch[i][3]  # next state
                terminal = minibatch[i][4]  # whether the agent died

                inputs[i:i + 1] = state_t

                targets[i] = model.predict(state_t)  # predicted q values
                Q_sa = model.predict(state_t1)  # predict q values for next step

                if terminal:
                    targets[i, action_t] = reward_t  # if the state is terminal the Q value equals the reward
                    games += 1
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            # PERFORM A SINGLE STEP OF GRADIENT DESCEND (ADAM) ON MINIBATCH
            loss += model.train_on_batch(inputs, targets)

        s_t = initial_state if terminal else s_t1  # reset game to initial frame if terminate
        t = t + 1

        # save progress every 1000 iterations: we do not want to lose progress
        if t % 1000 == 0:
            print("Now we save model")
            game_state._game.pause()  # pause game while saving to filesystem
            model.save_weights(basepath + "/model/model.h5", overwrite=True)
            save_pickle(D, "D")  # saving episodes
            save_pickle(t, "time")  # caching time steps
            save_pickle(epsilon, "epsilon")  # cache epsilon to avoid repeated randomness in actions
            scores_df.to_csv(scores_file_path, index=False)
            actions_df.to_csv(actions_file_path, index=False)
            game_state._game.resume()
        # print info
        if t <= OBSERVE:
            state = "observe"
        elif OBSERVE < t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if verbose:
            print(f"Timestep {t} / STATE {state} / EPSILON {epsilon} / REWARD {r_t}")
