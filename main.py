import gym
import torch
from es import EvolutionStrategy
from model import Model

#Hyperparams
ENV_NAME = "CartPole-v1"
POPULATION_SIZE = 200
ALPHA = 0.05
SIGMA = 0.1
NUM_GENERATIONS = 100
GAMMA = 0.99

def main():
    #make the environment
    env = gym.make(ENV_NAME)

    #declaring model and ES object
    model = Model()
    es = EvolutionStrategy(env, model, population_size=POPULATION_SIZE, alpha=ALPHA, sigma=SIGMA, gamma=GAMMA)

    #for each generation
    for generation in range(NUM_GENERATIONS):
        #show how well the model is doing in its current state
        es.run_act(True, True)
        #train for one generation
        es.train(1)

if __name__=='__main__':
    main()