import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class EvolutionStrategy():
    #constructor
    def __init__(self, env, model, population_size=100, alpha=0.1, sigma=0.1, gamma=0.99):
        #Hyperparams
        self.env = env
        self.model = model.double()
        self.alpha = alpha
        self.sigma = sigma
        self.gamma = gamma
        self.gamma_curr = 1
        self.generation = 0
        self.population_size = population_size

    #run the agent in the env and return the acquired reward
    def run_act(self, printReward=False, showAct=False):
        observation = self.env.reset()
        total_reward = 0

        while(True):
            if showAct:
                self.env.render()

            observation = torch.tensor(observation)
            action = self.model(observation).detach().numpy()[0].clip(0, 1).astype(int)

            observation, reward, done, info  = self.env.step(action)

            total_reward += reward
            if done:
                break

        if printReward:
            print("Generation", self.generation, "Reward:", total_reward)

        return total_reward

    #trains(updates) the model
    def train(self, generations):
        self.generation += generations
        for generation in range(generations):
            for w in self.model.parameters():
                rewards = []
                w_size = w.size()

                w_noises = torch.randn(self.population_size, *w_size)

                w_save = w
                for p in range(0, self.population_size):
                    w.data = w_save + w_noises[p].double()
                    rewards.append(self.run_act())

                rewards = torch.tensor(rewards).reshape(1, self.population_size)
                rewards = (rewards - torch.mean(rewards))/torch.std(rewards)

                nparams = 1

                for i in w_size:
                    nparams *= i

                w_add = torch.mm(rewards, w_noises.reshape(self.population_size, nparams))
                w_add = self.alpha*w.reshape(*w_size).double()/(self.sigma*self.population_size)

                w.data = w_save + torch.clamp(w_add, -1, 1)*self.gamma_curr
            self.gamma_curr *= self.gamma
