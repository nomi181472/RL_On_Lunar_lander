import random
import numpy as np
from pytorch_lightning import LightningModule
from NeuralNetwork import DQN
from ExperienceBuffer import ReplayBuffer
import heapq
from Dataset import RLDataset
from Policies import epsilon_greedy
import copy
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader


class GA:
    def __init__(self, hidden_layer, obj_size, n_actions, population_size=10, cross_over_rate=0.1,
                 mutate_rate=0.1,

                 ):
        self.obj_size = obj_size
        self.hidden_layer = hidden_layer
        self.n_actions = n_actions

        self.population_size = population_size

        self.cross_over_rate = cross_over_rate
        self.mutate_rate = mutate_rate
        self.populations = []
        self.population_performances = []
        self.generate_initial_population()

    def get_flatten_list(self, index):
        flattened_weights = []
        model = self.populations[index]
        for layer in model.net.children():
            if isinstance(layer, torch.nn.Linear):
                weights = layer.weight.view(-1).tolist()  # Flatten the weights and convert to a list
                flattened_weights.extend(weights)
        return flattened_weights

    def crossover(self, parent1, parent2):
        child = []
        for i in range(len(parent1)):
            if i % 2 == 0:
                child.append(parent2[i])
            else:
                child.append(parent1[i])
        return child

    def mutate(self, child):
        for i in range(int(len(child) * self.mutate_rate)):
            index = np.random.randint(0, len(child) - 1)
            child[index] = np.random.normal()
        return child

    def mating(self, ):
        # Get the top 10 maximum numbers with indices using a heap
        selection = 4
        top_max = heapq.nlargest(selection, enumerate(self.population_performances), key=lambda x: x[1])
        top_indices = [index for index, value in top_max]
        for i in range(selection / 2):
            # Todo selection
            first_parent_index = random.choice(top_indices)
            top_indices.remove(first_parent_index)
            second_parent_index = random.choice(top_indices)
            top_indices.remove(second_parent_index)
            # Todo flatten
            first_parent = self.get_flatten_list(first_parent_index)
            second_parent = self.get_flatten_list(second_parent_index)
            # Todo Crossover
            child = self.crossover(first_parent, second_parent)
            # Todo Mutation
            child = self.mutate(child)
            # Todo child expected score
            child_expected_score = (self.population_performances[first_parent_index] + self.population_performances[
                second_parent_index]) / 2
            # Todo replace worst solution
            worst_solution = min(self.population_performances)
            worst_solution_index = self.population_performances.index(worst_solution)
            self.population_performances[worst_solution_index] = child_expected_score
            model=self.populations[worst_solution_index]
            #Todo back to list


    def generate_initial_population(self):
        for i in range(self.population_size):
            self.populations.append(DQN(self.hidden_layer, self.obj_size, self.n_actions))  # qtable)
            # model = DQN(self.hidden_layer, self.obj_size, self.n_actions)
            # flattened_weights = []
            # # for layer in model.net.children():
            # #     if isinstance(layer, torch.nn.Linear):
            # #         weights = layer.weight.view(-1).tolist()  # Flatten the weights and convert to a list
            # #         flattened_weights.extend(weights)
            # # self.populations.append(flattened_weights)
            # for idx, layer in enumerate(model.net.children()):
            #     if isinstance(layer, torch.nn.Linear):
            #         input_size = layer.in_features
            #         output_size = layer.out_features
            #         print(f"Layer {idx + 1}: Input Size: {input_size}, Output Size: {output_size}")


class DeepQLearning(LightningModule):
    # initialize
    def __init__(self,
                 env, policy=epsilon_greedy, capacity=100_000,
                 batch_size=256, lr=1e-3, hidden_layer=128, gamma=0.99,
                 loss_fn=F.smooth_l1_loss, optim=AdamW, eps_start=0,
                 eps_end=0.15, eps_last_episode=100, sample_per_epoch=10_000, sync_rate=100,
                 max_iters=700

                 ):
        super().__init__()

        self.env = env
        self.max_iters = max_iters
        obj_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.GA = GA(hidden_layer, obj_size, n_actions)
        self.q_net = self.get_max_performer()
        # self.q_net = DQN(hidden_layer, obj_size, n_actions)  # qtable

        self.target_q_net = copy.deepcopy(self.q_net)  # backup
        self.policy = policy
        self.buffer = ReplayBuffer(capacity=capacity)
        self.save_hyperparameters()
        while len(self.buffer) < self.hparams.sample_per_epoch:
            print(f"{len(self.buffer)} sampes in experieces buffer filling.... ")
            self.play_episode(epsilon=self.hparams.eps_start)

    @torch.no_grad()
    def play_episode(self, policy=None, epsilon=0.):
        # print("play_episode")
        state = self.env.reset()
        done = False
        action = 2

        while not done:

            if policy:
                action = policy(state, self.env, self.q_net, epsilon=epsilon)
                # print(f"policy available, action: {action} {policy}")
            else:
                action = self.env.action_space.sample()
                # print(f"policy available, action: {action}")
            next_state, reward, done, info = self.env.step(action)
            exp = (state, action, reward, done, next_state)
            self.buffer.append(exp)
            state = next_state
        # print("play_episode-done")

    @torch.no_grad()
    def get_max_performer(self):
        index = 0
        highest_reward = float('-inf')
        for idx, solution in enumerate(self.GA.populations):
            solution.train()
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            total_reward = 0
            for i in range(self.max_iters):
                # Compute prediction error

                pred = solution(state)
                action = torch.argmax(pred).item()

                next_state, reward, done, _, info = self.env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)
                total_reward = total_reward + reward
                state = next_state
                if done:
                    break
            if total_reward > highest_reward:
                highest_reward = total_reward
                index = idx
            self.GA.population_performances.append(total_reward)
        return self.GA.populations[index]

    # forward
    def forward(self, x):
        # print("forward")
        return self.q_net(x);

    # configure Optimizer
    def configure_optimizers(self):
        # print("configure_optimizers")
        q_net_optimizer = self.hparams.optim(self.q_net.parameters(), lr=self.hparams.lr)
        # print("configure_optimizers-done")
        return [q_net_optimizer]

    def train_dataloader(self):
        # print("train_dataloader")
        dataset = RLDataset(self.buffer, self.hparams.sample_per_epoch);
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size
        )
        # print("train_dataloader-done")
        return dataloader

    # training steps
    def training_step(self, batch, batch_ids):
        # print("training_step")

        states, actions, rewards, dones, next_states = batch
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        state_action_values = self.q_net(states).gather(1, actions)

        next_action_values, _ = self.target_q_net(next_states).max(dim=1, keepdim=True)
        next_action_values[dones] = 0.0

        expected_state_action_values = rewards + self.hparams.gamma * next_action_values

        loss = self.hparams.loss_fn(state_action_values, expected_state_action_values)
        self.log("episode/Q-error", loss)
        # print("training_step-done")
        return loss

    # training epoch end
    def on_train_epoch_end(self):
        # print("on_train_epoch_end")

        epsilon = max(self.hparams.eps_end, self.hparams.eps_start - self.current_epoch / self.hparams.eps_last_episode)
        self.play_episode(policy=self.policy, epsilon=epsilon)
        self.log("episode/return", self.env.return_queue[-1])

        if self.current_epoch % self.hparams.sync_rate == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.GA.mating()

        # print("on_train_epoch_end-done")

# TODO:1 Add initial Populations -Collections Weights (done)
# Todo:2.1 Test On Each Populations And Maintain the list (done)
# Todo:2.2 complete the epoch till the end (done)
# TODO:2.3 Flatten each population  and CanCatenate
# Todo:3 Select Parents Which make highest performance (greedy)
# Todo:4 Crossover of Parents
# Todo:5 Mutate One Child
# Todo:6 Back to add to original Weights
# Todo:7
