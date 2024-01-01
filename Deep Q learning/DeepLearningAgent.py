from pytorch_lightning import LightningModule
from NeuralNetwork import DQN
from ExperienceBuffer import ReplayBuffer
import gym
from Dataset import RLDataset
from Policies import epsilon_greedy
import copy
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader


class DeepQLearning(LightningModule):
    # initialize
    def __init__(self,
                 env,policy=epsilon_greedy, capacity=100_000,
                 batch_size=256, lr=1e-3, hidden_layer=128, gamma=0.99,
                 loss_fn=F.smooth_l1_loss, optim=AdamW, eps_start=0,
                 eps_end=0.15, eps_last_episode=100, sample_per_epoch=10_000, sync_rate=100
                 ):
        super().__init__()
        self.env = env
        obj_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.q_net = DQN(hidden_layer, obj_size, n_actions)  #qtable
        self.target_q_net = copy.deepcopy(self.q_net) # backup
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

        while not done :

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
        # print("on_train_epoch_end-done")

