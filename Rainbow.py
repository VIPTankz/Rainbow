import time
import gymnasium as gym
import argparse
import multiprocessing as mp
import os
import torch
import torch as T
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from functools import partial
from torch import nn as nn, Tensor
from torch.nn import init
from math import sqrt
import math

############################################## Networks Section

class FactorizedNoisyLinear(nn.Module):
    """ The factorized Gaussian noise layer for noisy-nets dqn. """
    def __init__(self, in_features: int, out_features: int, sigma_0=0.5, self_norm=False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        # weight: w = \mu^w + \sigma^w . \epsilon^w
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        # bias: b = \mu^b + \sigma^b . \epsilon^b
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        if self_norm:
            self.reset_parameters_self_norm()
        else:
            self.reset_parameters()
        self.reset_noise()

        self.disable_noise()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # initialization is similar to Kaiming uniform (He. initialization) with fan_mode=fan_in
        scale = 1 / sqrt(self.in_features)

        init.uniform_(self.weight_mu, -scale, scale)
        init.uniform_(self.bias_mu, -scale, scale)

        init.constant_(self.weight_sigma, self.sigma_0 * scale)
        init.constant_(self.bias_sigma, self.sigma_0 * scale)

    @torch.no_grad()
    def reset_parameters_self_norm(self) -> None:
        # initialization is similar to Kaiming uniform (He. initialization) with fan_mode=fan_in

        nn.init.normal_(self.weight_mu, std=1 / math.sqrt(self.out_features))
        if self.bias_mu is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)

    @torch.no_grad()
    def _get_noise(self, size: int) -> Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        # f(x) = sgn(x)sqrt(|x|)
        return noise.sign().mul_(noise.abs().sqrt_())

    @torch.no_grad()
    def reset_noise(self) -> None:
        # like in eq 10 and 11 of the paper
        epsilon_in = self._get_noise(self.in_features)
        epsilon_out = self._get_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @torch.no_grad()
    def disable_noise(self) -> None:
        self.weight_epsilon[:] = 0
        self.bias_epsilon[:] = 0

    def forward(self, input: Tensor) -> Tensor:
        # y = wx + d, where
        # w = \mu^w + \sigma^w * \epsilon^w
        # b = \mu^b + \sigma^b * \epsilon^b
        return F.linear(input,
                        self.weight_mu + self.weight_sigma*self.weight_epsilon,
                        self.bias_mu + self.bias_sigma*self.bias_epsilon)


class NatureC51(nn.Module):
    """
    Implementation of the Nature CNN, with the Categorical heads used for C51.
    """
    def __init__(self, in_depth, actions, atoms=51, Vmin=-10, Vmax=10, device='cuda:0', linear_size=512):
        super().__init__()

        self.actions = actions
        self.atoms = atoms
        self.device = device
        self.linear_size = linear_size

        DELTA_Z = (Vmax - Vmin) / (atoms - 1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_depth, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = 3136

        # Noisy Linear Layers, with both value and advantage functions for dueling DQN
        self.fc1V = FactorizedNoisyLinear(conv_out_size, linear_size)
        self.fc1A = FactorizedNoisyLinear(conv_out_size, linear_size)
        self.fcV2 = FactorizedNoisyLinear(linear_size, self.atoms)
        self.fcA2 = FactorizedNoisyLinear(linear_size, actions * self.atoms)

        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

        self.to(device)

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def fc_val(self, x):
        x = F.relu(self.fc1V(x))
        x = self.fcV2(x)

        return x

    def fc_adv(self, x):
        x = F.relu(self.fc1A(x))
        x = self.fcA2(x)

        return x

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 255
        conv_out = self.conv(fx)

        conv_out = conv_out.view(batch_size, -1)

        val_out = self.fc_val(conv_out).view(batch_size, 1, self.atoms)
        adv_out = self.fc_adv(conv_out).view(batch_size, -1, self.atoms)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + (adv_out - adv_mean)

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x, advantages_only=False):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, self.atoms)).view(t.size())

    def save_checkpoint(self, name):
        #print('... saving checkpoint ...')
        torch.save(self.state_dict(), name + ".model")

    def load_checkpoint(self, name):
        #print('... loading checkpoint ...')
        self.load_state_dict(torch.load(name))

################# Now Entering the Prioritized Experience Replay Section

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree():
	def __init__(self, size, procgen=False):
		self.index = 0
		self.size = size
		self.full = False  # Used to track actual capacity
		self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
		self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
		self.max = 1  # Initial max value to return (1 = 1^ω)

	# Updates nodes values from current tree
	def _update_nodes(self, indices):
		children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
		self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

	# Propagates changes up tree given tree indices
	def _propagate(self, indices):
		parents = (indices - 1) // 2
		unique_parents = np.unique(parents)
		self._update_nodes(unique_parents)
		if parents[0] != 0:
			self._propagate(parents)

	# Propagates single value up tree given a tree index for efficiency
	def _propagate_index(self, index):
		parent = (index - 1) // 2
		left, right = 2 * parent + 1, 2 * parent + 2
		self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
		if parent != 0:
			self._propagate_index(parent)

	# Updates values given tree indices
	def update(self, indices, values):
		self.sum_tree[indices] = values  # Set new values
		self._propagate(indices)  # Propagate values
		current_max_value = np.max(values)
		self.max = max(current_max_value, self.max)

	# Updates single value given a tree index for efficiency
	def _update_index(self, index, value):
		self.sum_tree[index] = value  # Set new value
		self._propagate_index(index)  # Propagate value
		self.max = max(value, self.max)

	def append(self, value):
		self._update_index(self.index + self.tree_start, value)  # Update tree
		self.index = (self.index + 1) % self.size  # Update index
		self.full = self.full or self.index == 0  # Save when capacity reached
		self.max = max(value, self.max)

	# Searches for the location of values in sum tree
	def _retrieve(self, indices, values):
		children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
		# If indices correspond to leaf nodes, return them
		if children_indices[0, 0] >= self.sum_tree.shape[0]:
			return indices
		# If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
		elif children_indices[0, 0] >= self.tree_start:
			children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
		left_children_values = self.sum_tree[children_indices[0]]
		successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
		successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
		successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
		return self._retrieve(successor_indices, successor_values)

	# Searches for values in sum tree and returns values, data indices and tree indices
	def find(self, values):
		indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
		data_index = indices - self.tree_start
		return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

	def total(self):
		return self.sum_tree[0]

class PER:
    def __init__(self, size, device, n, envs, gamma, alpha=0.5, beta=0.4, framestack=4, imagex=84, imagey=84, rgb=False):

        self.st = SumTree(size)
        self.data = [None for _ in range(size)]
        self.index = 0
        self.size = size

        # this is the number of frames, not the number of transitions
        # the technical size to ensure there are errors with overwritten memory in theory is very high-
        # (2*framestack - overlap) * first_states + non_first_states
        # with N=3, framestack=4, size=1M, average ep length 20, we need a total frame storage of around 1.35M
        # this however is still pretty light given it uses discrete memory. Careful when using RGB though, as we don't need as much memory
        if rgb:
            self.storage_size = int(size * 4)
        else:
            self.storage_size = int(size * 1.25)
        self.gamma = gamma
        self.capacity = 0

        self.point_mem_idx = 0

        self.state_mem_idx = 0
        self.reward_mem_idx = 0

        self.imagex = imagex
        self.imagey = imagey

        self.max_prio = 1

        self.framestack = framestack

        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-6  # small constant to stop 0 probability
        self.device = device

        self.last_terminal = [True for i in range(envs)]
        self.tstep_counter = [0 for i in range(envs)]

        self.n_step = n
        self.state_buffer = [[] for i in range(envs)]
        self.reward_buffer = [[] for i in range(envs)]

        if rgb:
            self.state_mem = np.zeros((self.storage_size, 3, self.imagex, self.imagey), dtype=np.uint8)
        else:
            self.state_mem = np.zeros((self.storage_size, self.imagex, self.imagey), dtype=np.uint8)
        self.action_mem = np.zeros(self.storage_size, dtype=np.int64)
        self.reward_mem = np.zeros(self.storage_size, dtype=float)
        self.done_mem = np.zeros(self.storage_size, dtype=bool)
        self.trun_mem = np.zeros(self.storage_size, dtype=bool)

        # everything here is stored as ints as they are just pointers to the actual memory
        # reward contains N values. The first value contains the action. The set of N contains the pointers for both
        # the reward and dones
        self.trans_dtype = np.dtype([('state', int, self.framestack), ('n_state', int, self.framestack),
                                     ('reward', int, self.n_step)])

        self.blank_trans = (np.zeros(self.framestack, dtype=int), np.zeros(self.framestack, dtype=int),
                            np.zeros(self.n_step, dtype=int))

        self.pointer_mem = np.array([self.blank_trans] * size, dtype=self.trans_dtype)

        self.overlap = self.framestack - self.n_step

    def append(self, state, action, reward, n_state, done, trun, stream, prio=True):

        # append to memory
        self.append_memory(state, action, reward, n_state, done, trun, stream)

        # append to pointer
        self.append_pointer(stream, prio)

        if done or trun:
            self.finalize_experiences(stream)
            self.state_buffer[stream] = []
            self.reward_buffer[stream] = []

        self.last_terminal[stream] = done or trun

    def append_pointer(self, stream, prio):

        while len(self.state_buffer[stream]) >= self.framestack + self.n_step and len(self.reward_buffer[stream]) >= self.n_step:
            # First array in the experience
            state_array = self.state_buffer[stream][:self.framestack]

            # Second array in the experience (starts after N frames)
            n_state_array = self.state_buffer[stream][self.n_step:self.n_step + self.framestack]

            # Reward array (first N rewards)
            reward_array = self.reward_buffer[stream][:self.n_step]

            # Add the experience to the list
            self.pointer_mem[self.point_mem_idx] = (np.array(state_array, dtype=int), np.array(n_state_array, dtype=int),
                                                             np.array(reward_array, dtype=int))

            # update the sumtree with the priority
            self.st.append(self.max_prio ** self.alpha)

            self.capacity = min(self.size, self.capacity + 1)
            self.point_mem_idx = (self.point_mem_idx + 1) % self.size

            # Remove the first state and reward from the buffers to slide the window
            self.state_buffer[stream].pop(0)
            self.reward_buffer[stream].pop(0)

    def finalize_experiences(self, stream):
        # Process remaining states and rewards at the end of an episode
        while len(self.state_buffer[stream]) >= self.framestack and len(self.reward_buffer[stream]) > 0:
            # First array in the experience
            first_array = self.state_buffer[stream][:self.framestack]

            # Second array in the experience (Final `framestack` elements)
            second_array = self.state_buffer[stream][-self.framestack:]

            # Reward array

            reward_array = self.reward_buffer[stream][:]
            while len(reward_array) < self.n_step:
                reward_array.extend([0])

            # Add the experience
            try:
                self.pointer_mem[self.point_mem_idx] = (np.array(first_array, dtype=int), np.array(second_array, dtype=int),
                                                                 np.array(reward_array, dtype=int))
            except Exception as e:
                print(f"Input Array 0: {np.array(first_array, dtype=int).shape}")
                print(f"Input Array 1: {np.array(second_array, dtype=int).shape}")
                print(f"Input Array 2: {np.array(reward_array, dtype=int).shape}")
                print(f"Error Message: {e}")
                raise Exception("Error in finalize experience")

            # update the sumtree with the priority
            self.st.append(self.max_prio ** self.alpha)

            self.point_mem_idx = (self.point_mem_idx + 1) % self.size
            self.capacity = min(self.size, self.capacity + 1)

            # Remove the first state and reward from the buffers to slide the window
            self.state_buffer[stream].pop(0)
            if len(self.reward_buffer[stream]) > 0:
                self.reward_buffer[stream].pop(0)

    def append_memory(self, state, action, reward, n_state, done, trun, stream):

        if self.last_terminal[stream]:
            # add full transition
            for i in range(self.framestack):
                self.state_mem[self.state_mem_idx] = state[i]
                self.state_buffer[stream].append(self.state_mem_idx)
                self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            # remember n_step is not applied in this memory
            self.state_mem[self.state_mem_idx] = n_state[self.framestack - 1]
            self.state_buffer[stream].append(self.state_mem_idx)
            self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            self.action_mem[self.reward_mem_idx] = action
            self.reward_mem[self.reward_mem_idx] = reward
            self.done_mem[self.reward_mem_idx] = done
            self.trun_mem[self.reward_mem_idx] = trun

            self.reward_buffer[stream].append(self.reward_mem_idx)
            self.reward_mem_idx = (self.reward_mem_idx + 1) % self.storage_size

            self.tstep_counter[stream] = 0

        else:
            # just add relevant info
            self.state_mem[self.state_mem_idx] = n_state[self.framestack - 1]
            self.state_buffer[stream].append(self.state_mem_idx)
            self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            self.action_mem[self.reward_mem_idx] = action
            self.reward_mem[self.reward_mem_idx] = reward
            self.done_mem[self.reward_mem_idx] = done
            self.trun_mem[self.reward_mem_idx] = trun

            self.reward_buffer[stream].append(self.reward_mem_idx)
            self.reward_mem_idx = (self.reward_mem_idx + 1) % self.storage_size

    def sample(self, batch_size):

        # get total sumtree priority
        p_total = self.st.total()

        # first use sumtree prios to get the indices
        segment_length = p_total / batch_size
        segment_starts = np.arange(batch_size) * segment_length
        try:
            samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts
        except Exception as e:
            print(segment_length)
            print(segment_starts)
            print(e)
            raise Exception("Stop")

        prios, idxs, tree_idxs = self.st.find(samples)

        probs = prios / p_total

        # fetch the pointers by using indices
        pointers = self.pointer_mem[idxs]

        # Extract the pointers into separate arrays
        state_pointers = np.array([p[0] for p in pointers])
        n_state_pointers = np.array([p[1] for p in pointers])
        reward_pointers = np.array([p[2] for p in pointers])
        if self.n_step > 1:
            action_pointers = np.array([p[2][0] for p in pointers])
        else:
            action_pointers = np.array([p[2] for p in pointers])

        # get state info
        states = torch.tensor(self.state_mem[state_pointers], dtype=torch.uint8)
        n_states = torch.tensor(self.state_mem[n_state_pointers], dtype=torch.uint8)

        # reward and dones just use the same pointer. actions just use the first one
        rewards = self.reward_mem[reward_pointers]
        dones = self.done_mem[reward_pointers]
        truns = self.trun_mem[reward_pointers]
        actions = self.action_mem[action_pointers]

        # apply n_step cumulation to rewards and dones
        if self.n_step > 1:
            rewards, dones = self.compute_discounted_rewards_batch(rewards, dones, truns)

        # Compute importance-sampling weights w
        weights = (self.capacity * probs) ** -self.beta

        weights = torch.tensor(weights / weights.max(), dtype=torch.float32,
                               device=self.device)  # Normalise by max importance-sampling weight from batch

        if torch.isnan(weights).any():
            print("Nan Found is sample!")
            print(f"Prios {prios}")
            print(f"Probs {probs}")
            print(f"Weights {weights}")

        # move to pytorch GPU tensors
        states = states.to(torch.float32).to(self.device)
        n_states = n_states.to(torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)

        # return batch
        return tree_idxs, states, actions, rewards, n_states, dones, weights

    def compute_discounted_rewards_batch(self, rewards_batch, dones_batch, truns_batch):
        """
        Compute discounted rewards for a batch of rewards and dones.

        Parameters:
        rewards_batch (np.ndarray): 2D array of rewards with shape (batch_size, n_step)
        dones_batch (np.ndarray): 2D array of dones with shape (batch_size, n_step)

        Returns:
        np.ndarray: 1D array of discounted rewards for each element in the batch
        np.ndarray: 1D array of cumulative dones (True if any done is True in the sequence)
        """
        batch_size, n_step = rewards_batch.shape
        discounted_rewards = np.zeros(batch_size)
        cumulative_dones = np.zeros(batch_size, dtype=bool)

        for i in range(batch_size):
            cumulative_discount = 1
            for j in range(n_step):
                discounted_rewards[i] += cumulative_discount * rewards_batch[i, j]
                if dones_batch[i, j] == 1:
                    cumulative_dones[i] = True
                    break
                elif truns_batch[i, j] == 1:
                    break
                cumulative_discount *= self.gamma

        return discounted_rewards, cumulative_dones

    def update_priorities(self, idxs, priorities):
        priorities = priorities + self.eps

        if np.isnan(priorities).any():
            print("NaN found in priority!")
            print(f"priorities: {priorities}")

        self.max_prio = max(self.max_prio, np.max(priorities))
        self.st.update(idxs, priorities ** self.alpha)

def choose_eval_action(observation, eval_net, device):
    with torch.no_grad():
        state = T.tensor(observation, dtype=T.float).to(device)
        qvals = eval_net.qvals(state, advantages_only=True)
        x = T.argmax(qvals, dim=1).cpu()

    return x

def create_network(input_dims, n_actions, device, linear_size):

    return NatureC51(input_dims[0], n_actions, device=device, linear_size=linear_size)


#################### The big ol agent class, be prepared

class Agent:
    def __init__(self, n_actions, input_dims, device, num_envs, agent_name, total_steps, testing=False, batch_size=32,
                 rr=1, lr=6.25e-5, target_replace=2000, discount=0.99, linear_size=512,
                 framestack=4, rgb=False, imagex=84, imagey=84, per_alpha=0.5, max_mem_size=1048576,
                 n=3, grad_clip=10, spi=16):

        self.per_alpha = per_alpha

        self.Vmin = -10
        self.Vmax = 10
        self.N_ATOMS = 51

        self.spi = spi

        self.procgen = True if input_dims[1] == 64 else False
        self.grad_clip = grad_clip

        self.n_actions = n_actions
        self.input_dims = input_dims
        self.device = device
        self.agent_name = agent_name
        self.testing = testing

        self.loading_checkpoint = False

        self.per_beta = 0.4

        self.total_steps = total_steps
        self.num_envs = num_envs

        if self.testing:
            self.min_sampling_size = 1000
        else:
            # paper: 80K frames; frameskip=4 → 20K env-steps
            self.min_sampling_size = 20000

        self.lr = lr

        self.priority_weight_increase = (1 - self.per_beta) / self.total_steps

        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.chkpt_dir = ""

        self.n = n
        self.gamma = discount
        self.batch_size = batch_size

        # 1 Million rounded to the nearest power of 2 for tree implementation
        self.max_mem_size = max_mem_size

        self.replace_target_cnt = target_replace  # This is the number of grad steps - could be a little jank
        # when changing num_envs/batch size/replay ratio

        # Best used value is 32000 frames per replace. For bs 256, this is 500. For bs 16, this is every 8000!

        self.linear_size = linear_size

        self.framestack = framestack
        self.rgb = rgb
        self.memory = PER(self.max_mem_size, device, self.n, num_envs, self.gamma, alpha=self.per_alpha,
                          beta=self.per_beta, framestack=self.framestack, rgb=self.rgb, imagex=imagex, imagey=imagey)

        self.network_creator_fn = partial(create_network, self.input_dims, self.n_actions, self.device, self.linear_size)

        self.net = self.network_creator_fn()
        self.tgt_net = self.network_creator_fn()

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, eps=1.5e-4)

        self.net.train()

        self.eval_net = None

        # disable gradients for the target network
        for param in self.tgt_net.parameters():
            param.requires_grad = False

        self.env_steps = 0
        self.grad_steps = 0

        self.eval_mode = False

    def prep_evaluation(self):
        self.eval_net = deepcopy(self.net)
        self.disable_noise(self.eval_net)

    @torch.no_grad()
    def reset_noise(self, net):
        for m in net.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.reset_noise()

    @torch.no_grad()
    def disable_noise(self, net):
        for m in net.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.disable_noise()

    def choose_action(self, observation):
        # this chooses an action for a batch. Can be used with a batch of 1 if needed though
        with T.no_grad():
            if not self.eval_mode:
                self.reset_noise(self.net)

            state = T.tensor(observation, dtype=T.float).to(self.net.device)

            qvals = self.net.qvals(state, advantages_only=True)
            x = T.argmax(qvals, dim=1).cpu()

            return x

    def store_transition(self, state, action, reward, next_state, done, trun, stream, prio=True):

        if self.rgb:
            # expand dims to create "framestack" dim, so it works with my replay buffer
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)

        self.memory.append(state, action, reward, next_state, done, trun, stream, prio=prio)

        self.env_steps += 1
        # anneal PER's beta
        self.memory.beta = min(self.memory.beta + self.priority_weight_increase, 1.0)

    def replace_target_network(self):
        self.tgt_net.load_state_dict(self.net.state_dict())

    def save_model(self):
        self.net.save_checkpoint(self.agent_name + "_" + str(int((self.env_steps // 250000))) + "M")

    def load_models(self, name):
        self.net.load_checkpoint(name)
        self.tgt_net.load_checkpoint(name)

    def learn(self):
        for i in range(self.spi):
            self.learn_call()

    def learn_call(self):

        if self.env_steps < self.min_sampling_size:
            return

        # NoisyNet: resample noise on both networks per gradient step
        self.reset_noise(self.net)
        self.reset_noise(self.tgt_net)

        if self.grad_steps % self.replace_target_cnt == 0:
            self.replace_target_network()

        idxs, states, actions, rewards, next_states, dones, weights = self.memory.sample(self.batch_size)

        # use this code to check your states are correct if applying to a custom env
        # If you apply Rainbow to a custom env and don't check your states first, you are killing both
        # trees and your own time

        # plt.imshow(states[0][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        #
        # plt.imshow(states[0][1].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        #
        # plt.imshow(states[0][2].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        #
        # plt.imshow(states[1][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        #
        # plt.imshow(states[2][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()

        self.optimizer.zero_grad()
        distr_v, qvals_v = self.net.both(states)
        state_action_values = distr_v[range(self.batch_size), actions.data]
        state_log_sm_v = F.log_softmax(state_action_values, dim=1)

        with torch.no_grad():
            # this is using Double DQN
            next_distr_v, next_qvals_v = self.tgt_net.both(next_states)
            action_distr_v, action_qvals_v = self.net.both(next_states)

            next_actions_v = action_qvals_v.max(1)[1]

            next_best_distr_v = next_distr_v[range(self.batch_size), next_actions_v.data]
            next_best_distr_v = self.tgt_net.apply_softmax(next_best_distr_v)
            next_best_distr = next_best_distr_v.data.cpu()

            proj_distr = distr_projection(next_best_distr, rewards.cpu(), dones.cpu(), self.Vmin, self.Vmax,
                                          self.N_ATOMS, self.gamma ** self.n)

            proj_distr_v = proj_distr.to(self.net.device)

        # per-sample KL (cross-entropy form); used as priority before IS-weighting
        kl_per_sample = (-state_log_sm_v * proj_distr_v).sum(dim=1)

        # update PER priorities with the raw (unweighted) KL
        self.memory.update_priorities(idxs, kl_per_sample.detach().cpu().numpy())

        weights = T.squeeze(weights).to(self.net.device)
        loss = (weights * kl_per_sample).mean()

        loss.backward()

        # this wasn't explicitly mentioned in the Rainbow DQN paper, but was used in DQN and was likely kept
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.grad_steps += 1
        if self.grad_steps % 10000 == 0:
            print("Completed " + str(self.grad_steps) + " gradient steps")

def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = T.zeros((batch_size, n_atoms), dtype=T.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = T.clamp(rewards + (Vmin + atom * delta_z) * gamma, Vmin, Vmax)
        b_j = (tz_j - Vmin) / delta_z
        l = T.floor(b_j).type(T.int64)
        u = T.ceil(b_j).type(T.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = T.clamp(rewards[dones], Vmin, Vmax)
        b_j = (tz_j - Vmin) / delta_z
        l = T.floor(b_j).type(T.int64)
        u = T.ceil(b_j).type(T.int64)
        eq_mask = u == l
        eq_dones = T.clone(dones)
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = T.clone(dones)
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr

def make_env(envs_create, game, framestack, repeat_probs, terminal_on_life_loss=True):
    return gym.vector.SyncVectorEnv([lambda: gym.wrappers.FrameStack(
        gym.wrappers.AtariPreprocessing(gym.make("ALE/" + game + "-v5", frameskip=1, repeat_action_probability=repeat_probs), terminal_on_life_loss=terminal_on_life_loss), framestack,
        lz4_compress=False) for _ in range(envs_create)])


def non_default_args(args, parser):
    result = []
    for arg in vars(args):
        user_val = getattr(args, arg)
        default_val = parser.get_default(arg)
        if user_val != default_val and default_val != "NameThisGame" and arg != "include_evals" and arg != "eval_envs"\
                and arg != "num_eval_episodes":

            result.append(f"{arg}={user_val}")
    return ', '.join(result)


def format_arguments(arg_string):
    arg_string = arg_string.replace('=', '')
    arg_string = arg_string.replace('True', '1')
    arg_string = arg_string.replace('False', '0')
    arg_string = arg_string.replace(', ', '_')
    return arg_string


def evaluate_agent(net_state_dict, network_creator, eval_envs, num_eval_episodes, agent_name, testing, game,
                   n_actions, device, index, framestack, repeat_probs):

    # paper evaluates on full episodes (life loss is NOT terminal during eval)
    eval_env = make_env(eval_envs, game, framestack, repeat_probs, terminal_on_life_loss=False)
    evals = []
    eval_episodes = 0
    eval_scores = np.array([0 for i in range(eval_envs)])
    eval_observation, eval_info = eval_env.reset()

    eval_net = network_creator()

    # move state dict to gpu - pytorch doesn't allow sharing across threads on gpu
    state_dict_gpu = {k: v.to(device) for k, v in net_state_dict.items()}

    eval_net.load_state_dict(state_dict_gpu)

    while eval_episodes < num_eval_episodes:

        eval_action = choose_eval_action(eval_observation, eval_net, device)
        eval_observation_, eval_reward, eval_done_, eval_trun_, eval_info = eval_env.step(eval_action)
        eval_done_ = np.logical_or(eval_done_, eval_trun_)

        for i in range(eval_envs):
            eval_scores[i] += eval_reward[i]
            if eval_done_[i]:
                eval_episodes += 1
                evals.append(eval_scores[i])
                eval_scores[i] = 0
                if eval_episodes >= num_eval_episodes:
                    break

        eval_observation = eval_observation_

    if not testing:
        fname = agent_name + "Evaluation.npy"
        data = np.load(fname)

        # Update the specified index in the 0th dimension
        data[index] = evals
        print("Evaluation " + str(index + 1) + "M Complete, average score:")
        print(np.mean(evals))

        # Save the updated array back to the file
        np.save(fname, data)


def main():
    parser = argparse.ArgumentParser()

    # environment setup
    parser.add_argument('--game', type=str, default="NameThisGame")

    parser.add_argument('--envs', type=int, default=64) # parallel envs
    parser.add_argument('--frames', type=int, default=200_000_000) # total frames (frames / 4 = steps due to frameskip)
    parser.add_argument('--eval_envs', type=int, default=5)

    parser.add_argument('--bs', type=int, default=32)  # Rainbow paper batch size

    parser.add_argument('--repeat', type=int, default=0)  # this is just for repeating experiments (multiple seeds)
    parser.add_argument('--include_evals', type=int, default=1)  # use the evaluation protocol where every 250k steps, we evaluate the agent

    parser.add_argument('--num_eval_episodes', type=int, default=20)
    parser.add_argument('--framestack', type=int, default=4)
    parser.add_argument('--sticky', type=int, default=1)  # sticky actions

    # agent setup
    parser.add_argument('--nstep', type=int, default=3)  # n-step Q-learning
    parser.add_argument('--lr', type=float, default=6.25e-5)  # learning rate
    parser.add_argument('--testing', type=bool, default=False)  # testing mode
    parser.add_argument('--grad_clip', type=int, default=10)  # gradient clipping - not mentioned in Rainbow DQN, but used in DQN and was likely kept

    parser.add_argument('--discount', type=float, default=0.99)  # discount factor
    parser.add_argument('--target_replace_frames', type=int, default=32_000)  # target network update frequency in frames
    parser.add_argument('--linear_size', type=int, default=512)  # linear size of the network
    parser.add_argument('--per_alpha', type=float, default=0.5)  # priority exponent for PER
    parser.add_argument('--spi', type=int, default=16)  # Samples per insert ratio (SPI). This is the same as Rainbow DQN.

    args = parser.parse_args()

    arg_string = non_default_args(args, parser)
    formatted_string = format_arguments(arg_string)
    print(formatted_string)

    game = args.game
    envs = args.envs
    bs = args.bs
    # convert target-replace period from environment frames to gradient steps.
    # frames / 4 → env-steps; / envs → main-loop iterations; * spi → gradient steps.
    c = int((args.target_replace_frames / 4) * args.spi / envs)
    lr = args.lr

    num_eval_episodes = args.num_eval_episodes
    framestack = args.framestack
    sticky = args.sticky
    repeat_probs = 0 if not sticky else 0.25

    nstep = args.nstep
    grad_clip = args.grad_clip
    discount = args.discount
    linear_size = args.linear_size
    total_steps = args.frames // 4
    per_alpha = args.per_alpha
    spi = args.spi

    lr_str = "{:e}".format(lr)
    lr_str = str(lr_str).replace(".", "").replace("0", "")
    frame_name = str(int(args.frames / 1000000)) + "M"

    include_evals = bool(args.include_evals)
    agent_name = "Rainbow_" + game + frame_name

    if len(formatted_string) > 2:
        agent_name += '_' + formatted_string

    print("Agent Name:" + str(agent_name))
    testing = args.testing

    # creates new directory for results and models
    if not testing:
        counter = 0
        while True:
            if counter == 0:
                new_dir_name = agent_name
            else:
                new_dir_name = f"{agent_name}_{counter}"
            if not os.path.exists(new_dir_name):
                break
            counter += 1
        os.mkdir(new_dir_name)
        print(f"Created directory: {new_dir_name}")
        os.chdir(new_dir_name)

    if testing:
        # goes easy on the PC when debugging
        num_envs = 8
        eval_envs = 2
        eval_every = 11580000
        num_eval_episodes = 5
        n_steps = 11560000
        bs = 64
    else:
        num_envs = envs
        eval_envs = args.eval_envs
        n_steps = total_steps
        eval_every = 200000
    next_eval = eval_every

    # create blank evaluation file — size off the actual eval cadence so we never overflow.
    # +2 covers the end-of-training eval and rounding from num_envs overshooting next_eval.
    fname = agent_name + "Evaluation.npy"
    if not testing:
        num_eval_slots = n_steps // eval_every + 2
        np.save(fname, np.zeros((num_eval_slots, num_eval_episodes)))

    print("Currently Playing Game: " + str(game))

    gpu = "0"
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    env = make_env(num_envs, game, framestack, repeat_probs)
    print(env.observation_space)
    print(env.action_space[0])
    n_actions = env.action_space[0].n

    agent = Agent(n_actions=env.action_space[0].n, input_dims=[framestack, 84, 84], device=device, num_envs=num_envs,
                  agent_name=agent_name, total_steps=n_steps, testing=testing, batch_size=bs, lr=lr,
                  target_replace=c, discount=discount, linear_size=linear_size,
                  framestack=framestack, per_alpha=per_alpha, n=nstep, grad_clip=grad_clip, spi=spi)

    scores_temp = []
    steps = 0
    last_steps = 0
    last_time = time.time()
    episodes = 0
    current_eval = 0
    scores_count = [0 for i in range(num_envs)]
    scores = []
    observation, info = env.reset()
    processes = []

    while steps < n_steps:
        steps += num_envs
        action = agent.choose_action(observation)

        # sync vector env: step then learn (no overlap possible)
        observation_, reward, done_, trun_, info = env.step(action)
        agent.learn()

        # this just tracks the score for each environment
        for i in range(num_envs):
            scores_count[i] += reward[i]
            if done_[i] or trun_[i]:
                episodes += 1
                scores.append([scores_count[i], steps])
                scores_temp.append(scores_count[i])
                scores_count[i] = 0

        # reward clipping
        reward = np.clip(reward, -1., 1.)

        # add transitions to the replay buffer
        for stream in range(num_envs):
            terminal_in_buffer = done_[stream]

            next_obs = observation_[stream] if not trun_[stream] else np.array(info["final_observation"][stream])
            agent.store_transition(observation[stream], action[stream], reward[stream], next_obs,
                                   terminal_in_buffer, trun_[stream], stream=stream)

        observation = observation_

        # print progress
        if steps % 1200 == 0 and len(scores) > 0:
            avg_score = np.mean(scores_temp[-50:])
            if episodes % 1 == 0:
                print('{} {} avg score {:.2f} total_steps {:.0f} fps {:.2f} games {}'
                      .format(agent_name, game, avg_score, steps, (steps - last_steps) / (time.time() - last_time), episodes),
                      flush=True)
                last_steps = steps
                last_time = time.time()

        # Evaluation
        if steps >= next_eval or steps >= n_steps:
            print("Evaluating")

            # Save model
            if not testing and (current_eval + 1) == 1 or (current_eval + 1) == 10 or (current_eval + 1) == 50\
                    or (current_eval + 1) == 100 or (current_eval + 1) == 150 or (current_eval + 1) == 200:
                agent.save_model()

            fname = agent_name + "Experiment.npy"
            if not testing:
                np.save(fname, np.array(scores))

            if include_evals:

                # wait for our evaluations to finish before we start the next evaluation
                for process in processes:
                    process.join()

                agent.disable_noise(agent.net)
                net_state_dict = deepcopy({k: v.cpu() for k, v in agent.net.state_dict().items()})
                network_creator = deepcopy(agent.network_creator_fn)

                # Start evaluation in a separate process
                eval_process = mp.Process(target=evaluate_agent,
                                          args=(net_state_dict, network_creator, eval_envs, num_eval_episodes, agent_name, testing, game,
                                                n_actions, device, current_eval, framestack, repeat_probs))
                eval_process.start()
                processes.append(eval_process)

            current_eval += 1

            next_eval += eval_every

    # wait for our evaluations to finish before we quit the program
    for process in processes:
        process.join()

    print("Evaluations finished, job completed successfully!")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
