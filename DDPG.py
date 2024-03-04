import random

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils


# 分别为空调和蓄电池的维度
state_dims = [6,4]
action_dims = [1,1]

critic_input_dim = sum(state_dims) + sum(action_dims)
class PolicyNet(torch.nn.Module):
    '''AC以及Battery都可以用策略网络，bias调整连续输出取值范围'''
    def __init__(self, state_dim,  action_dim,hidden_dim, action_bound, bias):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        nn.utils.weight_norm(self.fc1)
        nn.utils.weight_norm(self.fc2)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值
        self.bias = bias

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        y= torch.tanh(x)
        return y

class QValueNet(torch.nn.Module):
    '''价值网络，输入状态以及动作给出评分'''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(critic_input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)
        nn.utils.weight_norm(self.fc1)
        nn.utils.weight_norm(self.fc2)
        nn.utils.weight_norm(self.fc_out)

    def forward(self, x):
        #cat = torch.cat([x, a], dim=1) # 拼接状态和动作
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class DDPG:
    def __init__(self,state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device,action_bound,bias,sigma):
        self.sigma = sigma
        self.device = device
        self.action_dim = action_dim
        self.actor = PolicyNet(state_dim, action_dim, hidden_dim,action_bound,bias).to(device)
        self.target_actor = PolicyNet(state_dim, action_dim,hidden_dim, action_bound, bias).to(device)
        self.critic = QValueNet(critic_input_dim, hidden_dim, 1).to(device)
        self.target_critic = QValueNet(critic_input_dim, hidden_dim,1).to(device)


        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

    def take_action(self, state, explore=False):
        action = self.actor(state).item()
        if self.sigma > 0.01:
            self.sigma = self.sigma * 0.999

        else:
            self.sigma = 0.01
        if explore:
            action = action + self.sigma * np.random.randn(self.action_dim)
        else:
            action = action
        if action >1:
            action = np.array([1.0])
        elif action < -1:
            action = np.array([-1.0])
        return action

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)


class MADDPG:
    def __init__(self, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau,action_bound,bias,sigma):
        self.agents = []
        for i in range(2):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device,action_bound[i],bias[i],sigma))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    #@property
    def target_policies(self):    # 智能体的目标执行网络
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore):
        states = [torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(len(self.agents))]
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]


    def update(self, sample, i_agent):
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]

        cur_agent.critic_optimizer.zero_grad()
        all_target_act = []
        for pi, _next_obs in zip(self.target_policies(), next_obs):
            all_target_act.append(pi(_next_obs))
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(-1, 1) + self.gamma * cur_agent.target_critic(target_critic_input) * (1 - done[i_agent].view(-1, 1))
        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value,
                                            target_critic_value.detach())

        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[i_agent])
        cur_act_vf_in = cur_actor_out
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(pi(_obs))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out**2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)