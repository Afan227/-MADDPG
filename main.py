import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import rl_utils
from DDPG import MADDPG
import environment
from tqdm import tqdm

data = 20240304
num_episodes = 5000
episode_length = 144  # 每条序列的最大长度
buffer_size = 10000
hidden_dim = 64
actor_lr = 8e-5
critic_lr = 8e-5

gamma = 0.95
tau = 1e-2
action_bound = [6,0]
bias = [23, 0]
sigma = 1
batch_size = 128
agent_num = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
update_interval = 100
minimal_size = 3000

replay_buffer = rl_utils.ReplayBuffer(buffer_size)

# 分别为空调和蓄电池的维度
state_dims = [6,4]
action_dims = [1,1]

critic_input_dim = sum(state_dims) + sum(action_dims)

maddpg = MADDPG(device, actor_lr, critic_lr, hidden_dim, state_dims,
                action_dims, critic_input_dim, gamma, tau, action_bound, bias,sigma)
hvac = environment.HVAC()
def evaluate(maddpg, n_episode=10, episode_length=25):
    # 对学习的策略进行评估,此时不会进行探索
    returns = np.zeros(agent_num)
    for _ in range(n_episode):
        obs = hvac.reset()
        for t_i in range(episode_length):
            actions = maddpg.take_action(obs, explore=False)
            obs, rew, done = hvac.HVAC_step(actions)
            rew = np.array(rew)
            returns += rew / n_episode
    return returns.tolist()

def return_state1(state1):
    state = [0]*6
    state[0] = state1[0]*23
    state[1] = state1[1]*20 + 10
    state[2] = state1[2] * 2000
    state[3] = state1[3]
    state[4] = state1[4] * 40 - 5
    state[5] = state1[5] * 800
    return state

def return_state2(state2):
    state = [0]*4
    state[0] = state2[0] * 23
    state[1] = state2[1] * 17280000
    state[2] = state2[2] * 20 + 10
    state[3] = state2[3] * 2000
    return state

policy1 = []
policy2 = []

state_set1 = []
state_set2 = []
dqn_loss = []
return_list = []
operating_list = []
energycost_list = []
reward_episode = []
cost_episode = []  # 记录每一轮的回报（return）
total_step = 0
state = hvac.reset()

for i in range(5):
    with tqdm(total=int(360/10), desc='ITeration %d' % i) as pbar:
        for i_episode in range(int(360/10)):
            rew_sum = 0
            cost_ = 0
            for e_i in range(episode_length):
                actions = maddpg.take_action(state, explore=True)

                policy1.append(actions[0]*7+23)

                policy2.append(actions[1])
                state_set1.append(return_state1(state[0]))
                state_set2.append(return_state2(state[1]))

                next_state, reward, done,cost, energy = hvac.HVAC_step(state,actions)
                cost_ += -cost

                replay_buffer.add(state, actions, reward, next_state, done)
                state = next_state

                return_list.append(reward)
                rew_sum += reward[0]
                rew_sum += reward[1]
                total_step += 1
                if replay_buffer.size(
                ) >= minimal_size :
                    sample = replay_buffer.sample(batch_size)
                    def stack_array(x):
                        rearranged = [[sub_x[i] for sub_x in x]
                                      for i in range(len(x[0]))]
                        return [
                            torch.FloatTensor(np.vstack(aa)).to(device)
                            for aa in rearranged
                        ]

                    sample = [stack_array(x) for x in sample]

                    for a_i in range(agent_num):
                        maddpg.update(sample, a_i)
                if total_step % 50 == 0 :
                    maddpg.update_all_targets()
            reward_episode.append(rew_sum)
            cost_episode.append(cost_)
            if (i_episode + 1) % 100 == 0:
                ep_returns = evaluate( maddpg, n_episode=100)
                return_list.append(ep_returns)
                print(f"Episode: {i_episode+1}, {ep_returns}")
            if hvac.step == 51408:
                hvac.rreset()
            pbar.update(1)
reward_list = pd.DataFrame(reward_episode, columns=['action_AC'])
reward_list.to_csv(f'./results/reward_{data}_episode{num_episodes}.csv')
cost_list = pd.DataFrame(cost_episode, columns=['cost_day'])
cost_list.to_csv(f'./results/cost_{data}_episode{num_episodes}.csv')

policy1 = pd.DataFrame(policy1, columns=['action'])
policy2 = pd.DataFrame(policy2, columns=['action'])

state_set1 = pd.DataFrame(state_set1, columns=['day_hour', 'T_zone','P_ITE_CPU','P_battery','T_Outdoor','I_Solar'])
state_set2 = pd.DataFrame(state_set2, columns=['day_hour', 'E_Storage_SOC','T_Zone','P_ITE_CPU'])

return_df = pd.DataFrame(return_list, columns=['return','action_battery'])
policy1.to_csv(
        f'./results/policy1_{data}_episode{num_episodes}_AC.csv')
policy2.to_csv(
        f'./results/policy2_{data}_episode{num_episodes}_AC.csv')
state_set1.to_csv(
        f'./results/state1_{data}_set_episode{num_episodes}_AC.csv')
state_set2.to_csv(
        f'./results/state2_{data}_set_episode{num_episodes}_AC.csv')
return_df.to_csv(
        f'./results/return_df_{data}_episode{num_episodes}_AC.csv')
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=False)
# 绘制策略图
axs.plot(reward_list, label='reward')
axs.legend()
axs.set_xticks(range(0, len(reward_list), 6))
# axs[0].set_title(f'{start_month}-{start_day}+{int(num_step)}_{Calibration}_wei{decimal_num}')
axs.grid(linestyle='--', linewidth=0.5, color='gray')
plt.subplots_adjust(hspace=0.02)

plt.show()

