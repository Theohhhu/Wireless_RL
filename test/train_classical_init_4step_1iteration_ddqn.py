import numpy as np
import torch
from myenv.hvenv_classical_init_nstep_niteration  import hvenv
from rlnnet.rl_dqn import DQN
import torch.nn.functional as F
import collections
import torch.nn as nn
import datetime
import string


Experience = collections.namedtuple('Experience', field_names=['obs','obs_next', 'delta_v', 'reward', 'done'])
GAMMA = 0.99
SYC_NUM = 2000
BUFFER_START_NUM = 30000
BUFFER_MAX_NUM = 300000

## 步数设定是全局变量 可取v_num的任意约数 包含v_num 不含1
step_num = 8


def int_to_list(int_num):
    b = []
    str = bin(int_num).replace('0b', '')
    for c in str:
        b.append(int(c))
    return  b

def main():
    cuda0 = torch.device('cuda:0')
    v_num = 64
    # discount_gamma = 0.9
    epsilon = 0
    learning_rate = 0.001
    epsilon_cut = 0.99
    epsilon_min = 0

    #跟新轮次设定
    iteration_num = 1

    v_state_num = 2
    batch_size = 100
    start_time = datetime.datetime.now()

    hv_env_num = 50
    envs = hvenv('../data',v_state_num, v_num, hv_env_num,step_num,iteration_num)

    hv_env_num_val=1000
    envs_val = hvenv('../data',v_state_num, v_num, hv_env_num_val,step_num,iteration_num)


    rlnnet = DQN(4*v_num,2**step_num).to(cuda0)
    rln_tgt_net = DQN(4*v_num,2**step_num).to(cuda0)
    optimizer = torch.optim.Adam(params=rlnnet.parameters(), lr=learning_rate)
    exp_buffer = collections.deque()
    exp_buffer_val = collections.deque()


    reward = 0
    reward_val = 0

    obs = envs.reset()
    obs_val = envs_val.reset()

    i = 0


    while True:

        if i % SYC_NUM == 0:
            print('----------------------------')
            print('start_time: ' + str(start_time))
            print('i/SYC_NUM: ' + str(i / SYC_NUM))
            print('syc epsilon: ' + str(epsilon))
            print('learning_rate: '+ str(learning_rate))
            print('v_state_num: '+str(v_state_num))
            print('v_num: '+str(v_num))
            print('hv_env_num: '+str(hv_env_num))
            print('batch_size: '+str(batch_size))
            print('step_num: '+str(step_num))
            print('iteration_num: '+str(iteration_num))
            print('BUFFER_START_NUM: '+str(BUFFER_START_NUM))
            print('BUFFER_length_NUM: '+str(len(exp_buffer)))
            print('----------------------------')


            epsilon *= epsilon_cut

            torch.save(rlnnet, '../weight/ddqn_classical_init_'+str(step_num)+'step_'+str(iteration_num)+'iteration_best_model.pth')
            rln_tgt_net.load_state_dict(rlnnet.state_dict())

        if epsilon < epsilon_min:
            epsilon = epsilon_min

        i += 1

        rlnnet.train()
        #获取经验
        exp_buffer, obs, reward = fresh_exp_buffer(exp_buffer, rlnnet, envs, obs, epsilon, cuda0,reward,BUFFER_START_NUM,BUFFER_MAX_NUM)
        optimizer.zero_grad()
        exp_batch_index = np.random.choice(np.arange(len(exp_buffer)), size=batch_size, replace=False)
        batch = batch_sample(exp_batch_index,exp_buffer)

        loss_t = calc_loss(batch, rlnnet, rln_tgt_net, device=cuda0)
        loss_t.backward()
        optimizer.step()

        if i%1000==0:
            rlnnet.eval()
            print('-------eval testing: epsilon = 0 -------')
            for _ in range(int(v_num*iteration_num/step_num)):
                exp_buffer_val, obs_val, reward_val = fresh_exp_buffer(exp_buffer_val, rlnnet, envs_val, obs_val, 0, cuda0, reward_val,
                                                       100000, 110000)
            print('length buffer: ' + str(len(exp_buffer_val)))
            print('-------eval end-------')




def batch_sample(exp_batch_index,exp_buffer):
    states, next_states, actions, rewards, dones = zip(*[exp_buffer[idx] for idx in exp_batch_index])
    return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(next_states)


def calc_loss(batch, net, tgt_net, device):
    states, actions, rewards, dones, next_states = batch
    actions = (actions+1)/2
    multiple_list = []
    for i in range(step_num):
        multiple_list.insert(0,2**i)
    multiple_array = np.array(multiple_list)

    actions = (multiple_array * actions).sum(1)
    actions_v = torch.Tensor(actions).to(device)
    rewards_v = torch.Tensor(rewards).to(device)

    obs_real = np.real(states)
    obs_imag = np.imag(states)
    obs_input = np.concatenate((obs_real, obs_imag), axis=1)

    obs_real_next = np.real(next_states)
    obs_imag_next = np.imag(next_states)
    obs_input_next = np.concatenate((obs_real_next, obs_imag_next), axis=1)

    states_v = torch.Tensor(obs_input).to(device).float()
    next_states_v = torch.Tensor(obs_input_next).to(device).float()

    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.long().unsqueeze(-1)).squeeze(-1)
    next_action_value = net(next_states_v).max(1)[1]
    next_state_values = tgt_net(next_states_v).gather(1, next_action_value.long().unsqueeze(-1)).squeeze(-1)
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def fresh_exp_buffer(exp_buffer, rlnnet, envs, obs, epsilon, cuda0,reward,buf_start_num,buf_max_num):

    while len(exp_buffer)<buf_start_num:
        exp,obs,done,reward = play_step(rlnnet, envs, obs, epsilon, cuda0,reward)
        if done:
            exp_unzip(done, exp, exp_buffer)
            obs = envs.reset()
            print(np.mean(reward))

            reward = 0
        else:
            exp_unzip(done, exp, exp_buffer)

        # if done:
        #     obs = envs.reset()
        #     reward = 0


    if len(exp_buffer) >= buf_start_num and len(exp_buffer) <buf_max_num:
        exp, obs, done, reward = play_step(rlnnet, envs, obs, epsilon, cuda0, reward)
        if done:
            exp_unzip(done, exp, exp_buffer)
            obs = envs.reset()
            print(np.mean(reward))

            reward = 0
        else:
            exp_unzip(done, exp, exp_buffer)

        return exp_buffer,obs,reward

    if len(exp_buffer) >= buf_max_num:
        exp, obs, done, reward = play_step(rlnnet, envs, obs, epsilon, cuda0, reward)

        if done:
            exp_unzip(done, exp, exp_buffer, True)
            obs = envs.reset()
            print(np.mean(reward))

            reward = 0
        else:
            exp_unzip(done, exp, exp_buffer, True)

        return  exp_buffer,obs,reward

def exp_unzip(done,exp,exp_buffer,flag=False):
    # if done:
    #     for obs_1,delta_v_1, r_1 in zip(exp[0].tolist(), exp[2].tolist(),
    #                                                  exp[3].tolist()):
    #         exp_1 = Experience(obs_1, None, delta_v_1, r_1, done)
    #         exp_buffer.append(exp_1)
    #         if flag:
    #             exp_buffer.popleft()
    #
    # else:
    #     for obs_1, obs_next_1, delta_v_1, r_1 in zip(exp[0].tolist(), exp[1].tolist(), exp[2].tolist(),
    #                                                  exp[3].tolist()):
    #         exp_1 = Experience(obs_1, obs_next_1, delta_v_1, r_1, done)
    #         exp_buffer.append(exp_1)
    #         if flag:
    #             exp_buffer.popleft()
    for obs_1, obs_next_1, delta_v_1, r_1 in zip(exp[0].tolist(), exp[1].tolist(), exp[2].tolist(),
                                                 exp[3].tolist()):
        exp_1 = Experience(obs_1, obs_next_1, delta_v_1, r_1, done)
        exp_buffer.append(exp_1)
        if flag:
            exp_buffer.popleft()


def play_step(rlnnet, envs, obs, epsilon, cuda0,last_reward):

    delta_v = action_choose(rlnnet, envs, obs, epsilon, cuda0)
    obs, obs_next, reward, done = envs.step(delta_v)
    exp = Experience(obs, obs_next, delta_v, reward-last_reward, done)
    return exp,obs_next,done,reward


def action_choose(nnet,envs,obs,epsilon,device):
    # 分解obs
    obs_real = np.real(obs)
    obs_imag = np.imag(obs)
    obs_input = np.concatenate((obs_real, obs_imag), axis=1)
    #贪婪策略决定动作
    if np.random.random()<epsilon:
        delta_v = envs.random_delta_v()
    else:
        delta_v_total = []
        delta_v_p_raw = nnet(torch.Tensor(obs_input).to(device).float())
        delta_v_index = torch.max(delta_v_p_raw,1)[1]
        delta_v_list = np.array(delta_v_index.cpu()).tolist()
        for v in delta_v_index:
            v_list = int_to_list(v)
            if len(v_list)< step_num:
                for _ in range(step_num-len(v_list)):
                    v_list.insert(0,0)
            delta_v_total.append(np.array(v_list))
        delta_v = np.array(delta_v_total)*2-1

    return  delta_v



if __name__ == '__main__':
    main()