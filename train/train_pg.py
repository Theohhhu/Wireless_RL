import numpy as np
import torch
from myenv.hvenv import hvenv
from rlnnet.rl_pg import PGN
import torch.nn.functional as F



def main():
    cuda0 = torch.device('cuda:0')
    v_num = 64
    discount_gamma = 0.9
    epsilon = 0.999
    learning_rate = 0.01

    v_state_num = 2
    hv_env_num = 1000

    envs = hvenv('../data',v_state_num, v_num, hv_env_num)

    rlnnet = PGN(4*v_num,2).to(cuda0)
    optimizer = torch.optim.Adam(params=rlnnet.parameters(), lr=learning_rate)

    while True:
        obs = envs.reset()

        reward_list = []
        obs_list = []
        obs_next_list = []
        delta_v_list = []
        last_reward = 0
        for i in range(v_num):
            #获取经验
            delta_v = action_choose(rlnnet,envs,obs,epsilon,cuda0)
            obs, obs_next,reward,done = envs.step(delta_v)
            #记录
            reward_list.append(reward-last_reward)
            obs_list.append(obs)
            obs_next_list.append(obs_next)
            delta_v_list.append(delta_v)

            last_reward = reward

        q_list = q_calculate(reward_list,discount_gamma)

        loss_policy_v = 0
        for obs,q,d_v in zip(obs_list,q_list,delta_v_list):
            obs_real = np.real(obs)
            obs_imag = np.imag(obs)
            obs_input = np.concatenate((obs_real, obs_imag), axis=1)
            action_p_raw  = rlnnet(torch.Tensor(obs_input).to(cuda0).float())
            log_prob_action = F.log_softmax(action_p_raw, dim=1)
            log_prob_actions_v = torch.Tensor(q).to(cuda0).float() * log_prob_action[range(hv_env_num), (d_v+1)/2]
            loss_policy_v= -log_prob_actions_v.mean()
            loss_policy_v.backward()
            optimizer.step()
        epsilon*= 0.999
        print('opsilon: '+str(epsilon))
        print(np.mean(q_list[63]))





def q_calculate(reward_list,discount_gamma):
    q_value_list = []
    last_reward = 0
    reward_list.reverse()
    for i,reward in enumerate(reward_list):
        if i!=0:
            # q_value_list.append(discount_gamma*last_reward)
            q_value_list.append(discount_gamma*last_reward+reward)
            last_reward = discount_gamma*last_reward+reward
        else:
            q_value_list.append(reward)
            last_reward = reward
    q_value_list.reverse()
    return  q_value_list

def action_choose(nnet,envs,obs,epsilon,device):
    # 分解obs
    obs_real = np.real(obs)
    obs_imag = np.imag(obs)
    obs_input = np.concatenate((obs_real, obs_imag), axis=1)
    #贪婪策略决定动作
    if np.random.random()<epsilon:
        delta_v = envs.random_delta_v()
    else:
        delta_v_p_raw = nnet(torch.Tensor(obs_input).to(device).float())
        delta_v_index = torch.max(delta_v_p_raw,1)[1]
        delta_v = delta_v_index.cpu().numpy()*2-1

    return  delta_v



if __name__ == '__main__':
    main()