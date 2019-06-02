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
    learning_rate = 0.001

    v_state_num = 2
    hv_env_num = 1000
    percent = 100

    envs = hvenv('../data',v_state_num, v_num, hv_env_num)

    rlnnet = PGN(4*v_num,2).to(cuda0)
    optimizer = torch.optim.Adam(params=rlnnet.parameters(), lr=learning_rate)

    while True:
        obs = envs.reset()

        reward_list = []
        obs_list = []
        obs_next_list = []
        delta_v_list = []
        top_index_list = []
        bottem_index_list = []
        for i in range(v_num):
            #获取经验
            delta_v = action_choose(rlnnet,envs,obs,epsilon,cuda0)
            obs, obs_next,reward,done = envs.step(delta_v)
            obs_list.append(obs)
            # obs_next_list.append(obs_next)
            delta_v_list.append(delta_v)
            #记录
        reward_raw = np.array(reward).tolist()
        reward.sort()
        print(np.mean(reward))
        optimizer.zero_grad()


        #top
        top_reward = reward[-int(len(reward) / percent):].tolist()
        print(np.mean(top_reward))
        for v in  top_reward:
            top_index_list.append(reward_raw.index(v))
        obs_list_top,delta_v_list_top = filter_batch(np.array(top_index_list),obs_list,delta_v_list)

        #bottom
        bottem_reward = reward[:int(len(reward) / percent)].tolist()
        print(np.mean(bottem_reward))
        for v in bottem_reward:
            bottem_index_list.append(reward_raw.index(v))
        obs_list_bottem, delta_v_list_bottem = filter_batch(np.array(bottem_index_list), obs_list, delta_v_list)

        #top
        loss1 = 0
        for obs,d_v in zip(obs_list_top.squeeze(),delta_v_list_top.squeeze()):
            obs_real = np.real(obs)
            obs_imag = np.imag(obs)
            obs_input = np.concatenate((obs_real, obs_imag), axis=1)
            action_p_raw  = rlnnet(torch.Tensor(obs_input).to(cuda0).float())
            prob_action = F.softmax(action_p_raw, dim=1)
            label = np.stack(((d_v + 1) / 2, 1 - (d_v + 1) / 2), axis=0).T
            # delta_v_list_top+1/2
            # log_prob_actions_v = torch.Tensor(q).to(cuda0).float().max() * log_prob_action[range(hv_env_num), (d_v+1)/2]
            loss1+= F.mse_loss(torch.Tensor(label).to(cuda0).float(),prob_action)
        # loss1.backward()
        # optimizer.step()

        #bottom
        loss2 = 0
        for obs,d_v in zip(obs_list_bottem.squeeze(),delta_v_list_bottem.squeeze()):
            obs_real = np.real(obs)
            obs_imag = np.imag(obs)
            obs_input = np.concatenate((obs_real, obs_imag), axis=1)
            action_p_raw  = rlnnet(torch.Tensor(obs_input).to(cuda0).float())
            prob_action = F.softmax(action_p_raw, dim=1)
            label = np.stack((1 - (d_v + 1) / 2, (d_v + 1) / 2), axis=0).T

            # label = np.stack(((d_v + 1) / 2, 1 - (d_v + 1) / 2), axis=0).T
            # delta_v_list_top+1/2
            # log_prob_actions_v = torch.Tensor(q).to(cuda0).float().max() * log_prob_action[range(hv_env_num), (d_v+1)/2]
            loss2+= F.mse_loss(torch.Tensor(label).to(cuda0).float(),prob_action)
        (loss2+loss1).backward()
        optimizer.step()
        epsilon*= 0.9999
        print('opsilon: '+str(epsilon))
        # print(np.mean(q_list[63]))


def filter_batch(top_index_list,obs_list,delta_v_list):
    return np.array(obs_list)[:,[top_index_list]],np.array(delta_v_list)[:,[top_index_list]]



def q_calculate(reward_list,discount_gamma):
    q_value_list = []
    last_reward = 0
    reward_list.reverse()
    for i,reward in enumerate(reward_list):
        index = []
        reward_raw = np.array(reward)
        reward.sort()
        top_reward = reward[-int(len(reward)/100):].tolist()
        # for reward in top_reward:

        max_index = np.argmax(reward)
        if i!=0:
            q_value_list.append(discount_gamma*last_reward[max_index])
            # q_value_list.append(discount_gamma*last_reward+reward)
            last_reward = discount_gamma*last_reward[max_index]
        else:
            q_value_list.append(reward[max_index])
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