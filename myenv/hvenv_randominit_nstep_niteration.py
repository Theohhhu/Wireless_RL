import numpy as np
import scipy.io as sio

ENV_TOTAL = 1000

class hvenv(object):

    def __init__(self,path,v_state_num, v_num,hv_env_num,step_num,iteration_num):
        super(hvenv, self).__init__()
        hv_batch_index = np.random.choice(np.arange(ENV_TOTAL), size=hv_env_num, replace=False)

        self.iter_num = iteration_num
        self.step_num = step_num
        self.h_total = self.mat_load(path)
        self.v_state_num = v_state_num
        self.v_num = v_num
        self.hv_env_num = hv_env_num

        #test
        # self.v = np.random.randint(2,size=(5,64))*2-1
        # 生成 一个batch的变相器状态v
        # self.v = np.random.randint((hv_env_num,v_num))

        self.v = np.random.randint(0, 2, size=(hv_env_num, v_num))*2-1


        # 随机挑选一个batch的信道 转置
        self.h = self.h_total[hv_batch_index].squeeze().T
        # self.v = np.random.randint(self.v_state_num, size = (self.hv_env_num,self.v_num)) * self.v_state_num - 1

        self.iter_count = self.iter_num
        self.count = self.v_num

    def step(self,delta_v):
        v = np.array(self.v)

        if  -self.count+self.step_num!=0:
            self.v[:,-self.count:-self.count+self.step_num]= delta_v
        else:
            self.v[:, -self.count:] = delta_v

        h = self.h
        reward = self.reward()

        self.count -= self.step_num

        if self.count == 0:
            self.iter_count-=1
            self.count = self.v_num
        # if self.count == 0:
        #     obs_next = None
        #     obs = np.concatenate((v, h.T), axis=1)
        # else:
        #     v_next = self.v
        #     obs = np.concatenate((v, h.T), axis=1)
        #     obs_next = np.concatenate((v_next, h.T), axis=1)
        v_next = self.v
        obs = np.concatenate((v, h.T), axis=1)
        obs_next = np.concatenate((v_next, h.T), axis=1)

        return obs,obs_next,reward,self.iter_count == 0


    def reward(self):
        self.v_real = np.real(self.v)
        self.v_imag = np.imag(self.v)
        self.h_real = np.real(self.h)
        self.h_imag = np.imag(self.h)
        element1 = (np.diag(np.matrix(self.v_real)*np.matrix(self.h_real)-np.matrix(self.v_imag)*np.matrix(self.h_imag)))**2
        element2 = (np.diag(np.matrix(self.v_real)*np.matrix(self.h_imag)+np.matrix(self.v_imag)*np.matrix(self.h_real)))**2

        rate = np.log2(1+1/64*(element1+element2))
        # reward = np.log2(1+1/64*np.matrix(self.v).H*np.matrix(self.h).H*self.h*self.v)
        return  rate

    def random_delta_v(self):
        return np.random.randint(self.v_state_num, size = (self.hv_env_num,self.step_num)) * self.v_state_num - 1

    def mat_load(self,path):
        print('loading data...')
        # load the perfect csi
        h = sio.loadmat(path + '/pcsi.mat')['pcsi']
        # load the estimated csi
        # h_est = sio.loadmat(path + '/ecsi.mat')['ecsi']
        print('loading complete')
        # print('The shape of CSI is: ', h_est.shape)
        return h

    #初始化环境 重新选择信道 重新生成变相器 重新计数
    def reset(self):
        hv_batch_index = np.random.choice(np.arange(ENV_TOTAL), size=self.hv_env_num, replace=False)
        self.v = np.random.randint(0, 2, size=(self.hv_env_num, self.v_num))*2-1
        self.h = self.h_total[hv_batch_index].squeeze().T
        self.count = self.v_num
        self.iter_count = self.iter_num
        return np.concatenate((self.v, self.h.T), axis=1)






def main():
    env = hvenv('../data',2, 64, 5)
    done = False
    for i in range(64):
        delta_v = env.random_delta_v()
        obs, obs_next,reward,done = env.step(delta_v)
        obs_real = np.real(obs)
        obs_imag = np.imag(obs)
        obs_input = np.concatenate((obs_real,obs_imag),axis = 1)
    if(done):
        obs= env.reset()
    for i in range(64):
        delta_v = env.random_delta_v()
        obs, obs_next,reward,done = env.step(delta_v)
    print(env)


if __name__ == '__main__':
    main()





