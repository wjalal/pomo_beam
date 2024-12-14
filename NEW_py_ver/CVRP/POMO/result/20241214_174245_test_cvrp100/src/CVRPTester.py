
import torch

import os
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model

from utils.utils import *
import copy
import gc

class CVRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        if self.tester_params['test_data_load']['enable']:
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score = self._test_one_batch(batch_size)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

    def _test_one_batch(self, batch_size):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        
   
            gc.collect()
            res_arr = [{} for e in range(self.tester_params['env_count'])]
            # state, reward, done = self.env.pre_step()
            for k in range ( self.tester_params['env_count']):
                res_arr[k]['state'], res_arr[k]['reward'], res_arr[k]['done'] = self.env.pre_step()

            state, reward, done = sef.env.pre_step()
            print('started 1 batch')

            i = 0

            while not done:
                i += 1
                j = 0
                # Deep copy the environment for the required number of environments
                self.envs = [copy.deepcopy(self.env) for _ in range(self.tester_params['env_count'])]
                
                for k, env in enumerate(self.envs):
                    if not res_arr[k]['done']:
                        j += 1
                        print(f'step {i}, env {j}', end='\r')
                        # Assuming `self.model` returns a selection and some additional output
                        res_arr[k]['selected'], _ = self.model(res_arr[k]['state'])
                        res_arr[k]['state'], res_arr[k]['reward'], res_arr[k]['done'] = env.step(res_arr[k]['selected'])

                # Check if all environments are done
                done = all(res['done'] for res in res_arr)

                if done:
                    # Find the index of the entry in res_arr with the maximum reward
                    max_reward_index = max(range(len(res_arr)), key=lambda idx: res_arr[idx]['reward'])

                    # Assign state and reward from the max reward entry
                    max_reward_entry = res_arr[max_reward_index]
                    # state = max_reward_entry['state']
                    # reward = max_reward_entry['reward']

                    # Assign the environment corresponding to the maximum reward
                    self.env = self.envs[max_reward_index]
                    state, reward, done = sef.env.step(max_reward_entry['selected'])
                    break

            
            print('finished 1 batch')

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item()
