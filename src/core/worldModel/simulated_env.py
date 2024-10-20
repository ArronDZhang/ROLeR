# -*- coding: utf-8 -*-
# @Time    : 2021/8/2 4:04 下午
# @Author  : Chongming GAO
# @FileName: simulated_env.py

import gym
import numpy as np
import torch

from torch import FloatTensor
from tqdm import tqdm

from core.util import compute_action_distance, clip0, compute_exposure
# from .utils import knn_based_pred_reward


# from virtualTB.model.UserModel import UserModel

# from environments.VirtualTaobao.virtualTB.utils import *


class SimulatedEnv(gym.Env):

    def __init__(self, ensemble_models,
                 # dataset_val, need_transform,
                 env_task_class, task_env_param: dict, task_name: str, version: str = "v1", tau: float = 1.0,
                 use_exposure_intervention=False,
                 lambda_variance=1, lambda_entropy=1,
                 alpha_u=None, beta_i=None,
                 predicted_mat=None, maxvar_mat=None, entropy_dict=None,
                 entropy_window=None,
                 gamma_exposure=1,
                 step_n_actions=1,
                 entropy_min=None,
                 entropy_max=None,
                 knn_based_pred_reward=None,
                 change_pred_reward=False,
                 knn_based_uncertainty=None,
                 change_uncertainty=False,
                 r_and_u=None
                 ):

        self.ensemble_models = ensemble_models.eval()
        # self.dataset_val = dataset_val
        # self.need_transform = need_transform
        # self.user_model = user_model.eval()

        # if task_name == "KuaiEnv-v0":
        #     from environments.KuaiRec.env.KuaiEnv import KuaiEnv
        #     self.env_task = KuaiEnv(**task_env_param)
        self.env_task = env_task_class(**task_env_param)

        self.observation_space = self.env_task.observation_space
        self.action_space = self.env_task.action_space
        self.cum_reward = 0  # total_a in virtualtaobao
        self.total_turn = 0  # total_c in virtualtaobao
        self.env_name = task_name
        self.version = version
        self.tau = tau
        self.use_exposure_intervention = use_exposure_intervention
        self.alpha_u = alpha_u
        self.beta_i = beta_i
        self.predicted_mat = predicted_mat
        self.maxvar_mat = maxvar_mat
        self.entropy_dict = entropy_dict
        self.entropy_window = entropy_window
        self.step_n_actions = step_n_actions

        self.gamma_exposure = gamma_exposure
        self._reset_history()

        # entropy_min = 0
        # entropy_max = 0
        # if 0 in self.entropy_window:
        #     entropy_min = self.entropy_dict["on_user"].min()
        #     entropy_max = self.entropy_dict["on_user"].max()
        #
        # # todo: comments
        # entropy_set = set(self.entropy_window) - set([0])
        # if len(entropy_set):
        #     for entropy_term in entropy_set:
        #         entropy_min += min([v for k, v in self.entropy_dict["map"].items() if len(k) == entropy_term])
        #         entropy_max += max([v for k, v in self.entropy_dict["map"].items() if len(k) == entropy_term])

        self.lambda_variance = lambda_variance
        self.lambda_entropy = lambda_entropy
        self.MIN_R = predicted_mat.min() - lambda_variance * maxvar_mat.max() + lambda_entropy * entropy_min
        self.MAX_R = predicted_mat.max() - lambda_variance * maxvar_mat.min() + lambda_entropy * entropy_max

        self.knn_based_pred_reward = knn_based_pred_reward
        self.change_pred_reward = change_pred_reward
        self.knn_based_var = knn_based_uncertainty
        self.change_uncertainty = change_uncertainty
        self.r_and_u = r_and_u

    # def compile(self, num_env=1):
    #     self.env_list = DummyVectorEnv([lambda: gym.make(self.env_task) for _ in range(num_env)])

    def _construct_state(self, reward):
        res = self.env_task.state
        return res

    def seed(self, sd=0):
        torch.manual_seed(sd)

    def reset(self):
        self.cum_reward = 0
        self.total_turn = 0
        self.reward = 0
        self.action = None
        self.env_task.action = None
        self.state = self.env_task.reset()

        self._reset_history()
        if self.env_name == "VirtualTB-v0":
            self.cur_user = self.state[:-3]
        else:  # elif self.env_name == "KuaiEnv-v0":
            self.cur_user = self.state
        return self.state

    def render(self, mode='human', close=False):
        self.env_task.render(mode)

    def _compute_pred_reward(self, exposure_effect, action):
        if self.env_name == "VirtualTB-v0":
            feature = np.concatenate((self.cur_user, np.array([self.reward, 0, self.total_turn]), action), axis=-1)
            feature_tensor = torch.unsqueeze(torch.tensor(feature, device=self.user_model.device, dtype=torch.float), 0)
            # pred_reward = self.user_model(feature_tensor).detach().cpu().numpy().squeeze().round()
            pred_reward = self.user_model.forward(feature_tensor).detach().cpu().numpy().squeeze()
            if pred_reward < 0:
                pred_reward = 0
            if pred_reward > 10:
                pred_reward = 10
        else:  # elif self.env_name == "KuaiEnv-v0":

            # # get prediction
            # pred_reward = self.predicted_mat[self.cur_user[0], action]  # todo

            ## Don YZ 
            if self.change_pred_reward:
                pred_reward = self.knn_based_pred_reward[self.cur_user[0], action]
                # pred_reward = max(self.knn_based_pred_reward[self.cur_user[0], action], self.predicted_mat[self.cur_user[0], action])
                # pred_reward = self.knn_based_pred_reward[self.cur_user[0], action]+self.predicted_mat[self.cur_user[0], action]
            else:
                pred_reward = self.predicted_mat[self.cur_user[0], action]

            if self.change_uncertainty:
                max_var = self.knn_based_var[self.cur_user[0], action]
            else:
                max_var = self.maxvar_mat[self.cur_user[0], action]
            
            # if max_var != 0:
            #     print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")

            # # get variance
            # max_var = self.maxvar_mat[self.cur_user[0], action]  # todo

            # get entropy
            # entropy_u = 0
            # if 0 in self.entropy_window:
            #     entropy_u = self.entropy_dict["on_user"].loc[self.cur_user[0]]
            entropy = 0
            entropy_set = set(self.entropy_window) - {0}
            if len(entropy_set):
                action_k = self.history_action[max(0, self.total_turn - self.step_n_actions + 1):self.total_turn + 1]
                if hasattr(self.env_task, "lbe_item") and self.env_task.lbe_item:
                    action_trans = self.env_task.lbe_item.inverse_transform(action_k)
                else:
                    action_trans = action_k
                action_reverse = action_trans[::-1]
                for k in range(len(action_reverse)):
                    action_set = tuple(sorted(action_reverse[:k + 1]))
                    # print(action_set)
                    if action_set in self.entropy_dict["map"]:
                        entropy += self.entropy_dict["map"][action_set]
                    else:
                        entropy += 1 # todo! 补足差额
                if len(action_reverse) < self.step_n_actions:
                    entropy += self.step_n_actions - len(action_reverse) # todo 补足差额

            # original
            if self.r_and_u == 'minus':
                penalized_reward = pred_reward - self.lambda_variance * max_var + \
                                self.lambda_entropy * entropy - self.MIN_R
            elif self.r_and_u == 'inverse':
                penalized_reward = pred_reward * self.lambda_variance * (1/(max_var+1e-9)) + \
                                self.lambda_entropy * entropy - self.MIN_R
            elif self.r_and_u == 'sample':
                ru = torch.normal(torch.tensor(pred_reward), torch.tensor(self.lambda_variance * max_var))
                penalized_reward = np.array(ru + self.lambda_entropy * entropy - self.MIN_R)
            else:
                NotImplementedError


            ## Don YZ test
            # rew = torch.normal(torch.tensor(pred_reward), torch.tensor(self.lambda_entropy * entropy \ 
            # - self.MIN_R - self.lambda_variance * max_var))
            # penalized_reward = np.array(rew)
            # penalized_reward = self.lambda_entropy * entropy
            # penalized_reward = - self.lambda_variance * max_var + self.lambda_entropy * entropy - self.MIN_R
            # penalized_reward = pred_reward - self.lambda_variance * max_var # compare with MOPO
            # penalized_reward = - self.lambda_variance * max_var
            # penalized_reward = pred_reward + self.lambda_entropy * entropy - self.MIN_R
            # penalized_reward = pred_reward

            # penalized_reward = pred_reward - self.MIN_R

            ## self.max_history, self.env_task.total_turn self.total_turn (need to +1) can reflect env step here

            # decay_func, ascending_func = np.exp(-decay_rate*t), np.exp(0.06*t)
            # penalized_reward = pred_reward * np.exp(-1 * self.max_history) - self.lambda_variance * max_var + \
            #                    np.exp(0.06 * self.max_history) * entropy - self.MIN_R
            # penalized_reward = pred_reward - self.lambda_variance * max_var + \
            #                             np.exp(0.06 * self.max_history) * entropy - self.MIN_R
            # penalized_reward = pred_reward * np.exp(-1 * self.max_history) - self.lambda_variance * max_var + \
            #                    self.lambda_entropy * entropy - self.MIN_R

            # decay_func, ascending_func = np.exp(-decay_rate*t), np.exp(0.06*t)
            # penalized_reward = pred_reward * np.exp(-1 * self.max_history) + np.exp(0.06 * self.max_history) * entropy - self.MIN_R
            # penalized_reward = pred_reward + np.exp(0.06 * self.max_history) * entropy - self.MIN_R
            # penalized_reward = pred_reward * np.exp(-1 * self.max_history) + self.lambda_entropy * entropy - self.MIN_R

            # decay_func, ascending_func = np.exp(-0.06*t), np.exp(0.06*t)
            # penalized_reward = pred_reward * np.exp(-0.06 * self.max_history) + np.exp(0.06 * self.max_history) * entropy - self.MIN_R
            # penalized_reward = pred_reward + np.exp(0.06 * self.max_history) * entropy - self.MIN_R
            # penalized_reward = pred_reward * np.exp(-0.06 * self.max_history) + self.lambda_entropy * entropy - self.MIN_R

            # decay_func, ascending_func = np.exp(-0.06*t), np.exp(0.06*t)
            # penalized_reward = pred_reward * np.exp(-0.06 * self.max_history) - self.lambda_variance * max_var + \
            #                    np.exp(0.06 * self.max_history) * entropy - self.MIN_R
            # penalized_reward = pred_reward - self.lambda_variance * max_var + \
            #                             np.exp(0.06 * self.max_history) * entropy - self.MIN_R
            # penalized_reward = pred_reward * np.exp(-1 * self.max_history) - self.lambda_variance * max_var + \
            #                    self.lambda_entropy * entropy - self.MIN_R

            # penalized_reward = pred_reward - np.exp(-0.1*self.max_history) * max_var + \
            #                    self.lambda_entropy * entropy - self.MIN_R
            # penalized_reward = pred_reward - (1/600*self.max_history) * max_var + \
            #                    self.lambda_entropy * entropy - self.MIN_R

        if self.version == "v1":
            # version 1
            final_reward = clip0(penalized_reward) / (1.0 + exposure_effect)
        else:
            # version 2
            final_reward = clip0(penalized_reward - exposure_effect)

        return final_reward

    def step(self, action: FloatTensor):
        # 1. Collect ground-truth transition info
        self.action = action
        real_state, real_reward, real_done, real_info = self.env_task.step(action)

        # 2. Compute intervened exposure effect e^*_t(u, i)
        t = int(self.total_turn)
        if self.use_exposure_intervention:
            exposure_effect = self._compute_exposure_effect(t, action)
        else:
            exposure_effect = 0

        if t < self.env_task.max_turn:
            self._add_action_to_history(t, action, exposure_effect)

        # 3. Predict click score, i.e, reward
        pred_reward = self._compute_pred_reward(exposure_effect, action)

        ## Don YZ test
        # max_var = self.maxvar_mat[self.cur_user[0], action]
        # entropy = 0
        # entropy_set = set(self.entropy_window) - {0}
        # if len(entropy_set):
        #     action_k = self.history_action[max(0, self.total_turn - self.step_n_actions + 1):self.total_turn + 1]
        #     if hasattr(self.env_task, "lbe_item") and self.env_task.lbe_item:
        #         action_trans = self.env_task.lbe_item.inverse_transform(action_k)
        #     else:
        #         action_trans = action_k
        #     action_reverse = action_trans[::-1]
        #     for k in range(len(action_reverse)):
        #         action_set = tuple(sorted(action_reverse[:k + 1]))
        #         # print(action_set)
        #         if action_set in self.entropy_dict["map"]:
        #             entropy += self.entropy_dict["map"][action_set]
        #         else:
        #             entropy += 1 # todo! 补足差额
        #     if len(action_reverse) < self.step_n_actions:
        #         entropy += self.step_n_actions - len(action_reverse) # todo 补足差额
        # pred_reward = real_reward - self.lambda_variance * max_var + \
        #                        self.lambda_entropy * entropy - self.MIN_R

        # pred_reward = real_reward

        self.reward = pred_reward
        self.cum_reward += pred_reward
        self.total_turn = self.env_task.total_turn

        done = real_done

        # Rethink commented, do not use new user as new state
        # if done:
        #     self.state = self.env_task.reset()

        self.state = self._construct_state(pred_reward)
        return self.state, pred_reward, done, {'CTR': self.cum_reward / self.total_turn / 10}

    def _compute_exposure_effect(self, t, action):

        if t == 0:
            return 0

        a_history = self.history_action[:t]
        distance = compute_action_distance(action, a_history, self.env_name, self.env_task)
        t_diff = t - np.arange(t)
        exposure_effect = compute_exposure(t_diff, distance, self.tau)

        if self.alpha_u is not None:
            u_id = self.env_task.lbe_user.inverse_transform(self.cur_user)[0]
            p_id = self.env_task.lbe_item.inverse_transform([action])[0]
            a_u = self.alpha_u[u_id]
            b_i = self.beta_i[p_id]
            exposure_effect_new = float(exposure_effect * a_u * b_i)
        else:
            exposure_effect_new = exposure_effect

        exposure_gamma = exposure_effect_new * self.gamma_exposure

        return exposure_gamma

    def _reset_history(self):
        # self.history_action = {}
        if self.env_name == "VirtualTB-v0":
            # self.history_action = np.empty([0, self.action_space.shape[0]])
            self.history_action = np.zeros([self.env_task.max_turn, self.env_task.action_space.shape[0]])
        else:  # elif self.env_name == "KuaiEnv-v0":
            self.history_action = np.zeros(self.env_task.max_turn, dtype=int)
        self.history_exposure = {}
        self.max_history = 0

    def _add_action_to_history(self, t, action, exposure):
        if self.env_name == "VirtualTB-v0":
            action2 = np.expand_dims(action, 0)
            # self.history_action = np.append(self.history_action, action2, axis=0)
            self.history_action[t] = action2
        else:  # elif self.env_name == "KuaiEnv-v0":
            self.history_action[t] = action

        self.history_exposure[t] = exposure

        assert self.max_history == t
        self.max_history += 1
