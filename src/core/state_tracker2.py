# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 9:49 上午
# @Author  : Chongming GAO
# @FileName: state_tracker.py

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
import torch.utils.checkpoint as ckpt

from core.inputs import input_from_feature_columns
from core.user_model import build_input_features, compute_input_dim
from deepctr_torch.inputs import combined_dnn_input

FLOAT = torch.FloatTensor


def reverse_padded_sequence(tensor: Tensor, lengths: Tensor):
    """
    Change the input tensor from:
    [[1, 2, 3, 4, 5],
    [1, 2, 0, 0, 0],
    [1, 2, 3, 0, 0]]
    to:
    [[5, 4, 3, 2, 1],
    [2, 1, 0, 0, 0],
    [3, 2, 1, 0, 0]]
    :param tensor: (B, max_length, *)
    :param lengths: (B,)
    :return:
    """
    out = torch.zeros_like(tensor)
    for i in range(tensor.shape[0]):
        out[i, :lengths[i]] = tensor[i, :lengths[i]].flip(dims=[0])
    return out


def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res


class StateTracker_Base(nn.Module):
    def __init__(self, user_columns, action_columns, feedback_columns, dim_model, device="cpu", window_size=10):
        super().__init__()
        self.user_columns = user_columns
        self.action_columns = action_columns
        self.feedback_columns = feedback_columns

        self.user_index = build_input_features(user_columns)
        self.action_index = build_input_features(action_columns)
        self.feedback_index = build_input_features(feedback_columns)

        self.dim_model = dim_model
        self.window_size = window_size
        self.device = device

    def get_embedding(self, X, type):
        if type == "user":
            feat_columns = self.user_columns
            feat_index = self.user_index
        elif type == "action":
            feat_columns = self.action_columns
            feat_index = self.action_index
        elif type == "feedback":
            feat_columns = self.feedback_columns
            feat_index = self.feedback_index

        X[X == -1] = self.num_item # 将X中的-1换成self.num_item

        sparse_embedding_list, dense_value_list = input_from_feature_columns(FLOAT(X).to(self.device), feat_columns,
                                                                             self.embedding_dict, feat_index,
                                                                             support_dense=True, device=self.device)
        new_X = combined_dnn_input(sparse_embedding_list, dense_value_list)
        X_res = new_X

        return X_res

    def build_state(self, obs=None,
                    env_id=None,
                    obs_next=None,
                    reset=False, **kwargs):
        if reset:
            self.user = None
            return

        if obs is not None:  # 1. initialize the state vectors
            self.user = obs
            # item = np.ones_like(obs) * np.nan
            item = np.ones_like(obs) * self.num_item
            ui_pair = np.hstack([self.user, item])
            res = {"obs": ui_pair}

        elif obs_next is not None:  # 2. add action autoregressively
            item = obs_next
            user = self.user[env_id]
            ui_pair = np.hstack([user, item])
            res = {"obs_next": ui_pair}

        return res


class StateTrackerAvg2(StateTracker_Base):
    def __init__(self, user_columns, action_columns, feedback_columns, dim_model, saved_embedding,
                 train_max=None, train_min=None, test_max=None, test_min=None, reward_handle="no",
                 device="cpu", use_userEmbedding=False, window_size=10, scratch=False):
        super(StateTrackerAvg2, self).__init__(user_columns=user_columns, action_columns=action_columns,
                                               feedback_columns=feedback_columns, dim_model=dim_model, device=device,
                                               window_size=window_size)

        self.test_min = test_min
        self.test_max = test_max
        self.train_min = train_min
        self.train_max = train_max
        self.reward_handle = reward_handle

        assert saved_embedding is not None
        self.embedding_dict = saved_embedding.to(device)

        self.num_item = self.embedding_dict.feat_item.weight.shape[0] # KuaiEnv 3327
        self.dim_item = self.embedding_dict.feat_item.weight.shape[1] # KuaiEnv 41

        # Add a new embedding vector
        new_embedding = FLOAT(1, self.dim_model).to(device) # KuaiEnv 1*41
        nn.init.normal_(new_embedding, mean=0, std=0.01)
        emb_cat = torch.cat([self.embedding_dict.feat_item.weight.data, new_embedding]) # KuaiEnv 3328*41
        # print("======================================================")
        # print("self.embedding_dict.feat_item.weight.requires_grad", self.embedding_dict.feat_item.weight.requires_grad)
        # print("======================================================")

        # whether train embedding from scratch
        self.scratch = scratch
        if self.scratch:
            nn.init.normal_(emb_cat, mean=0, std=0.01)

        new_item_embedding = torch.nn.Embedding.from_pretrained(
            emb_cat, freeze=not self.embedding_dict.feat_item.weight.requires_grad)

        ## KuaiEnv freeze=False means the embs will change during training
        self.embedding_dict.feat_item = new_item_embedding

        self.use_userEmbedding = use_userEmbedding
        if self.use_userEmbedding:
            self.ffn_user = nn.Linear(compute_input_dim(self.user_columns), self.dim_model, device=self.device)

    def forward(self, buffer=None, indices=None, obs=None, reset=None, is_obs=None, is_train=True):
        # print(reset)

        if reset:  # get user embedding # used when trajectory is started

            users = np.expand_dims(obs[:, 0], -1)
            items = np.expand_dims(obs[:, 1], -1)

            # e_i = torch.ones(obs.shape[0], self.dim_item, device=self.device)
            # nn.init.normal_(e_i, mean=0, std=0.0001)

            e_i = self.get_embedding(items, "action")
            # print("======================================================")
            # print("action embedding", e_i.shape)
            # print("======================================================")

            if self.use_userEmbedding: # 没有用到，与论文sec5.4描述一致
                e_u = self.get_embedding(users, "user")
                print("======================================================")
                print("user embedding", e_u.shape)
                print("======================================================")
                # s0 = self.ffn_user(e_u)
                s0 = torch.cat([e_u, e_i], dim=-1)
            else:
                s0 = e_i

            r0 = torch.ones(len(s0), 1).to(s0.device) # todo: define init reward as 1
            if self.reward_handle == "mul":
                state_res = s0 * 1
            elif self.reward_handle == "cat":
                state_res = torch.cat([s0, r0], 1)
            elif self.reward_handle == "cat2":
                state_res = torch.cat([s0, r0], 1)
            else:
                state_res = s0

            # print("======================================================")
            # print("state_res", state_res.shape)
            # print("======================================================")
            return state_res

        else: # used when trajectory is continuing
            index = indices
            flag_has_init = np.zeros_like(index, dtype=bool)

            obs_all = np.zeros([0, 2], dtype=int)
            rew_all = np.zeros([0])

            live_mat = np.zeros([0, len(index)], dtype=bool)

            first_flag = True

            '''
            Logic: Always use obs_next(t) and reward(t) to construct state(t+1), since obs_next(t) == obs(t+1).
            Note: The inital obs(0) == obs_next(-1) and reward(-1) are not recorded. So we have to initialize them.  
            '''
            while not all(flag_has_init) and len(live_mat) < self.window_size: 
                # 当flag_has_init全为True或live_mat大于等于window size时跳出循环
                # if not remove_recommended_ids and len(live_mat) >= self.window_size:
                #     break

                if is_obs or not first_flag:
                    live_id_prev = buffer.prev(index) != index
                    index = buffer.prev(index)
                else:
                    live_id_prev = np.ones_like(index, dtype=bool)

                first_flag = False
                # live_id_prev = buffer.prev(index) != index

                ind_init = ~live_id_prev & ~flag_has_init # 对应元素均为false左边才有True
                obs_prev = buffer[index].obs_next
                rew_prev = buffer[index].rew

                obs_prev[ind_init, 1] = self.num_item  # initialize obs # 修改第二列中ind_init为True的值为num_item
                rew_prev[ind_init] = 1  # todo: initialize reward.
                flag_has_init[ind_init] = True
                live_id_prev[ind_init] = True

                live_mat = np.vstack([live_mat, live_id_prev])
                obs_all = np.concatenate([obs_all, obs_prev])
                rew_all = np.concatenate([rew_all, rew_prev])

            # item_all_complete = np.expand_dims(obs_all[:, 1], -1)
            if len(live_mat) > self.window_size:
                live_mat = live_mat[:self.window_size, :]
                obs_all = obs_all[:len(index) * self.window_size, :]
                rew_all = rew_all[:len(index) * self.window_size]

            user_all = np.expand_dims(obs_all[:, 0], -1)
            item_all = np.expand_dims(obs_all[:, 1], -1)

            e_i = self.get_embedding(item_all, "action")
            # print("======================================================")
            # print("action embedding", e_i.shape)
            # print("======================================================")

            rew_matrix = rew_all.reshape((-1, 1))
            e_r = self.get_embedding(rew_matrix, "feedback")
            # print("======================================================")
            # print("reward", e_r.shape)
            # print("======================================================")

            if self.use_userEmbedding: # 没有用到，与论文sec5.4描述一致
                e_u = self.get_embedding(user_all, "user")
                print("======================================================")
                print("user embedding", e_u.shape)
                print("======================================================")
                s_t = torch.cat([e_u, e_i], dim=-1)
            else:
                s_t = e_i

            if is_train:
                r_max = self.train_max
                r_min = self.train_min
            else:
                r_max = self.test_max
                r_min = self.test_min

            if r_max is not None and r_min is not None:
                normed_r = (e_r - r_min) / (r_max - r_min)
                # if not (all(normed_r<=1) and all(normed_r>=0)):
                #     a = 1
                normed_r[normed_r>1] = 1 # todo: corresponding to the initialize reward line above.
                # assert (all(normed_r<=1) and all(normed_r>=0))

            else:
                normed_r = e_r

            if self.reward_handle == "mul":
                state_flat = s_t * normed_r
            elif self.reward_handle == "cat":
                state_flat = torch.cat([s_t, normed_r], 1)
            elif self.reward_handle == "cat2":
                state_flat = torch.cat([s_t, e_r], 1)
            else:
                state_flat = s_t

            state_cube = state_flat.reshape((-1, len(index), state_flat.shape[-1]))

            mask = torch.from_numpy(np.expand_dims(live_mat, -1)).to(self.device)
            state_masked = state_cube * mask

            state_sum = state_masked.sum(dim=0)
            state_final = state_sum / torch.from_numpy(np.expand_dims(live_mat.sum(0), -1)).to(self.device)

            # if remove_recommended_ids:
            #     recommended_ids = item_all_complete.reshape(-1, len(index)).T
            #     return state_final, recommended_ids
            # else:
            #     return state_final, None

            # print("======================================================")
            # print("state_final", state_final.shape)
            # print("======================================================")

            return state_final


class StateTracker_Caser(StateTracker_Base):
    def __init__(self, user_columns, action_columns, feedback_columns, dim_model, device,
                 use_userEmbedding=False, window_size=10, filter_sizes=[2, 3, 4], num_filters=16,
                 dropout_rate=0.1):
        super().__init__(user_columns=user_columns, action_columns=action_columns,
                         feedback_columns=feedback_columns, dim_model=dim_model, device=device,
                         window_size=window_size)

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate

        self.hidden_size = action_columns[0].embedding_dim
        self.num_item = action_columns[0].vocabulary_size

        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.final_dim = self.hidden_size + self.num_filters_total

        # Item embedding
        embedding_dict = torch.nn.ModuleDict(
            {"feat_item": torch.nn.Embedding(num_embeddings=self.num_item + 1,
                                             embedding_dim=self.hidden_size)})
        self.embedding_dict = embedding_dict.to(device)
        nn.init.normal_(self.embedding_dict.feat_item.weight, mean=0, std=0.1)

        # Horizontal Convolutional Layers
        self.horizontal_cnn = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (i, self.hidden_size)) for i in self.filter_sizes])
        # Initialize weights and biases
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        # Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(1, 1, (self.window_size, 1))
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        nn.init.constant_(self.vertical_cnn.bias, 0.1)

        # dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def convert_to_k_state_embedding(self, buffer=None, indices=None, obs=None, reset=None, is_obs=None):
        if reset:
            # users = np.expand_dims(obs[:, 0], -1)
            items = np.expand_dims(obs[:, 1], -1)

            # a = self.get_embedding(items_window, "action")

            e_i = self.get_embedding(items, "action")
            emb_state = e_i.repeat_interleave(self.window_size, dim=0).reshape([len(e_i), self.window_size, -1])

            mask = torch.zeros([emb_state.shape[0], emb_state.shape[1], 1], device=self.device)
            mask[:, 0, :] = 1
            emb_state *= mask

            # items_window = items.repeat(self.window_size, axis=1)

            return emb_state

        else:
            index = indices
            flag_has_init = np.zeros_like(index, dtype=bool)

            obs_all = np.zeros([0, 2], dtype=int)
            rew_all = np.zeros([0])
            live_mat = np.zeros([0, len(index)], dtype=bool)

            first_flag = True
            '''
                Logic: Always use obs_next(t) and reward(t) to construct state(t+1), since obs_next(t) == obs(t+1).
                Note: The inital obs(0) == obs_next(-1) and reward(-1) are not recorded. So we have to initialize them.  
            '''
            # while not all(flag_has_init) and len(live_mat) < self.window_size:
            while len(live_mat) < self.window_size:
                if is_obs or not first_flag:
                    live_id_prev = buffer.prev(index) != index
                    index = buffer.prev(index)
                else:
                    live_id_prev = np.ones_like(index, dtype=bool)

                first_flag = False
                # live_id_prev = buffer.prev(index) != index

                ind_init = ~live_id_prev & ~flag_has_init  # just dead and have not been initialized before.
                obs_prev = buffer[index].obs_next
                rew_prev = buffer[index].rew

                obs_prev[~live_id_prev, 1] = self.num_item
                rew_prev[~live_id_prev] = 1
                # obs_prev[ind_init, 1] = self.num_item
                # rew_prev[ind_init] = 1
                flag_has_init[ind_init] = True
                live_id_prev[ind_init] = True

                live_mat = np.vstack([live_mat, live_id_prev])

                obs_all = np.concatenate([obs_all, obs_prev])
                rew_all = np.concatenate([rew_all, rew_prev])

            # user_all = np.expand_dims(obs_all[:, 0], -1)
            item_all = np.expand_dims(obs_all[:, 1], -1)

            e_i = self.get_embedding(item_all, "action")

            s_t = e_i

            # state_flat = s_t * e_r
            state_flat = s_t
            state_cube = state_flat.reshape((-1, len(index), state_flat.shape[-1]))

            mask = torch.from_numpy(np.expand_dims(live_mat, -1)).to(self.device)
            state_masked = state_cube * mask

            emb_state = torch.swapaxes(state_masked, 0, 1)

            # items_window = np.swapaxes(item_all.reshape(-1,(len(index))), 0, 1)
            # emb_state * torch.ne(FLOAT(items_window), self.num_item).unsqueeze(-1) == emb_state

            return emb_state

    def forward(self, buffer=None, indices=None, obs=None, reset=None, is_obs=None, **kwargs):

        emb_state = self.convert_to_k_state_embedding(buffer, indices, obs, reset, is_obs)

        emb_state_final = emb_state.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(emb_state_final))
            h_out = h_out.squeeze(-1)
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(emb_state_final))
        v_flat = v_out.view(-1, self.hidden_size)

        state_hidden = torch.cat([h_pool_flat, v_flat], 1)
        state_hidden_dropout = self.dropout(state_hidden)

        return state_hidden_dropout


class StateTracker_GRU(StateTracker_Base):
    def __init__(self, user_columns, action_columns, feedback_columns, dim_model, device,
                 use_userEmbedding=False, window_size=10, gru_layers=1):
        super().__init__(user_columns=user_columns, action_columns=action_columns,
                         feedback_columns=feedback_columns, dim_model=dim_model, device=device,
                         window_size=window_size)

        self.hidden_size = action_columns[0].embedding_dim
        self.num_item = action_columns[0].vocabulary_size

        self.final_dim = self.hidden_size

        # Item embedding
        embedding_dict = torch.nn.ModuleDict(
            {"feat_item": torch.nn.Embedding(num_embeddings=self.num_item + 1,
                                             embedding_dim=self.hidden_size)})
        self.embedding_dict = embedding_dict.to(device)
        nn.init.normal_(self.embedding_dict.feat_item.weight, mean=0, std=0.1)

        # Horizontal Convolutional Layers
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )

    def convert_to_k_state_embedding(self, buffer=None, indices=None, obs=None, reset=None, is_obs=None):
        if reset:
            # users = np.expand_dims(obs[:, 0], -1)
            items = np.expand_dims(obs[:, 1], -1)

            e_i = self.get_embedding(items, "action")
            emb_state = e_i.repeat_interleave(self.window_size, dim=0).reshape([len(e_i), self.window_size, -1])

            len_states = np.ones([len(emb_state)]) # 对a*b*c的tensor,取出[:,0,:]

            emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb_state, len_states,
                                                                 batch_first=True, enforce_sorted=False)

            return emb_packed

        else:
            index = indices
            flag_has_init = np.zeros_like(index, dtype=bool)

            obs_all = np.zeros([0, 2], dtype=int)
            rew_all = np.zeros([0])
            live_mat = np.zeros([0, len(index)], dtype=bool)

            first_flag = True
            '''
                Logic: Always use obs_next(t) and reward(t) to construct state(t+1), since obs_next(t) == obs(t+1).
                Note: The inital obs(0) == obs_next(-1) and reward(-1) are not recorded. So we have to initialize them.  
            '''
            # while not all(flag_has_init) and len(live_mat) < self.window_size:
            while len(live_mat) < self.window_size:
                if is_obs or not first_flag:
                    live_id_prev = buffer.prev(index) != index
                    index = buffer.prev(index)
                else:
                    live_id_prev = np.ones_like(index, dtype=bool)

                first_flag = False
                # live_id_prev = buffer.prev(index) != index

                ind_init = ~live_id_prev & ~flag_has_init  # just dead and have not been initialized before.
                obs_prev = buffer[index].obs_next
                rew_prev = buffer[index].rew

                obs_prev[~live_id_prev, 1] = self.num_item
                rew_prev[~live_id_prev] = 1
                flag_has_init[ind_init] = True
                live_id_prev[ind_init] = True

                live_mat = np.vstack([live_mat, live_id_prev])

                obs_all = np.concatenate([obs_all, obs_prev])
                rew_all = np.concatenate([rew_all, rew_prev])

            # user_all = np.expand_dims(obs_all[:, 0], -1)
            item_all = np.expand_dims(obs_all[:, 1], -1)

            e_i = self.get_embedding(item_all, "action")
            s_t = e_i
            state_flat = s_t
            state_cube = state_flat.reshape((-1, len(index), state_flat.shape[-1]))

            mask = torch.from_numpy(np.expand_dims(live_mat, -1)).to(self.device)
            state_masked = state_cube * mask

            emb_state = torch.swapaxes(state_masked, 0, 1)

            len_states = mask.sum(0).squeeze(-1).cpu().numpy()

            emb_state_reverse = reverse_padded_sequence(emb_state, len_states)
            emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb_state_reverse, len_states,
                                                                 batch_first=True, enforce_sorted=False)
            return emb_packed

    def forward(self, buffer=None, indices=None, obs=None, reset=None, is_obs=None, **kwargs):

        emb_packed = self.convert_to_k_state_embedding(buffer, indices, obs, reset, is_obs)

        emb_packed_final, hidden = self.gru(emb_packed)
        hidden_final = hidden.view(-1, hidden.shape[2])

        return hidden_final


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0

        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys):
        """
        :param queries: A 3d tensor with shape of [N, T_q, C_q]
        :param keys: A 3d tensor with shape of [N, T_k, C_k]

        :return: A 3d tensor with shape of (N, T_q, C)

        """
        Q = self.linear_q(queries)  # (N, T_q, C)
        K = self.linear_k(keys)  # (N, T_k, C)
        V = self.linear_v(keys)  # (N, T_k, C)

        # Split and Concat
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5  # (h*N, T_q, T_k)

        # Key Masking
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_k)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (h*N, T_q, T_k)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  # (h*N, T_q, T_k)

        # Causality - Future Blinding
        diag_vals = torch.ones_like(matmul_output[0, :, :])  # (T_q, T_k)
        tril = torch.tril(diag_vals)  # (T_q, T_k)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)  # (h*N, T_q, T_k)
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings,
                                       matmul_output_m1)  # (h*N, T_q, T_k)

        # Activation
        matmul_output_sm = self.softmax(matmul_output_m2)  # (h*N, T_q, T_k)

        # Query Masking
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_q)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])  # (h*N, T_q, T_k)
        matmul_output_qm = matmul_output_sm * query_mask

        # Dropout
        matmul_output_dropout = self.dropout(matmul_output_qm)

        # Weighted Sum
        output_ws = torch.bmm(matmul_output_dropout, V_)  # ( h*N, T_q, C/h)

        # Restore Shape
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  # (N, T_q, C)

        # Residual Connection
        output_res = output + queries

        return output_res


class StateTracker_SASRec(StateTracker_Base):
    def __init__(self, user_columns, action_columns, feedback_columns, dim_model, device,
                 use_userEmbedding=False, window_size=10, dropout_rate=0.1, num_heads=1):
        super().__init__(user_columns=user_columns, action_columns=action_columns,
                         feedback_columns=feedback_columns, dim_model=dim_model, device=device,
                         window_size=window_size)

        self.hidden_size = action_columns[0].embedding_dim
        self.num_item = action_columns[0].vocabulary_size
        self.dropout_rate = dropout_rate

        self.final_dim = self.hidden_size

        # Item embedding
        embedding_dict = torch.nn.ModuleDict(
            {"feat_item": torch.nn.Embedding(num_embeddings=self.num_item + 1,
                                             embedding_dim=self.hidden_size)})
        self.embedding_dict = embedding_dict.to(device)
        nn.init.normal_(self.embedding_dict.feat_item.weight, mean=0, std=0.1)

        self.positional_embeddings = nn.Embedding(
            num_embeddings=window_size,
            embedding_dim=self.hidden_size
        )
        nn.init.normal_(self.positional_embeddings.weight, 0, 0.01)

        # Supervised Head Layers
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.ln_1 = nn.LayerNorm(self.hidden_size)
        self.ln_2 = nn.LayerNorm(self.hidden_size)
        self.ln_3 = nn.LayerNorm(self.hidden_size)
        self.mh_attn = MultiHeadAttention(self.hidden_size, self.hidden_size, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(self.hidden_size, self.hidden_size, dropout_rate)

    def convert_to_k_state_embedding(self, buffer=None, indices=None, obs=None, reset=None, is_obs=None):
        if reset:  # get user embedding
            # users = np.expand_dims(obs[:, 0], -1)
            items = np.expand_dims(obs[:, 1], -1)

            # a = self.get_embedding(items_window, "action")

            e_i = self.get_embedding(items, "action")
            emb_state = e_i.repeat_interleave(self.window_size, dim=0).reshape([len(e_i), self.window_size, -1])

            len_states = np.ones([len(emb_state)], dtype=int)

            inputs_emb = emb_state * self.hidden_size ** 0.5
            inputs_pos_emb = inputs_emb + self.positional_embeddings(torch.arange(self.window_size).to(self.device))
            seq = self.emb_dropout(inputs_pos_emb)

            # mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
            mask = torch.zeros([inputs_pos_emb.shape[0], inputs_pos_emb.shape[1], 1], device=self.device)
            mask[:, 0, :] = 1

            return seq, mask, len_states

        else:
            index = indices
            flag_has_init = np.zeros_like(index, dtype=bool)

            obs_all = np.zeros([0, 2], dtype=int)
            rew_all = np.zeros([0])
            live_mat = np.zeros([0, len(index)], dtype=bool)

            first_flag = True
            '''
                Logic: Always use obs_next(t) and reward(t) to construct state(t+1), since obs_next(t) == obs(t+1).
                Note: The inital obs(0) == obs_next(-1) and reward(-1) are not recorded. So we have to initialize them.  
            '''
            # while not all(flag_has_init) and len(live_mat) < self.window_size:
            while len(live_mat) < self.window_size:
                if is_obs or not first_flag:
                    live_id_prev = buffer.prev(index) != index
                    index = buffer.prev(index)
                else:
                    live_id_prev = np.ones_like(index, dtype=bool)

                first_flag = False
                # live_id_prev = buffer.prev(index) != index

                ind_init = ~live_id_prev & ~flag_has_init  # just dead and have not been initialized before.
                obs_prev = buffer[index].obs_next
                rew_prev = buffer[index].rew

                obs_prev[~live_id_prev, 1] = self.num_item
                rew_prev[~live_id_prev] = 1
                # obs_prev[ind_init, 1] = self.num_item
                # rew_prev[ind_init] = 1
                flag_has_init[ind_init] = True
                live_id_prev[ind_init] = True

                live_mat = np.vstack([live_mat, live_id_prev])

                obs_all = np.concatenate([obs_all, obs_prev])
                rew_all = np.concatenate([rew_all, rew_prev])

            # user_all = np.expand_dims(obs_all[:, 0], -1)
            item_all = np.expand_dims(obs_all[:, 1], -1)

            e_i = self.get_embedding(item_all, "action")

            s_t = e_i

            state_flat = s_t
            state_cube = state_flat.reshape((-1, len(index), state_flat.shape[-1]))

            mask = torch.from_numpy(np.expand_dims(live_mat, -1)).to(self.device)
            state_masked = state_cube * mask

            emb_state = torch.swapaxes(state_masked, 0, 1)

            len_states = mask.sum(0).squeeze(-1)
            mask = mask.swapaxes(0, 1)

            inputs_emb = emb_state * self.hidden_size ** 0.5
            inputs_pos_emb = inputs_emb + self.positional_embeddings(torch.arange(self.window_size).to(self.device))
            seq = self.emb_dropout(inputs_pos_emb)

            seq_reverse = reverse_padded_sequence(seq, len_states)

            return seq_reverse, mask, len_states

    def forward(self, buffer=None, indices=None, obs=None, reset=None, is_obs=None, **kwargs):

        seq, mask, len_states = self.convert_to_k_state_embedding(buffer, indices, obs, reset, is_obs)

        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out_masked = ff_out * mask
        ff_out_3 = self.ln_3(ff_out_masked)

        # state_final = ff_out_3[:, 0, :]
        state_final = extract_axis_1(ff_out_3, len_states - 1).squeeze(1)

        return state_final

class StateTrackerGRU(StateTracker_Base):
    def __init__(self, user_columns, action_columns, feedback_columns, dim_model, saved_embedding,
                 train_max=None, train_min=None, test_max=None, test_min=None, reward_handle="no",
                 device="cpu", use_userEmbedding=False, window_size=10, hidden_dim=32, num_layers=1, 
                 use_hidden=True, scratch=False):
        super(StateTrackerGRU, self).__init__(user_columns=user_columns, action_columns=action_columns,
                                               feedback_columns=feedback_columns, dim_model=dim_model, device=device,
                                               window_size=window_size)

        self.test_min = test_min
        self.test_max = test_max
        self.train_min = train_min
        self.train_max = train_max
        self.reward_handle = reward_handle

        assert saved_embedding is not None
        self.embedding_dict = saved_embedding.to(device)

        self.num_item = self.embedding_dict.feat_item.weight.shape[0] # KuaiEnv 3327
        self.dim_item = self.embedding_dict.feat_item.weight.shape[1] # KuaiEnv 41

        # Add a new embedding vector
        new_embedding = FLOAT(1, self.dim_model).to(device) # KuaiEnv 1*41
        nn.init.normal_(new_embedding, mean=0, std=0.01)
        emb_cat = torch.cat([self.embedding_dict.feat_item.weight.data, new_embedding]) # KuaiEnv 3328*41

        # whether train embedding from scratch
        self.scratch = scratch
        if self.scratch:
            nn.init.normal_(emb_cat, mean=0, std=0.01)

        new_item_embedding = torch.nn.Embedding.from_pretrained(
            emb_cat, freeze=not self.embedding_dict.feat_item.weight.requires_grad) 
        ## KuaiEnv freeze=False means the embs will change during training
        self.embedding_dict.feat_item = new_item_embedding

        self.use_userEmbedding = use_userEmbedding
        if self.use_userEmbedding:
            self.ffn_user = nn.Linear(compute_input_dim(self.user_columns), self.dim_model, device=self.device)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_hidden = use_hidden
        self.gru = nn.GRU(self.dim_item+1, self.dim_item+1, self.num_layers)

    def forward(self, buffer=None, indices=None, obs=None, reset=None, is_obs=None, is_train=True):

        if reset:  # get user embedding # used when trajectory is terminated

            users = np.expand_dims(obs[:, 0], -1)
            items = np.expand_dims(obs[:, 1], -1)

            # e_i = torch.ones(obs.shape[0], self.dim_item, device=self.device)
            # nn.init.normal_(e_i, mean=0, std=0.0001)

            e_i = self.get_embedding(items, "action")

            if self.use_userEmbedding: # 没有用到，与论文sec5.4描述一致
                e_u = self.get_embedding(users, "user")
                # s0 = self.ffn_user(e_u)
                s0 = torch.cat([e_u, e_i], dim=-1)
            else:
                s0 = e_i

            r0 = torch.ones(len(s0), 1).to(s0.device) # todo: define init reward as 1
            if self.reward_handle == "mul":
                state_res = s0 * 1
            elif self.reward_handle == "cat":
                state_res = torch.cat([s0, r0], 1)
            elif self.reward_handle == "cat2":
                state_res = torch.cat([s0, r0], 1)
            else:
                state_res = s0

            return state_res

        else: # used when trajectory is continuing
            index = indices
            flag_has_init = np.zeros_like(index, dtype=bool)

            obs_all = np.zeros([0, 2], dtype=int)
            rew_all = np.zeros([0])

            live_mat = np.zeros([0, len(index)], dtype=bool)

            first_flag = True

            '''
            Logic: Always use obs_next(t) and reward(t) to construct state(t+1), since obs_next(t) == obs(t+1).
            Note: The inital obs(0) == obs_next(-1) and reward(-1) are not recorded. So we have to initialize them.  
            '''
            while not all(flag_has_init) and len(live_mat) < self.window_size: 
                # 当flag_has_init全为True或live_mat大于等于window size时跳出循环
                # if not remove_recommended_ids and len(live_mat) >= self.window_size:
                #     break

                if is_obs or not first_flag:
                    live_id_prev = buffer.prev(index) != index
                    index = buffer.prev(index)
                else:
                    live_id_prev = np.ones_like(index, dtype=bool)

                first_flag = False
                # live_id_prev = buffer.prev(index) != index

                ind_init = ~live_id_prev & ~flag_has_init # 对应元素均为false左边才有True
                obs_prev = buffer[index].obs_next
                rew_prev = buffer[index].rew

                obs_prev[ind_init, 1] = self.num_item  # initialize obs # 修改第二列中ind_init为True的值为num_item
                rew_prev[ind_init] = 1  # todo: initialize reward.
                flag_has_init[ind_init] = True
                live_id_prev[ind_init] = True

                live_mat = np.vstack([live_mat, live_id_prev])
                obs_all = np.concatenate([obs_all, obs_prev])
                rew_all = np.concatenate([rew_all, rew_prev])

            # item_all_complete = np.expand_dims(obs_all[:, 1], -1)
            if len(live_mat) > self.window_size:
                live_mat = live_mat[:self.window_size, :]
                obs_all = obs_all[:len(index) * self.window_size, :]
                rew_all = rew_all[:len(index) * self.window_size]

            user_all = np.expand_dims(obs_all[:, 0], -1)
            item_all = np.expand_dims(obs_all[:, 1], -1)

            e_i = self.get_embedding(item_all, "action")

            rew_matrix = rew_all.reshape((-1, 1))
            e_r = self.get_embedding(rew_matrix, "feedback")

            if self.use_userEmbedding: # 没有用到，与论文sec5.4描述一致
                e_u = self.get_embedding(user_all, "user")
                s_t = torch.cat([e_u, e_i], dim=-1)
            else:
                s_t = e_i

            if is_train:
                r_max = self.train_max
                r_min = self.train_min
            else:
                r_max = self.test_max
                r_min = self.test_min

            if r_max is not None and r_min is not None:
                normed_r = (e_r - r_min) / (r_max - r_min)
                # if not (all(normed_r<=1) and all(normed_r>=0)):
                #     a = 1
                normed_r[normed_r>1] = 1 # todo: corresponding to the initialize reward line above.
                # assert (all(normed_r<=1) and all(normed_r>=0))

            else:
                normed_r = e_r

            if self.reward_handle == "mul":
                state_flat = s_t * normed_r
            elif self.reward_handle == "cat":
                state_flat = torch.cat([s_t, normed_r], 1)
            elif self.reward_handle == "cat2":
                state_flat = torch.cat([s_t, e_r], 1)
            else:
                state_flat = s_t

            state_cube = state_flat.reshape((-1, len(index), state_flat.shape[-1]))
            mask = torch.from_numpy(np.expand_dims(live_mat, -1)).to(self.device)
            state_masked = state_cube * mask

            state_masked, hidden = self.gru(state_masked)
            # kk = state_masked[-1,:,:] == hidden
            # print(kk.all()) # 12月16前设定(单层gru)下不用使用use_hidden这个超参

            if self.use_hidden: # use hidden of gru
                state_final = hidden.view(-1, hidden.shape[2])
            else: # use output of gru
                state_final = state_masked[-1,:,:]

            return state_final
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)

        # print('============================')
        # print(position.shape, div_term.shape, d_model)
        # print('============================')
        # pe[:, 0, 1::2] = torch.cos(position * div_term)

        pe[:,0,1::2]=torch.cos(position * div_term)[...,:pe.shape[-1]//2]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class StateTrackerAtt(StateTracker_Base):
    def __init__(self, user_columns, action_columns, feedback_columns, dim_model, saved_embedding,
                 train_max=None, train_min=None, test_max=None, test_min=None, reward_handle="no",
                 device="cpu", use_userEmbedding=False, window_size=10, scratch=False, 
                 dropout_rate=0.1, num_att_heads=1, num_att_layers=2, use_ckpt=False):
        super(StateTrackerAtt, self).__init__(user_columns=user_columns, action_columns=action_columns,
                                               feedback_columns=feedback_columns, dim_model=dim_model, device=device,
                                               window_size=window_size)

        self.test_min = test_min
        self.test_max = test_max
        self.train_min = train_min
        self.train_max = train_max
        self.reward_handle = reward_handle

        assert saved_embedding is not None
        self.embedding_dict = saved_embedding.to(device)

        self.num_item = self.embedding_dict.feat_item.weight.shape[0] # KuaiEnv 3327
        self.dim_item = self.embedding_dict.feat_item.weight.shape[1] # KuaiEnv 41

        # Add a new embedding vector
        new_embedding = FLOAT(1, self.dim_model).to(device) # KuaiEnv 1*41
        nn.init.normal_(new_embedding, mean=0, std=0.01)
        emb_cat = torch.cat([self.embedding_dict.feat_item.weight.data, new_embedding]) # KuaiEnv 3328*41

        # whether train embedding from scratch
        self.scratch = scratch
        if self.scratch:
            nn.init.normal_(emb_cat, mean=0, std=0.01)

        new_item_embedding = torch.nn.Embedding.from_pretrained(
            emb_cat, freeze=not self.embedding_dict.feat_item.weight.requires_grad) 
        ## KuaiEnv freeze=False means the embs will change during training
        self.embedding_dict.feat_item = new_item_embedding

        self.use_userEmbedding = use_userEmbedding
        if self.use_userEmbedding:
            self.ffn_user = nn.Linear(compute_input_dim(self.user_columns), self.dim_model, device=self.device)

        ## prepare Transformer Encoding layer 
        self.d_model = self.dim_item + 1
        self.num_heads = num_att_heads
        self.dim_ff = self.dim_item + 1
        self.dropout = dropout_rate
        self.num_layers = num_att_layers
        self.use_ckpt = use_ckpt
        self.attention_enc_layer = torch.nn.TransformerEncoderLayer(self.d_model, self.num_heads, self.dim_ff, 
                                                                    self.dropout)
        self.attention_enc = nn.TransformerEncoder(self.attention_enc_layer, self.num_layers)
        ## use positional encoding
        self.pe = PositionalEncoding(self.d_model, max_len=self.window_size)

    def forward(self, buffer=None, indices=None, obs=None, reset=None, is_obs=None, is_train=True):
        # print(reset)

        if reset:  # get user embedding # used when trajectory is terminated

            users = np.expand_dims(obs[:, 0], -1)
            items = np.expand_dims(obs[:, 1], -1)

            # e_i = torch.ones(obs.shape[0], self.dim_item, device=self.device)
            # nn.init.normal_(e_i, mean=0, std=0.0001)

            e_i = self.get_embedding(items, "action")

            if self.use_userEmbedding: # 没有用到，与论文sec5.4描述一致
                e_u = self.get_embedding(users, "user")
                # s0 = self.ffn_user(e_u)
                s0 = torch.cat([e_u, e_i], dim=-1)
            else:
                s0 = e_i

            r0 = torch.ones(len(s0), 1).to(s0.device) # todo: define init reward as 1
            if self.reward_handle == "mul":
                state_res = s0 * 1
            elif self.reward_handle == "cat":
                state_res = torch.cat([s0, r0], 1)
            elif self.reward_handle == "cat2":
                state_res = torch.cat([s0, r0], 1)
            else:
                state_res = s0

            return state_res

        else: # used when trajectory is continuing
            index = indices
            flag_has_init = np.zeros_like(index, dtype=bool)

            obs_all = np.zeros([0, 2], dtype=int)
            rew_all = np.zeros([0])

            live_mat = np.zeros([0, len(index)], dtype=bool)

            first_flag = True

            '''
            Logic: Always use obs_next(t) and reward(t) to construct state(t+1), since obs_next(t) == obs(t+1).
            Note: The inital obs(0) == obs_next(-1) and reward(-1) are not recorded. So we have to initialize them.  
            '''
            while not all(flag_has_init) and len(live_mat) < self.window_size: 
                # 当flag_has_init全为True或live_mat大于等于window size时跳出循环
                # if not remove_recommended_ids and len(live_mat) >= self.window_size:
                #     break

                if is_obs or not first_flag:
                    live_id_prev = buffer.prev(index) != index
                    index = buffer.prev(index)
                else:
                    live_id_prev = np.ones_like(index, dtype=bool)

                first_flag = False
                # live_id_prev = buffer.prev(index) != index

                ind_init = ~live_id_prev & ~flag_has_init # 对应元素均为false左边才有True
                obs_prev = buffer[index].obs_next
                rew_prev = buffer[index].rew

                obs_prev[ind_init, 1] = self.num_item  # initialize obs # 修改第二列中ind_init为True的值为num_item
                rew_prev[ind_init] = 1  # todo: initialize reward.
                flag_has_init[ind_init] = True
                live_id_prev[ind_init] = True

                live_mat = np.vstack([live_mat, live_id_prev])
                obs_all = np.concatenate([obs_all, obs_prev])
                rew_all = np.concatenate([rew_all, rew_prev])

            # item_all_complete = np.expand_dims(obs_all[:, 1], -1)
            if len(live_mat) > self.window_size:
                live_mat = live_mat[:self.window_size, :]
                obs_all = obs_all[:len(index) * self.window_size, :]
                rew_all = rew_all[:len(index) * self.window_size]

            user_all = np.expand_dims(obs_all[:, 0], -1)
            item_all = np.expand_dims(obs_all[:, 1], -1)

            e_i = self.get_embedding(item_all, "action")

            rew_matrix = rew_all.reshape((-1, 1))
            e_r = self.get_embedding(rew_matrix, "feedback")

            if self.use_userEmbedding: # 没有用到，与论文sec5.4描述一致
                e_u = self.get_embedding(user_all, "user")
                s_t = torch.cat([e_u, e_i], dim=-1)
            else:
                s_t = e_i

            if is_train:
                r_max = self.train_max
                r_min = self.train_min
            else:
                r_max = self.test_max
                r_min = self.test_min

            if r_max is not None and r_min is not None:
                normed_r = (e_r - r_min) / (r_max - r_min)
                # if not (all(normed_r<=1) and all(normed_r>=0)):
                #     a = 1
                normed_r[normed_r>1] = 1 # todo: corresponding to the initialize reward line above.
                # assert (all(normed_r<=1) and all(normed_r>=0))

            else:
                normed_r = e_r

            if self.reward_handle == "mul":
                state_flat = s_t * normed_r
            elif self.reward_handle == "cat":
                state_flat = torch.cat([s_t, normed_r], 1)
            elif self.reward_handle == "cat2":
                state_flat = torch.cat([s_t, e_r], 1)
            else:
                state_flat = s_t

            state_cube = state_flat.reshape((-1, len(index), state_flat.shape[-1]))

            mask = torch.from_numpy(np.expand_dims(live_mat, -1)).to(self.device)
            state_masked = state_cube * mask

            ## use attention
            if self.use_ckpt:
                state_pe = self.pe(state_masked)
                states = ckpt.checkpoint(self.attention_enc, state_pe, use_reentrant=False)
                ## after testing, use_reentrant=False or True make no difference
                ## but using False will not yield warning 
                state_final = states[-1,:,:]
            else:
                state_final = self.attention_enc(self.pe(state_masked))[-1,:,:]
                # state_final = self.attention_enc(state_masked)[-1,:,:]

            return state_final