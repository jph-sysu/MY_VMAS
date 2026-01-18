import torch
import numpy as np
import vmas
import matplotlib.pyplot as plt
from collections import deque
import time
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BalanceTask:
    """直接实现Balance任务的训练，不依赖RLlib"""
    
    def __init__(self, num_agents=3, device="cpu"):
        self.num_agents = num_agents
        self.device = device
        
        # 创建环境
        self.env = vmas.make_env(
            scenario="balance",
            num_envs=1,
            device=self.device,
            continuous_actions=True,
            max_steps=200,
            wrapper=None,
            dict_spaces=True,
        )
        
        # 获取环境信息
        self.agent_ids = list(self.env.observation_space.spaces.keys())
        self.obs_dim = self.env.observation_space[self.agent_ids[0]].shape[0]  # 16
        self.action_dim = self.env.action_space[self.agent_ids[0]].shape[0]    # 2
        
        print(f"Balance任务初始化完成")
        print(f"智能体数量: {self.num_agents}")
        print(f"观测维度: {self.obs_dim}")
        print(f"动作维度: {self.action_dim}")
        
        # 重置环境
        self.reset()
    
    def reset(self):
        """重置环境"""
        obs_dict = self.env.reset()
        # 转换为numpy并移除批量维度
        obs = {}
        for agent_id in self.agent_ids:
            agent_obs = obs_dict[agent_id]
            if isinstance(agent_obs, torch.Tensor):
                obs[agent_id] = agent_obs[0].cpu().numpy()
            else:
                obs[agent_id] = np.array(agent_obs)
        return obs
    
    def step(self, actions):
        """执行一步"""
        # 准备动作张量
        action_tensors = {}
        for agent_id in self.agent_ids:
            action = actions[agent_id]
            # 确保动作在[-1, 1]范围内
            action = np.clip(action, -1.0, 1.0)
            action_tensors[agent_id] = torch.tensor(action).unsqueeze(0).float()
        
        # 执行一步
        obs_dict, rewards, dones, infos = self.env.step(action_tensors)
        
        # 转换返回值
        obs = {}
        for agent_id in self.agent_ids:
            agent_obs = obs_dict[agent_id]
            if isinstance(agent_obs, torch.Tensor):
                obs[agent_id] = agent_obs[0].cpu().numpy()
            else:
                obs[agent_id] = np.array(agent_obs)
        
        # 计算总奖励
        if isinstance(rewards, dict):
            total_reward = sum([float(rewards[agent_id].item()) for agent_id in self.agent_ids])
        else:
            total_reward = float(rewards.item()) * self.num_agents
        
        # 检查是否结束
        if isinstance(dones, dict):
            done = any([bool(dones[agent_id].item()) for agent_id in self.agent_ids])
        else:
            done = bool(dones.item())
        
        return obs, total_reward, done, {}

class IPPOAgent:
    """独立PPO智能体"""
    
    def __init__(self, obs_dim, action_dim, agent_id, learning_rate=3e-4, gamma=0.99):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = learning_rate
        
        # 策略网络（actor）
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, action_dim),
        )
        
        # 价值网络（critic）
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1),
        )
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # 经验缓冲区
        self.memory = []
        
        # 用于计算动作标准差
        self.log_std = torch.nn.Parameter(torch.zeros(1, action_dim))
        
    def get_action(self, obs, deterministic=False):
        """根据观测选择动作"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        # 获取均值
        action_mean = self.policy_net(obs_tensor)
        
        if deterministic:
            return action_mean.detach().numpy().squeeze(0), None
        
        # 添加探索噪声
        action_std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        
        # 计算log概率
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.detach().numpy().squeeze(0), log_prob
    
    def compute_value(self, obs):
        """计算状态价值"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        return self.value_net(obs_tensor).squeeze()
    
    def store_transition(self, obs, action, log_prob, reward, next_obs, done):
        """存储转移"""
        self.memory.append({
            'obs': obs,
            'action': action,
            'log_prob': log_prob.detach() if log_prob is not None else torch.tensor(0.0),
            'reward': reward,
            'next_obs': next_obs,
            'done': done
        })
    
    def update(self, clip_param=0.2, value_coef=0.5, entropy_coef=0.01):
        """更新策略和价值网络"""
        if len(self.memory) < 32:  # 最小批量大小
            return 0, 0
        
        # 计算折扣回报和优势
        returns = []
        advantages = []
        
        # 计算每个时间步的回报
        R = 0
        for t in reversed(range(len(self.memory))):
            transition = self.memory[t]
            reward = transition['reward']
            done = transition['done']
            
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 标准化
        
        # 计算优势
        for i, transition in enumerate(self.memory):
            obs = transition['obs']
            value = self.compute_value(obs)
            advantage = returns[i] - value.detach()
            advantages.append(advantage)
        
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 准备批量数据 - 修复张量创建警告
        obs_list = [t['obs'] for t in self.memory]
        observations = torch.FloatTensor(np.array(obs_list))
        
        action_list = [t['action'] for t in self.memory]
        actions = torch.FloatTensor(np.array(action_list))
        
        old_log_probs = torch.stack([t['log_prob'] for t in self.memory])
        
        # 计算新log概率
        action_means = self.policy_net(observations)
        action_stds = torch.exp(self.log_std).expand_as(action_means)
        dist = torch.distributions.Normal(action_means, action_stds)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # 计算概率比
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        values = self.value_net(observations).squeeze()
        value_loss = 0.5 * torch.nn.functional.mse_loss(values, returns)
        
        # 熵正则化
        entropy = dist.entropy().mean()
        
        # 总损失
        total_policy_loss = policy_loss - entropy_coef * entropy
        total_value_loss = value_coef * value_loss
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
        
        # 更新价值网络
        self.value_optimizer.zero_grad()
        total_value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.value_optimizer.step()
        
        # 清空内存
        self.memory.clear()
        
        return policy_loss.item(), value_loss.item()

class CPPOAgent:
    """集中式PPO智能体（所有智能体共享策略）"""
    
    def __init__(self, num_agents, obs_dim, action_dim, learning_rate=3e-4, gamma=0.99):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = learning_rate
        
        # 集中式策略网络（输入所有智能体的观测）
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim * num_agents, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, action_dim * num_agents),
        )
        
        # 集中式价值网络
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim * num_agents, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1),
        )
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # 经验缓冲区
        self.memory = []
        
        # 动作标准差 - 为每个智能体单独设置
        self.log_std = torch.nn.Parameter(torch.zeros(num_agents, action_dim))
    
    def get_actions(self, obs_dict, deterministic=False):
        """获取所有智能体的动作"""
        # 拼接所有智能体的观测
        obs_list = []
        for i in range(self.num_agents):
            agent_id = f'agent_{i}'
            if agent_id in obs_dict:
                obs_list.append(obs_dict[agent_id])
            else:
                # 如果没有该智能体的观测，使用零向量
                obs_list.append(np.zeros(self.obs_dim))
        
        joint_obs = np.concatenate(obs_list)
        obs_tensor = torch.FloatTensor(joint_obs).unsqueeze(0)
        
        # 获取所有动作的均值
        action_means = self.policy_net(obs_tensor)
        # 重塑为 (batch_size, num_agents, action_dim)
        action_means = action_means.view(-1, self.num_agents, self.action_dim)
        
        actions = {}
        log_probs_list = []
        
        if deterministic:
            for i in range(self.num_agents):
                action = action_means[0, i].detach().numpy()
                actions[f'agent_{i}'] = action
            return actions, None
        
        # 为每个智能体采样动作
        for i in range(self.num_agents):
            # 获取该智能体的均值和标准差
            agent_mean = action_means[0, i]
            agent_log_std = self.log_std[i]
            agent_std = torch.exp(agent_log_std)
            
            # 创建分布并采样
            dist = torch.distributions.Normal(agent_mean, agent_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            
            # 存储动作和log概率
            actions[f'agent_{i}'] = action.detach().numpy()
            log_probs_list.append(log_prob)
        
        log_probs = torch.stack(log_probs_list) if log_probs_list else None
        return actions, log_probs
    
    def compute_value(self, obs_dict):
        """计算联合状态价值"""
        # 拼接所有智能体的观测
        obs_list = []
        for i in range(self.num_agents):
            agent_id = f'agent_{i}'
            if agent_id in obs_dict:
                obs_list.append(obs_dict[agent_id])
            else:
                obs_list.append(np.zeros(self.obs_dim))
        
        joint_obs = np.concatenate(obs_list)
        obs_tensor = torch.FloatTensor(joint_obs).unsqueeze(0)
        return self.value_net(obs_tensor).squeeze()
    
    def store_transition(self, obs_dict, actions, log_probs, reward, next_obs_dict, done):
        """存储转移"""
        if log_probs is not None:
            log_probs = log_probs.detach()
        self.memory.append({
            'obs_dict': obs_dict,
            'actions': actions,
            'log_probs': log_probs,
            'reward': reward,
            'next_obs_dict': next_obs_dict,
            'done': done
        })
    
    def update(self, clip_param=0.2, value_coef=0.5, entropy_coef=0.01):
        """更新网络"""
        if len(self.memory) < 32:
            return 0, 0
        
        # 计算折扣回报
        returns = []
        R = 0
        for t in reversed(range(len(self.memory))):
            transition = self.memory[t]
            reward = transition['reward']
            done = transition['done']
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 计算优势
        advantages = []
        for i, transition in enumerate(self.memory):
            obs_dict = transition['obs_dict']
            value = self.compute_value(obs_dict)
            advantage = returns[i] - value.detach()
            advantages.append(advantage)
        
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 准备数据
        old_log_probs_sum = torch.stack([t['log_probs'].sum() for t in self.memory])
        
        # 计算新log概率
        observations = []
        actions_list = []
        
        for transition in self.memory:
            obs_dict = transition['obs_dict']
            # 拼接观测
            obs_concat = []
            for i in range(self.num_agents):
                agent_id = f'agent_{i}'
                if agent_id in obs_dict:
                    obs_concat.append(obs_dict[agent_id])
                else:
                    obs_concat.append(np.zeros(self.obs_dim))
            observations.append(np.concatenate(obs_concat))
            
            # 拼接动作
            actions = transition['actions']
            action_concat = []
            for i in range(self.num_agents):
                agent_id = f'agent_{i}'
                if agent_id in actions:
                    action_concat.append(actions[agent_id])
                else:
                    action_concat.append(np.zeros(self.action_dim))
            actions_list.append(np.concatenate(action_concat))
        
        observations = torch.FloatTensor(np.array(observations))
        actions_tensor = torch.FloatTensor(np.array(actions_list))
        
        # 前向传播获取动作均值
        action_means_all = self.policy_net(observations)
        action_means_all = action_means_all.view(-1, self.num_agents, self.action_dim)
        actions_tensor = actions_tensor.view(-1, self.num_agents, self.action_dim)
        
        # 计算新log概率
        new_log_probs_list = []
        for i in range(len(observations)):
            log_prob_sum = 0
            for j in range(self.num_agents):
                agent_mean = action_means_all[i, j]
                agent_log_std = self.log_std[j]
                agent_std = torch.exp(agent_log_std)
                
                dist = torch.distributions.Normal(agent_mean, agent_std)
                log_prob = dist.log_prob(actions_tensor[i, j]).sum()
                log_prob_sum += log_prob
            new_log_probs_list.append(log_prob_sum)
        
        new_log_probs = torch.stack(new_log_probs_list)
        
        # PPO损失
        ratio = torch.exp(new_log_probs - old_log_probs_sum)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        values = self.value_net(observations).squeeze()
        value_loss = 0.5 * torch.nn.functional.mse_loss(values, returns)
        
        # 熵正则化
        entropy = 0
        for i in range(len(observations)):
            for j in range(self.num_agents):
                agent_mean = action_means_all[i, j]
                agent_log_std = self.log_std[j]
                agent_std = torch.exp(agent_log_std)
                
                dist = torch.distributions.Normal(agent_mean, agent_std)
                entropy += dist.entropy().mean()
        
        entropy = entropy / (len(observations) * self.num_agents)
        
        # 总损失
        total_policy_loss = policy_loss - entropy_coef * entropy
        total_value_loss = value_coef * value_loss
        
        # 更新 - 修复反向传播问题
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        
        # 分别计算梯度
        total_policy_loss.backward(retain_graph=True)
        total_value_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
        # 清空内存
        self.memory.clear()
        
        return policy_loss.item(), value_loss.item()

class MAPPOAgent:
    """MAPPO智能体（集中式Critic，分布式Actor）"""
    
    def __init__(self, num_agents, obs_dim, action_dim, learning_rate=3e-4, gamma=0.99):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = learning_rate
        
        # 为每个智能体创建独立的策略网络
        self.policy_nets = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(obs_dim, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, action_dim),
            ) for _ in range(num_agents)
        ])
        
        # 集中式价值网络（输入所有智能体的观测）
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim * num_agents, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1),
        )
        
        # 优化器
        self.policy_optimizers = [torch.optim.Adam(net.parameters(), lr=learning_rate) 
                                 for net in self.policy_nets]
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # 经验缓冲区
        self.memories = [[] for _ in range(num_agents)]
        
        # 动作标准差
        self.log_stds = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(1, action_dim)) for _ in range(num_agents)
        ])
    
    def get_actions(self, obs_dict, deterministic=False):
        """获取所有智能体的动作"""
        actions = {}
        log_probs = []
        
        for i in range(self.num_agents):
            agent_id = f'agent_{i}'
            obs = obs_dict[agent_id]
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            # 获取动作均值
            action_mean = self.policy_nets[i](obs_tensor)
            
            if deterministic:
                actions[agent_id] = action_mean.detach().numpy().squeeze(0)
                continue
            
            # 采样动作
            action_std = torch.exp(self.log_stds[i])
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            actions[agent_id] = action.detach().numpy().squeeze(0)
            log_probs.append(log_prob)
        
        if deterministic:
            return actions, None
        
        return actions, torch.stack(log_probs)
    
    def compute_value(self, obs_dict):
        """计算联合状态价值"""
        obs_list = [obs_dict[f'agent_{i}'] for i in range(self.num_agents)]
        joint_obs = np.concatenate(obs_list)
        obs_tensor = torch.FloatTensor(joint_obs).unsqueeze(0)
        return self.value_net(obs_tensor).squeeze()
    
    def store_transition(self, agent_idx, obs, action, log_prob, reward, next_obs, done):
        """为单个智能体存储转移"""
        self.memories[agent_idx].append({
            'obs': obs,
            'action': action,
            'log_prob': log_prob.detach() if log_prob is not None else torch.tensor(0.0),
            'reward': reward,
            'next_obs': next_obs,
            'done': done
        })
    
    def update(self, clip_param=0.2, value_coef=0.5, entropy_coef=0.01):
        """更新所有网络"""
        # 首先检查是否有足够的数据
        if not all(len(memory) >= 32 for memory in self.memories):
            return 0, 0
        
        # 假设所有智能体的经验长度相同
        episode_length = len(self.memories[0])
        
        # 计算折扣回报（使用集中式Critic）
        joint_returns = []
        R = 0
        
        for t in reversed(range(episode_length)):
            # 计算时间步t的总奖励
            total_reward = sum(self.memories[i][t]['reward'] for i in range(self.num_agents))
            done = any(self.memories[i][t]['done'] for i in range(self.num_agents))
            
            R = total_reward + self.gamma * R * (1 - int(done))
            joint_returns.insert(0, R)
        
        joint_returns = torch.FloatTensor(joint_returns)
        joint_returns = (joint_returns - joint_returns.mean()) / (joint_returns.std() + 1e-8)
        
        # 为每个时间步计算集中式价值
        joint_values = []
        for t in range(episode_length):
            obs_dict = {}
            for i in range(self.num_agents):
                obs_dict[f'agent_{i}'] = self.memories[i][t]['obs']
            value = self.compute_value(obs_dict)
            joint_values.append(value)
        
        joint_values = torch.stack(joint_values)
        
        # 计算优势，并分离计算图
        advantages = joint_returns - joint_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 更新价值网络
        value_loss = 0.5 * torch.nn.functional.mse_loss(joint_values, joint_returns)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.value_optimizer.step()
        
        # 更新每个智能体的策略网络
        policy_losses = []
        
        # 分离优势值，避免与价值网络的计算图关联
        advantages = advantages.detach()
        
        for i in range(self.num_agents):
            if len(self.memories[i]) >= 32:
                # 提取智能体i的经验
                memory = self.memories[i]
                
                # 准备数据 - 修复张量创建警告
                obs_list = [t['obs'] for t in memory]
                observations = torch.FloatTensor(np.array(obs_list))
                
                action_list = [t['action'] for t in memory]
                actions = torch.FloatTensor(np.array(action_list))
                
                old_log_probs = torch.stack([t['log_prob'] for t in memory])
                
                # 前向传播
                action_means = self.policy_nets[i](observations)
                action_std = torch.exp(self.log_stds[i])
                dist = torch.distributions.Normal(action_means, action_std)
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
                
                # PPO损失
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 熵正则化
                entropy = dist.entropy().mean()
                
                # 总损失
                total_loss = policy_loss - entropy_coef * entropy
                
                # 更新策略网络
                self.policy_optimizers[i].zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_nets[i].parameters(), 0.5)
                self.policy_optimizers[i].step()
                
                policy_losses.append(policy_loss.item())
        
        # 清空所有内存
        for memory in self.memories:
            memory.clear()
        
        return np.mean(policy_losses) if policy_losses else 0, value_loss.item()

def train_algorithm(algorithm_name, num_episodes=100):
    """训练指定算法 - 只在训练完成后保存一次结果"""
    print(f"\n{'='*60}")
    print(f"训练 {algorithm_name} 算法")
    print(f"{'='*60}")
    
    # 创建环境和智能体
    task = BalanceTask()
    num_agents = task.num_agents
    obs_dim = task.obs_dim
    action_dim = task.action_dim
    
    # 根据算法选择智能体类型
    if algorithm_name == "IPPO":
        # 每个智能体独立学习
        agents = [IPPOAgent(obs_dim, action_dim, i) for i in range(num_agents)]
        agent_type = "independent"
    elif algorithm_name == "CPPO":
        # 所有智能体共享一个策略
        agent = CPPOAgent(num_agents, obs_dim, action_dim)
        agent_type = "centralized"
    elif algorithm_name == "MAPPO":
        # MAPPO：分布式Actor，集中式Critic
        agent = MAPPOAgent(num_agents, obs_dim, action_dim)
        agent_type = "mappo"
    else:
        raise ValueError(f"未知算法: {algorithm_name}")
    
    # 训练统计
    episode_rewards = []
    avg_rewards = []
    loss_history = {'policy': [], 'value': []}
    
    # 训练循环
    start_time = time.time()
    for episode in range(num_episodes):
        # 重置环境
        obs = task.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        # 存储每个步骤的经验
        episode_experience = []
        
        while not done and step_count < 200:
            # 根据算法类型选择动作
            if algorithm_name == "IPPO":
                actions = {}
                log_probs = []
                
                for i, agent in enumerate(agents):
                    agent_id = f'agent_{i}'
                    action, log_prob = agent.get_action(obs[agent_id])
                    actions[agent_id] = action
                    log_probs.append(log_prob)
                
                # 执行动作
                next_obs, reward, done, _ = task.step(actions)
                
                # 存储经验（独立）
                for i, agent in enumerate(agents):
                    agent_id = f'agent_{i}'
                    agent.store_transition(
                        obs[agent_id], actions[agent_id], log_probs[i],
                        reward/num_agents, next_obs[agent_id], done
                    )
                
                # 定期更新
                if step_count % 32 == 0:
                    policy_losses, value_losses = [], []
                    for agent in agents:
                        pl, vl = agent.update()
                        if pl != 0:
                            policy_losses.append(pl)
                            value_losses.append(vl)
                    
                    if policy_losses:
                        loss_history['policy'].append(np.mean(policy_losses))
                        loss_history['value'].append(np.mean(value_losses))
            
            elif algorithm_name == "CPPO":
                # 集中式PPO
                actions, log_probs = agent.get_actions(obs)
                
                # 检查动作形状
                for agent_id, action in actions.items():
                    if action.shape != (action_dim,):
                        # 如果形状不正确，修正为零动作
                        actions[agent_id] = np.zeros(action_dim)
                
                # 执行动作
                next_obs, reward, done, _ = task.step(actions)
                
                # 存储经验（集中）
                if log_probs is not None:
                    agent.store_transition(obs, actions, log_probs, reward, next_obs, done)
                
                # 定期更新
                if step_count % 32 == 0:
                    pl, vl = agent.update()
                    if pl != 0:
                        loss_history['policy'].append(pl)
                        loss_history['value'].append(vl)
            
            elif algorithm_name == "MAPPO":
                # MAPPO
                actions, log_probs = agent.get_actions(obs)
                
                # 执行动作
                next_obs, reward, done, _ = task.step(actions)
                
                # 存储经验（分布式）
                for i in range(num_agents):
                    agent_id = f'agent_{i}'
                    agent.store_transition(
                        i, obs[agent_id], actions[agent_id], log_probs[i],
                        reward/num_agents, next_obs[agent_id], done
                    )
                
                # 定期更新
                if step_count % 32 == 0:
                    pl, vl = agent.update()
                    if pl != 0:
                        loss_history['policy'].append(pl)
                        loss_history['value'].append(vl)
            
            # 更新状态
            obs = next_obs
            total_reward += reward
            step_count += 1
        
        # 记录奖励
        episode_rewards.append(total_reward)
        
        # 计算滑动平均奖励
        if len(episode_rewards) >= 10:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_rewards.append(avg_reward)
        else:
            avg_reward = np.mean(episode_rewards)
            avg_rewards.append(avg_reward)
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"Reward = {total_reward:.2f}, "
                  f"Avg Reward = {avg_reward:.2f}, "
                  f"Steps = {step_count}")
    
    # 训练完成后保存一次结果
    end_time = time.time()
    training_time = end_time - start_time
    
    # 保存结果
    save_results(algorithm_name, episode_rewards, avg_rewards, loss_history, num_episodes, training_time)
    
    return episode_rewards, avg_rewards, loss_history

def save_results(algorithm_name, episode_rewards, avg_rewards, loss_history, num_episodes, training_time):
    """保存训练结果 - 每个算法只保存一个文件"""
    os.makedirs(f"./results/{algorithm_name}", exist_ok=True)
    
    # 保存数据到单个JSON文件
    results = {
        'algorithm': algorithm_name,
        'num_episodes': num_episodes,
        'training_time_seconds': training_time,
        'episode_rewards': episode_rewards,
        'avg_rewards': avg_rewards,
        'loss_history': loss_history,
        'final_avg_reward': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards),
        'best_reward': max(episode_rewards),
        'worst_reward': min(episode_rewards),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 保存到单个JSON文件
    with open(f"./results/{algorithm_name}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # 绘制图表
    plt.figure(figsize=(15, 10))
    
    # 1. 奖励曲线
    plt.subplot(2, 3, 1)
    plt.plot(episode_rewards, alpha=0.6, label='Episode Reward', color='blue')
    if avg_rewards:
        plt.plot(avg_rewards, 'r-', linewidth=2, label='Average Reward (window=10)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{algorithm_name} - Training Rewards\nFinal Avg: {results["final_avg_reward"]:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 损失曲线
    plt.subplot(2, 3, 2)
    if loss_history['policy']:
        plt.plot(loss_history['policy'], label='Policy Loss', color='green')
    if loss_history['value']:
        plt.plot(loss_history['value'], label='Value Loss', color='orange')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.title(f'{algorithm_name} - Training Losses')
    if loss_history['policy'] or loss_history['value']:
        plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 奖励分布直方图
    plt.subplot(2, 3, 3)
    if episode_rewards and len(episode_rewards) >= 20:
        last_n = min(100, len(episode_rewards))
        plt.hist(episode_rewards[-last_n:], bins=20, alpha=0.7, edgecolor='black', color='purple')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title(f'Reward Distribution (last {last_n} episodes)')
    
    # 4. 滑动平均曲线（更平滑）
    plt.subplot(2, 3, 4)
    if len(avg_rewards) > 20:
        window = 20
        smooth_avg = np.convolve(avg_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(avg_rewards)), smooth_avg, 'g-', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Smoothed Average Reward')
        plt.title(f'Smoothed Average (window={window})')
        plt.grid(True, alpha=0.3)
    
    # 5. 训练进度分析
    plt.subplot(2, 3, 5)
    if len(episode_rewards) >= 4:
        # 将训练分为4个阶段
        quarter = len(episode_rewards) // 4
        quarter_avgs = []
        quarter_labels = ['0-25%', '25-50%', '50-75%', '75-100%']
        
        for i in range(4):
            start_idx = i * quarter
            end_idx = (i + 1) * quarter if i < 3 else len(episode_rewards)
            quarter_avg = np.mean(episode_rewards[start_idx:end_idx])
            quarter_avgs.append(quarter_avg)
        
        bars = plt.bar(quarter_labels, quarter_avgs, alpha=0.7, color=['red', 'orange', 'yellow', 'green'])
        plt.xlabel('Training Phase')
        plt.ylabel('Average Reward')
        plt.title('Learning Progress by Phase')
        
        for bar, value in zip(bars, quarter_avgs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.2f}', ha='center', va='bottom')
    
    # 6. 训练信息总结
    plt.subplot(2, 3, 6)
    plt.axis('off')
    info_text = (
        f"Algorithm: {algorithm_name}\n"
        f"Total Episodes: {num_episodes}\n"
        f"Training Time: {training_time:.2f}s\n"
        f"Final Avg Reward: {results['final_avg_reward']:.2f}\n"
        f"Best Reward: {results['best_reward']:.2f}\n"
        f"Worst Reward: {results['worst_reward']:.2f}\n"
        f"Last 10 Avg: {np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else 'N/A'}\n"
        f"Timestamp: {results['timestamp']}"
    )
    plt.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{algorithm_name} Algorithm Training Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存单个PNG文件
    plt.savefig(f"./results/{algorithm_name}/training_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n训练完成! 结果已保存到: ./results/{algorithm_name}/")
    print(f"  - JSON文件: results.json")
    print(f"  - 图表文件: training_results.png")
    print(f"  - 训练时间: {training_time:.2f}秒")
    print(f"  - 最终平均奖励: {results['final_avg_reward']:.2f}")

def analyze_results(algorithms=None):
    """分析多种算法的结果"""
    if algorithms is None:
        algorithms = ["IPPO", "CPPO", "MAPPO"]
    
    print("\n" + "="*80)
    print("分析算法训练结果")
    print("="*80)
    
    results = {}
    
    # 加载各个算法的结果
    for algo in algorithms:
        try:
            # 加载固定文件名的结果文件
            result_file = f"./results/{algo}/results.json"
            with open(result_file, 'r') as f:
                data = json.load(f)
                results[algo] = data
                print(f"{algo}训练结果:")
                print(f"  训练回合数: {data['num_episodes']}")
                print(f"  训练时间: {data['training_time_seconds']:.2f}秒")
                print(f"  最终平均奖励: {data['final_avg_reward']:.2f}")
                print(f"  最佳回合奖励: {data['best_reward']:.2f}")
                print(f"  最差回合奖励: {data['worst_reward']:.2f}")
                print("-" * 40)
        except Exception as e:
            print(f"未找到{algo}结果文件或读取失败: {e}")
            results[algo] = None
    
    # 绘制比较图
    valid_results = {k: v for k, v in results.items() if v is not None}
    if len(valid_results) >= 2:
        # 创建图形，调整图形大小和布局
        fig = plt.figure(figsize=(16, 12))
        
        # 颜色和线型
        colors = {'IPPO': 'blue', 'CPPO': 'red', 'MAPPO': 'green'}
        
        # 1. 奖励曲线对比
        ax1 = plt.subplot(3, 3, 1)
        for algo, data in valid_results.items():
            rewards = data['episode_rewards']
            if rewards:
                plt.plot(rewards, alpha=0.6, label=f'{algo}', color=colors.get(algo, 'black'))
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward Curve Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 平滑奖励曲线对比
        ax2 = plt.subplot(3, 3, 2)
        window = 20
        
        for algo, data in valid_results.items():
            rewards = data['episode_rewards']
            if rewards and len(rewards) >= window:
                smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, len(rewards)), smooth, 
                        label=f'{algo} (smoothed)', linewidth=2, color=colors.get(algo, 'black'))
        
        plt.xlabel('Episode')
        plt.ylabel('Smoothed Reward')
        plt.title(f'Smoothed Reward Comparison (window={window})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 最终表现对比
        ax3 = plt.subplot(3, 3, 3)
        final_performance = []
        labels = []
        colors_list = []
        
        for algo, data in valid_results.items():
            final_avg = data['final_avg_reward']
            final_performance.append(final_avg)
            labels.append(algo)
            colors_list.append(colors.get(algo, 'gray'))
        
        if final_performance:
            bars = plt.bar(labels, final_performance, color=colors_list, alpha=0.7, edgecolor='black')
            plt.xlabel('Algorithm')
            plt.ylabel('Average Reward')
            plt.title('Final Performance Comparison')
            
            # 在柱状图上添加数值
            for bar, value in zip(bars, final_performance):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 训练时间对比
        ax4 = plt.subplot(3, 3, 4)
        training_times = []
        labels = []
        
        for algo, data in valid_results.items():
            training_times.append(data['training_time_seconds'])
            labels.append(algo)
        
        if training_times:
            bars = plt.bar(labels, training_times, color=colors_list, alpha=0.7, edgecolor='black')
            plt.xlabel('Algorithm')
            plt.ylabel('Training Time (seconds)')
            plt.title('Training Time Comparison')
            
            for bar, value in zip(bars, training_times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.1f}s', ha='center', va='bottom')
        
        # 5. 学习稳定性对比
        ax5 = plt.subplot(3, 3, 5)
        stabilities = []
        labels = []
        
        for algo, data in valid_results.items():
            rewards = data['episode_rewards']
            if rewards and len(rewards) >= 20:
                # 计算最后50%回合的方差（逆序，方差越小越稳定）
                half_len = len(rewards) // 2
                latter_half = rewards[half_len:]
                variance = np.var(latter_half)
                stabilities.append(1/(variance + 1e-6))  # 稳定性指标，方差越小值越大
                labels.append(algo)
        
        if stabilities:
            bars = plt.bar(labels, stabilities, color=colors_list, alpha=0.7, edgecolor='black')
            plt.xlabel('Algorithm')
            plt.ylabel('Stability (1/Variance)')
            plt.title('Learning Stability Comparison')
            
            for bar, value in zip(bars, stabilities):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.2f}', ha='center', va='bottom')
        
        # 6. 综合评分雷达图
        ax6 = plt.subplot(3, 3, (6, 9), polar=True)  # 占用两个位置，使雷达图更大
        
        if len(valid_results) >= 2:
            categories = ['Final Reward', 'Training Speed', 'Stability', 'Best Performance']
            N = len(categories)
            
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            for algo, data in valid_results.items():
                # 归一化评分 (0-1)
                final_score = min(1.0, data['final_avg_reward'] / 100)  # 假设最大奖励为100
                speed_score = min(1.0, 300 / max(data['training_time_seconds'], 1))  # 假设300秒为基准
                
                # 计算稳定性评分
                rewards = data['episode_rewards']
                if len(rewards) >= 20:
                    half_len = len(rewards) // 2
                    latter_half = rewards[half_len:]
                    stability = 1 - min(1.0, np.std(latter_half) / 50)  # 假设标准差50为最差
                else:
                    stability = 0.5
                
                best_score = min(1.0, data['best_reward'] / 150)  # 假设最佳奖励150为满分
                
                values = [final_score, speed_score, stability, best_score]
                values += values[:1]
                
                plt.plot(angles, values, linewidth=2, label=algo, color=colors.get(algo, 'black'))
                plt.fill(angles, values, alpha=0.1)
            
            plt.xticks(angles[:-1], categories)
            plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'])
            plt.title('Comprehensive Ability Radar Chart')
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 设置主标题和调整布局
        plt.suptitle('Multi-Algorithm Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 调整子图间距，减少上部空白
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, 
                           wspace=0.3, hspace=0.4)
        
        plt.savefig("./results/algorithm_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # 输出详细分析
        print("\n" + "="*80)
        print("算法性能详细分析")
        print("="*80)
        
        for algo, data in valid_results.items():
            rewards = data['episode_rewards']
            
            # 计算统计指标
            final_avg = data['final_avg_reward']
            best_episode = data['best_reward']
            worst_episode = data['worst_reward']
            training_time = data['training_time_seconds']
            
            if rewards and len(rewards) >= 20:
                std_dev = np.std(rewards[-20:])
            else:
                std_dev = np.std(rewards) if rewards else 0
            
            print(f"\n{algo}算法:")
            print(f"  最终平均奖励: {final_avg:.2f}")
            print(f"  最佳回合奖励: {best_episode:.2f}")
            print(f"  最差回合奖励: {worst_episode:.2f}")
            print(f"  稳定性(标准差): {std_dev:.2f}")
            print(f"  训练时间: {training_time:.2f}秒")
            
            # 判断学习趋势
            if len(rewards) >= 40:
                first_half = np.mean(rewards[:20])
                second_half = np.mean(rewards[-20:])
                improvement = second_half - first_half
                if improvement > 10:
                    trend = "显著上升"
                elif improvement > 0:
                    trend = "缓慢上升"
                elif improvement > -10:
                    trend = "基本稳定"
                else:
                    trend = "下降"
                print(f"  学习趋势: {trend} (改善: {improvement:.2f})")
        
        # 找出最佳算法
        best_algo = None
        best_score = -float('inf')
        
        for algo, data in valid_results.items():
            # 综合评分：最终表现 * 稳定性 / 训练时间
            final_avg = data['final_avg_reward']
            training_time = max(data['training_time_seconds'], 1)  # 避免除以0
            
            # 计算稳定性
            rewards = data['episode_rewards']
            if len(rewards) >= 20:
                half_len = len(rewards) // 2
                latter_half = rewards[half_len:]
                stability = 1 - min(1.0, np.std(latter_half) / 50)
            else:
                stability = 0.5
            
            score = final_avg * stability / training_time
            
            if score > best_score:
                best_score = score
                best_algo = algo
        
        if best_algo:
            print(f"\n🏆 综合最佳算法: {best_algo}")
            print(f"   综合评分: {best_score:.4f}")
            print(f"   最终奖励: {valid_results[best_algo]['final_avg_reward']:.2f}")
            print(f"   训练时间: {valid_results[best_algo]['training_time_seconds']:.2f}秒")
            
    return results

def train_all_algorithms(num_episodes=100):
    """训练所有三种算法并对比"""
    print("\n" + "="*80)
    print("训练所有三种算法: IPPO, CPPO, MAPPO")
    print("="*80)
    
    # 创建结果目录
    os.makedirs("./results", exist_ok=True)
    
    # 训练三种算法
    all_results = {}
    
    algorithms = ["IPPO", "CPPO", "MAPPO"]
    
    for algo in algorithms:
        print(f"\n开始训练 {algo} 算法...")
        episode_rewards, avg_rewards, loss_history = train_algorithm(algo, num_episodes=num_episodes)
        all_results[algo] = episode_rewards
    
    # 分析结果
    analyze_results(algorithms)
    
    # 输出大作业完成总结
    print("\n" + "="*80)
    print("\n📊 大作业完成总结")
    print("="*80)
    print(f"\n结果保存在: ./results/ 目录")
    print("\n每个算法包含:")
    print("  - results.json: 完整的训练结果数据")
    print("  - training_results.png: 训练结果图表")
    print("\n对比分析包含:")
    print("  - algorithm_comparison.png: 多算法对比图表")
    
    print("\n🔍 实验结果摘要:")
    
    for algo in algorithms:
        result_file = f"./results/{algo}/results.json"
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
                print(f"  {algo}:")
                print(f"    最终平均奖励: {data['final_avg_reward']:.2f}")
                print(f"    训练时间: {data['training_time_seconds']:.2f}秒")
                print(f"    最佳回合: {data['best_reward']:.2f}")
    
    print("\n📈 进一步改进建议:")
    print("1. 调整网络结构（层数、神经元数量）")
    print("2. 优化超参数（学习率、折扣因子、clip参数）")
    print("3. 增加智能体间通信机制")
    print("4. 使用更复杂的多智能体场景")
    print("5. 实现经验回放缓冲区（Replay Buffer）")
    print("6. 添加课程学习（Curriculum Learning）策略")

    return all_results

def main():
    """主函数"""
    print("VMAS Balance场景 - 直接实现MARL算法")
    print("=" * 80)
    
    # 创建结果目录
    os.makedirs("./results", exist_ok=True)
    
    print("\n选择操作:")
    print("1. 训练IPPO算法")
    print("2. 训练CPPO算法")
    print("3. 训练MAPPO算法")
    print("4. 训练三种算法并对比")
    print("5. 分析已有结果")
    
    try:
        choice = int(input("请输入选择 (1-5): "))
    except:
        choice = 4  # 默认训练所有算法并对比
    
    if choice == 1:
        print("\n训练IPPO算法...")
        episode_rewards, avg_rewards, loss_history = train_algorithm("IPPO", num_episodes=20000)
        
    elif choice == 2:
        print("\n训练CPPO算法...")
        episode_rewards, avg_rewards, loss_history = train_algorithm("CPPO", num_episodes=20000)
        
    elif choice == 3:
        print("\n训练MAPPO算法...")
        episode_rewards, avg_rewards, loss_history = train_algorithm("MAPPO", num_episodes=20000)
        
    elif choice == 4:
        print("\n训练三种算法并对比...")
        train_all_algorithms(num_episodes=20000)
        
    elif choice == 5:
        print("\n分析已有结果...")
        analyze_results()
        
    else:
        print("\n无效选择，默认训练所有算法并对比")
        train_all_algorithms(num_episodes=20000)

if __name__ == "__main__":
    main() 