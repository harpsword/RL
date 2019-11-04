import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from envs import create_atari_env
from model import ActorCritic
from preprocess2015 import ProcessUnit


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def start_game(env, pu):
    no_op_frames = np.random.randint(1, 30)
    pu.clear()
    obs = env.reset()
    pu.step(obs)
    for i in range(no_op_frames):
        obs, r, done, _ = env.step(0)
        pu.step(obs)
        if done:
            return False
    return True

def train(rank, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.frameskip = 1
    env.seed(args.seed + rank)
    pu = ProcessUnit(4, args.frameskip)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    #print(optimizer)
    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    p = False
    while not p:
        p = start_game(env, pu)

    done = True
    episode_length = 0
    while True and counter.value < args.max_steps:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            reward = 0
            state = pu.to_torch_tensor()
            value, logit = model(state)
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            for idx in range(args.frameskip):
                episode_length += 1
                state, r__, done, _ = env.step(action.numpy())
                reward += r__
                pu.step(state)
                done = done or episode_length >= args.max_episode_length
                #reward = max(min(reward, 1), -1)
                with lock:
                    counter.value += 1

                if done:
                    episode_length = 0
                    p = False
                    while not p:
                        p = start_game(env, pu)
                    break

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            if done:
                break

        R = torch.zeros(1, 1)
        state = pu.to_torch_tensor()
        if not done:
            value, _ = model(state)
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
