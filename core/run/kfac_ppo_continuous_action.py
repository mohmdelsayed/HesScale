# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from functools import partial
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
import traceback

import gym
from mujoco_py.builder import MujocoException
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal

# from torch.utils.tensorboard import SummaryWriter

from backpack import extend
from core.logger import Logger
from hesscale.core.additional_losses import GaussianNLLLossMuPPO, GaussianNLLLossVarPPO
from hesscale.core.additional_activations import Exponential

class NanNetworkOutputError(Exception):
    'Raised when network output is nan. It is a proxy for divergence.'
    pass

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v2"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    torch_threads: int = 0
    optim: str = 'adam'
    delta: float = None
    eps: float = None
    kl_clip: float = None


def make_env(env_id, idx, capture_video, run_name, gamma, seed=0):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, use_extend=True):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_var = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
            Exponential(),
        )
        # self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

        if use_extend:
            self.actor_mean = extend(self.actor_mean)
            self.actor_var = extend(self.actor_var)
            self.critic = extend(self.critic)

    def get_value(self, x):
        val = self.critic(x)
        if torch.any(torch.isnan(val)):
            raise NanNetworkOutputError
        return val

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        # action_logstd = self.actor_logstd.expand_as(action_mean)
        # action_std = torch.exp(action_logstd)
        action_var = self.actor_var(x)
        if torch.any(torch.isnan(action_mean)) or torch.any(torch.isnan(action_var)):
            raise NanNetworkOutputError

        action_std = torch.sqrt(action_var)
        probs = Normal(action_mean, action_std.maximum(torch.tensor(1e-8)))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), action_mean, action_var


def main():
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # if args.track:
    #     import wandb

    #     wandb.init(
    #         project=args.wandb_project_name,
    #         entity=args.wandb_entity,
    #         sync_tensorboard=True,
    #         config=vars(args),
    #         name=run_name,
    #         monitor_gym=True,
    #         save_code=True,
    #     )
    # writer = SummaryWriter(f"runs/{run_name}")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma, seed=args.seed) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    use_extend = args.optim != 'kfac'
    agent = Agent(envs, use_extend=use_extend).to(device)
    if args.optim != 'kfac':
        from core.utils import optims as hes_optims
        optim_cls = hes_optims[args.optim]
        # optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    else:
        from KFAC_Pytorch.optimizers import KFACOptimizer
        optim_cls = KFACOptimizer

    optim_kwargs = {'lr': args.learning_rate,}
    if args.delta is not None: optim_kwargs.update({'delta': args.delta})
    if args.eps is not None: optim_kwargs.update({'eps': args.eps})
    if args.kl_clip is not None: optim_kwargs.update({'kl_clip': args.kl_clip})
    if args.optim != 'kfac':
        ac_mu_optimizer = optim_cls(agent.actor_mean.parameters(), **optim_kwargs)
        ac_var_optimizer = optim_cls(agent.actor_var.parameters(), **optim_kwargs)
        cr_optimizer = optim_cls(agent.critic.parameters(), **optim_kwargs)
    else:
        ac_mu_optimizer = optim_cls(agent.actor_mean, **optim_kwargs)
        ac_var_optimizer = optim_cls(agent.actor_var, **optim_kwargs)
        cr_optimizer = optim_cls(agent.critic, **optim_kwargs)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # losses
    ac_mu_lossf = GaussianNLLLossMuPPO(reduction="mean", eps=1e-6)
    ac_var_lossf = GaussianNLLLossVarPPO(reduction="mean", eps=1e-6)
    cr_lossf = nn.MSELoss()

    if use_extend:
        ac_mu_lossf = extend(ac_mu_lossf)
        ac_var_lossf = extend(ac_var_lossf)
        cr_lossf = extend(cr_lossf)

    ts = []
    return_per_episode = []
    logging_data = {
            'exp_name': args.exp_name,
            'task': args.env_id,
            'learner': 'ppo',
            'network': 'var_net',
            'optimizer': args.optim,
            'optimizer_hps': optim_kwargs,
            'n_samples': args.total_timesteps,
            'seed': args.seed,
    }
    logger = Logger(log_dir='/home/farrahi/scratch/HesScale/experiments/rl/logs')
    try:
        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value, _, _ = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, next_done, infos = envs.step(action.cpu().numpy())
                # next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                if next_done[0]:
                    next_obs = torch.Tensor(envs.reset()).to(device)
                    ep_ret = float(infos[0]['episode']['r'])
                    ts.append(global_step)
                    return_per_episode.append(ep_ret)
                    print(f"global_step={global_step}, episodic_return={ep_ret}")

                # if "final_info" in infos:
                #     for info in infos["final_info"]:
                #         if info and "episode" in info:
                #             print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                #             writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                #             writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue, newmu, newvar = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    # pg_loss1 = -mb_advantages * ratio
                    # pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    # pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    mb_probs = b_logprobs[mb_inds].exp().unsqueeze(1).detach()
                    mb_advantages = mb_advantages.unsqueeze(1).detach()
                    ac_mu_loss = ac_mu_lossf(newmu, newvar, b_actions[mb_inds], mb_probs, mb_advantages)
                    ac_var_loss = ac_var_lossf(newvar, newmu, b_actions[mb_inds], mb_probs, mb_advantages)

                    # Value loss
                    # newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        raise NotImplementedError
                        # v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        # v_clipped = b_values[mb_inds] + torch.clamp(
                        #     newvalue - b_values[mb_inds],
                        #     -args.clip_coef,
                        #     args.clip_coef,
                        # )
                        # v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        # v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        # v_loss = 0.5 * v_loss_max.mean()
                    else:
                        # v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                        v_loss = cr_lossf(newvalue, b_returns[mb_inds].unsqueeze(1).detach())

                    # entropy_loss = entropy.mean()
                    # loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    # optimizer.zero_grad()
                    # loss.backward()
                    # nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    # optimizer.step()

                    if args.optim == 'kfac' and ac_mu_optimizer.steps % ac_mu_optimizer.TCov == 0:
                        ac_mu_optimizer.acc_stats = True
                        ac_var_optimizer.acc_stats = True
                        cr_optimizer.acc_stats = True
                        ac_mu_optimizer.zero_grad()
                        ac_var_optimizer.zero_grad()
                        cr_optimizer.zero_grad()
                        with torch.no_grad():
                            value_noise = torch.randn(newvalue.size())

                        pg_fisher_loss = -newlogprob.mean()
                        sample_values = newvalue + value_noise
                        vf_fisher_loss = -(newvalue - sample_values.detach()).pow(2).mean()
                        fisher_loss = pg_fisher_loss + vf_fisher_loss
                        fisher_loss.backward(retain_graph=True)
                        ac_mu_optimizer.acc_stats = False
                        ac_var_optimizer.acc_stats = False
                        cr_optimizer.acc_stats = False

                    if args.optim != 'kfac':
                        # closures
                        def ac_mu_closure():
                            return ac_mu_loss, newmu.detach()
                        def ac_var_closure():
                            return ac_var_loss, newvar.detach()
                        def cr_closure():
                            return v_loss, newvalue.detach()

                        # step
                        ac_mu_optimizer.step(ac_mu_closure)
                        ac_var_optimizer.step(ac_var_closure)
                        cr_optimizer.step(cr_closure)
                    else:
                        # import pudb;pu.db
                        ac_mu_optimizer.zero_grad()
                        ac_mu_loss.backward()
                        ac_mu_optimizer.step()
                        ac_var_optimizer.zero_grad()
                        ac_var_loss.backward()
                        ac_var_optimizer.step()
                        cr_optimizer.zero_grad()
                        v_loss.backward()
                        cr_optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            # writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logging_data.update({
                'ts': ts,
                'returns': return_per_episode,
        })
    except (NanNetworkOutputError, MujocoException) as err:
        print(err)
        logging_data.update({'diverged': True})

    logger.log(**logging_data)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        # for idx, episodic_return in enumerate(episodic_returns):
        #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    # writer.close()

def signal_handler(msg, signal, frame):
    print('Exit signal: ', signal)
    args_str = msg[0]
    with open(f'timeout_ppo.txt', 'a') as f:
        f.write(f"{args_str} \n")
    exit(0)

if __name__ == "__main__":
    args_str = ' '.join(sys.argv)
    signal.signal(signal.SIGUSR1, partial(signal_handler, (args_str,)))
    try:
        main()
    except Exception as e:
        print(e)
        # with open(f"failed_ppo.txt", "a") as f:
        #     f.write(f"{cmd} \n")
        with open(f"failed_ppo_msgs.txt", "a") as f:
            f.write(f"{args_str} \n")
            f.write(f"{traceback.format_exc()} \n\n")
