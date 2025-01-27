import torch
from typing import TypeVar

from policies import Policy

from pacbot_rs import PacmanGym
import models
from policies import MaxQPolicy
import safetensors.torch

P = TypeVar("P", bound=Policy)

# N/E/S/W and stay
NUM_ACTIONS = 5

def lerp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b

def load_qnetv2_from_file(filename: str) -> P:
    if not filename.endswith(".safetensors"):
        print("WARNING: load_model_from_file expects `.safetensors` files")

    obs_shape = PacmanGym({}).obs_numpy().shape
    q_net_old = models.QNetV2(obs_shape, NUM_ACTIONS).to("cpu")
    q_net_old.load_state_dict(safetensors.torch.load_file(filename))
    q_net_old.eval()
    policy = MaxQPolicy(q_net_old)

    return policy

LOADED_POLICIES = {}

def safetensors_qnet(filename: str, force_reload: bool=False) -> Policy:
    """
    Safe to call in a loop, uses cached policies
    """
    if not force_reload and filename in LOADED_POLICIES:
        return LOADED_POLICIES[filename]
    
    LOADED_POLICIES[filename] = load_qnetv2_from_file(filename)
    return LOADED_POLICIES[filename]

def reset_env(env: PacmanGym) -> None:
    env.reset()

    # to run a different AI for the first part of the game
    # while not env.first_ai_done():
    #     _, done = step_env(env, safetensors_qnet("checkpoints/q_net-old.safetensors"))
    #     if done:
    #         # the first ai died :( try again
    #         env.reset()

def step_env(env: PacmanGym, policy: Policy) -> tuple[int, bool]:
    obs = torch.from_numpy(env.obs_numpy()).to("cpu").unsqueeze(0)
    action_mask = torch.tensor(env.action_mask(), device="cpu").unsqueeze(0)
    return env.step(policy(obs, action_mask).item())