import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageEnhance

import gymnasium as gym
from gymnasium import spaces

from tqdm import tqdm


# ---------------- POLICY NETWORK ---------------- #

class PolicyNetwork(nn.Module):
    def __init__(self, n_actions):
        super(PolicyNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ---------------- CUSTOM GYM ENV ---------------- #

class AdvancedImageEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, dataloader, max_steps=10):
        super().__init__()

        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)
        self.max_steps = max_steps
        self.current_step = 0

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, 64, 64), dtype=np.float32
        )

    def tensor_to_pil(self, tensor):
        return transforms.ToPILImage()(tensor.cpu())

    def pil_to_tensor(self, pil_image):
        return transforms.ToTensor()(pil_image)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        try:
            self.original_tensor, _ = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            self.original_tensor, _ = next(self.data_iter)

        self.current_tensor = self.original_tensor.clone()
        self.current_step = 0

        return self.current_tensor.squeeze(0), {}

    def step(self, action):
        pil_img = self.tensor_to_pil(self.current_tensor.squeeze(0))

        if action == 0:
            pil_img = ImageEnhance.Brightness(pil_img).enhance(1.1)
        elif action == 1:
            pil_img = ImageEnhance.Brightness(pil_img).enhance(0.9)
        elif action == 2:
            pil_img = ImageEnhance.Color(pil_img).enhance(1.2)
        elif action == 3:
            pil_img = ImageEnhance.Color(pil_img).enhance(0.8)
        elif action == 4:
            pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
        elif action == 5:
            pass  # No-op (Blur placeholder)

        self.current_tensor = self.pil_to_tensor(pil_img).unsqueeze(0)

        reward = self.current_tensor.std().item()
        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self.current_tensor.squeeze(0), reward, done, False, {}


# ---------------- TRAINING LOOP ---------------- #

def train(env, policy_net, optimizer, n_episodes=100):
    action_names = {
        0: "Brighten",
        1: "Darken",
        2: "Increase Saturation",
        3: "Decrease Saturation",
        4: "Sharpen",
        5: "No-op"
    }

    for episode in tqdm(range(n_episodes), desc="Training Episodes"):
        state, _ = env.reset()
        rewards, log_probs, actions_taken = [], [], []

        for _ in range(env.max_steps):
            state_tensor = state.unsqueeze(0)
            logits = policy_net(state_tensor)
            probs = torch.softmax(logits, dim=1)
            dist = Categorical(probs)

            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            actions_taken.append(action.item())

            state, reward, done, _, _ = env.step(action.item())
            rewards.append(reward)

            if done:
                break

        returns = []
        discounted_reward = 0

        for r in reversed(rewards):
            discounted_reward = r + 0.99 * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = [-lp * R for lp, R in zip(log_probs, returns)]

        optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 10 == 0:
            final_img = env.current_tensor.squeeze(0)

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(env.original_tensor.squeeze(0).permute(1, 2, 0))
            axs[0].set_title("Original")
            axs[0].axis("off")

            axs[1].imshow(final_img.permute(1, 2, 0))
            axs[1].set_title("After Enhancements")
            axs[1].axis("off")

            action_sequence = " → ".join(action_names[a] for a in actions_taken)
            plt.suptitle(f"Episode {episode + 1}\nActions: {action_sequence}")
            plt.show()


# ---------------- MAIN ---------------- #

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    print("Loading STL10 dataset...")
    dataset = datasets.STL10(
        root="./data",
        split="unlabeled",
        download=True,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    env = AdvancedImageEnv(dataloader, max_steps=10)
    policy_net = PolicyNetwork(n_actions=env.action_space.n)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    train(env, policy_net, optimizer, n_episodes=100)
