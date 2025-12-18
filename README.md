# Deep Reinforcement Learning for Image Manipulation

## Overview
This project applies **Deep Reinforcement Learning (DRL)** to automatically enhance images. An AI agent learns to perform image manipulation actions such as brightness adjustment, saturation control, and sharpening using a **custom Gym environment** and a **CNN-based policy network**.

---

## Approach
- **Environment:** Custom OpenAI Gym environment for image enhancement
- **Dataset:** STL10 (64×64 RGB images)
- **Actions:** Brighten, Darken, Increase/Decrease Saturation, Sharpen, No-op
- **Model:** CNN-based policy network
- **Algorithm:** Policy Gradient (REINFORCE)
- **Reward:** Image contrast measured via standard deviation

---

## Training
- Agent interacts with the environment over multiple episodes
- Actions sampled using softmax probabilities
- Policy updated using cumulative discounted rewards
- Visual comparison shown periodically (original vs enhanced image)

---

## Conclusion
The project demonstrates that **reinforcement learning can learn effective image enhancement strategies without labeled data**, highlighting its potential for automated photo editing and creative visual applications.

