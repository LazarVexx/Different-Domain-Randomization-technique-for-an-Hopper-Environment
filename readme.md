# Hopper Domain Randomization in Reinforcement Learning

**Author:** Luca Ianniello  
**Date:** January 17, 2025  
**Course:** Robot Learning – Politecnico of Turin  

## 📌 Overview

This project investigates various **Domain Randomization (DR)** techniques in the **Hopper environment** from OpenAI Gym, using two state-of-the-art reinforcement learning algorithms: **Soft Actor-Critic (SAC)** and **Proximal Policy Optimization (PPO)**. The core goal is to evaluate and compare the impact of different DR strategies on the generalization and robustness of learned policies.

The work builds on the classical **Uniform Domain Randomization (UDR)** and extends it by proposing and testing **five novel DR strategies**, focusing on dynamic adjustments of the randomization ranges during training.

---

## 🧠 Motivation

In Reinforcement Learning, **generalization** to unseen environments remains a critical challenge, especially when transferring from simulation to the real world—a phenomenon known as the **reality gap**. **Domain Randomization** tackles this by training agents across varying simulation parameters to improve robustness.

This project explores whether more structured or dynamic randomization techniques can outperform traditional UDR in promoting agent generalization.

---

## 🚀 Objectives

- Implement and train agents using **SAC** and **PPO** algorithms.
- Apply and compare **six domain randomization techniques** in the Hopper environment.
- Analyze the performance impact of each technique.
- Assess robustness and transferability across varying dynamics.

---

## 🧪 Domain Randomization Techniques

### 1. Uniform Domain Randomization (UDR)
- Applies static uniform sampling for body part masses (thigh, leg, foot).
- Uses three distinct ranges based on original mass values ±0.2.
- Simple and fixed randomization across the whole training.

### 2. Reducing Ranged Domain Randomization (RRDR)
- Begins with large randomization ranges, gradually shrinking over time.
- Masses sampled from a **normal distribution**, focusing over time.
- Mimics a reverse-curriculum learning strategy.

### 3. Incremental Ranges Expansion (IRE)
- Starts with narrow UDR-like ranges.
- Expands the **upper bound** logarithmically throughout training.
- Masses sampled from a normal distribution with decreasing variance.

### 4. Exploration UDR (EUDR)
- Combines IRE → RRDR → UDR in 3 training phases:
  - **Exploration phase** (IRE, high entropy).
  - **Controlled adaptation phase** (RRDR, medium entropy).
  - **Exploitation phase** (UDR, low entropy).
- Entropy coefficients are adjusted based on training phase and algorithm.

### 5. Dynamic Range Cycle (DRC)
- Cycles between UDR → IRE → RRDR.
- 20% → 40% → 40% phase split.
- Promotes regular exploration–exploitation shifts during training.
- Entropy dynamically tuned across cycles.

### 6. Dynamic Exploration Domain Randomization (DEDR)
- Uses the sequence RRDR → UDR → IRE (40% → 30% → 30%).
- Early exploration, mid stability, late-stage adaptation.
- Balanced entropy scheduling across phases.

---

## 🛠 Technologies

- **Language**: Python 3.10+
- **Frameworks**:  
  - [PyTorch](https://pytorch.org/)  
  - [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
  - [OpenAI Gym](https://www.gymlibrary.dev/)  
- **Environment**: Custom Hopper (based on MuJoCo)

