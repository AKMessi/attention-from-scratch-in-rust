# 🧠 RustAttention - Bare-Metal Multi-Head Attention from Scratch in Rust

![Rust](https://img.shields.io/badge/language-Rust-orange)
![Zero Dependency](https://img.shields.io/badge/dependencies-0-green)

A zero-dependency, bare-metal implementation of the **Causal Multi-Head Scaled Dot-Product Attention** mechanism—the fundamental mathematical engine powering modern Large Language Models (LLMs) like GPT and Llama. 

Built entirely from scratch using pure, raw Rust, this project implements low-level linear algebra, memory layouts, and numerical stability algorithms without relying on `ndarray`, `torch`, or any external crates.

---

## 🛠️ Architectural Features Included

- **Flat Row-Major Memory Layout:** Implemented an optimized 1D contiguous vector allocation (`Vec<f32>`) with fast, inlined coordinate mapping functions to maximize CPU cache locality.
- **Cache-Aware Matrix Multiplication:** Designed a localized $O(n^3)$ `matmul` iteration sequence that minimizes cache misses across large dimensions.
- **Numerically Stable Softmax:** Guarded against catastrophic floating-point exponent overflow (`NaN` tracking) using the mathematical **Log-Sum-Exp / Max-Subtraction trick**.
- **GPT-Style Causal Masking:** Engineered a dynamic lower-triangular matrix modification algorithm (`apply_causal_mask`) that forces future token attention weights to exactly `0.0` by driving raw logits to $-10^9$.
- **Modular Multi-Head Scaled Attention:** Reused single-head primitives to execute simultaneous orthogonal context extractions before concatenating and projecting back into the hidden model space ($d_{\text{model}}$).

---

## 📐 Mathematical Framework Covered

This engine executes the exact transformation formula established in *Attention Is All You Need*:

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{Mask}\right)V
$$

---

## 📂 Repository Structure

```text
├── src/
│   ├── main.rs       # The verification test harness and execution script
│   ├── matrix.rs     # Bare-metal Matrix data structures & linear algebra math kernels
│   └── attention.rs  # Single-Head and Multi-Head Attention structural layers
├── Cargo.toml        # Zero-dependency package manifest
└── README.md         # Documentation
```

## 👤 Author

Built by [Aaryan](https://github.com/AKMessi)
