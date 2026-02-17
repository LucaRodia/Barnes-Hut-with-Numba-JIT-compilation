# Barnes-Hut N-Body Simulation with Numba

This project creates a high-performance simulation of gravitational interactions between a large number of bodies using the **Barnes-Hut algorithm**, optimized with **Numba JIT (Just-In-Time) compilation**.

The goal is to demonstrate how Python, typically considered slow for heavy computations, can achieve real-time performance in N-Body simulations by leveraging Data-Oriented Design and JIT compilation.

## 🚀 Performance & Features

* **High Efficiency:** Capable of simulating up to **50,000 bodies** in real-time.
* **Speed:** Achieves approximately **3.4ms per frame** with 20,000 bodies.
* **Visualization:** Real-time rendering using Matplotlib with dynamic energy monitoring.
* **Comparison:** The folder `Object Oriented Implementation/` contains a traditional recursive Python implementation. It struggles to simulate 1,000 bodies, demonstrating the improovements obtained by flattening data structures and using Numba.
