# Consciousness Dynamics Experiment

## What This Is

A computational probe of metastability and integration metrics 
in large recurrent neural networks, examining whether 
consciousness-relevant dynamical properties emerge at criticality.

## The Core Question

Does integrated information (Φ) and normalized mutual information 
converge at the edge of chaos — and if so, are they measuring 
the same underlying property from different angles?

## Architecture

Five interacting modules — visual encoding, attention gating, 
memory recall, simulation, global workspace with feedback broadcast.
10,000 nodes. Sparse connectivity (1%). Tanh activations. 
Hebbian plasticity on memory connections.

## Metrics Tracked

- Lyapunov exponent — criticality measure
- Effective dimension — independent degrees of freedom
- Phi ratio — Poisson KL divergence approximation of integrated information
- Normalized mutual information — inter-subsystem information sharing

## Key Finding

At spectral radius ~1.0, Φ ratio and MI ratio track each other 
almost perfectly across a range of connectivity targets, 
suggesting convergence at criticality toward a shared underlying property.

## Open Question

What would a trajectory-level Φ look like — integrated information 
computed across a system's path through state space rather than 
at a fixed moment? This experiment suggests static Φ may miss 
temporal integration properties relevant to consciousness.

## Requirements

numpy, sklearn, matplotlib
