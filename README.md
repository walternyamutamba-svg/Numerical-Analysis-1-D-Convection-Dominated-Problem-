# Numerical-Analysis-1-D-Convection-Dominated-Problem-
Interactive Streamlit app for visualizing the effect of epsilon on a singularly perturbed 1D convection-diffusion boundary value problem, including animated solution behavior and error analysis for central and upwind finite difference schemes.

Singularly Perturbed BVP Streamlit App

An interactive Streamlit application for exploring the numerical behavior of a **1D singularly perturbed convection-diffusion boundary value problem** under different values of the perturbation parameter `epsilon`.

The app is designed to clearly show:

- how the numerical solution changes as `epsilon` varies,
- the development of **boundary layers**,
- the behavior of **central difference** and **upwind difference** schemes,
- and how the **error changes** with respect to `epsilon`.

## Problem Overview

We study a one-dimensional convection-diffusion boundary value problem of the form:

\[
- \epsilon u''(x) + b u'(x) = 0, \quad x \in (a,c)
\]

with suitable boundary conditions.

For small values of `epsilon`, the problem becomes **convection-dominated**, and standard numerical methods may behave differently. In particular:

- the **central difference scheme** may develop oscillations,
- the **upwind scheme** is more stable,
- and the numerical error changes significantly as `epsilon` decreases.

## Features

This Streamlit app includes:

- **Interactive epsilon control**
- **Animated visualization** of the solution as epsilon changes
- Comparison of:
  - Exact solution
  - Central difference solution
  - Upwind difference solution
- **Error analysis plots**
- Stability interpretation using the **cell Péclet number**

## Why this app matters

This project helps visualize key numerical analysis concepts such as:

- singular perturbation,
- boundary layer behavior,
- stability of finite difference methods,
- effect of mesh size and perturbation parameter,
- comparison of numerical schemes.

It is useful for:

- numerical analysis learning,
- classroom demonstrations,
- exam preparation,
- and research/project presentation.

## Technologies Used

- **Python**
- **Streamlit**
- **NumPy**
- **Matplotlib**
- **Plotly**

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/singularly-perturbed-bvp-streamlit.git
cd singularly-perturbed-bvp-streamlit
