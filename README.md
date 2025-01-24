# oo-ctrl-py
Object-oriented wrapper over multiple control frameworks for Python

Currently supported frameworks and controllers:
- NumPy
    - MPPI ([Williams et al.](https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf))

In closest future, the plan is to add:
- PyTorch
    - MPPI
- [do-mpc](https://github.com/do-mpc/do-mpc)
    - Discrete-time MPC

## Installation
We support Python 3.7 and above.

Currently, the package is not available on PyPI, so you need to install it from the source:
```bash
git clone https://github.com/planning-team/oo-ctrl-py.git
cd oo-ctrl-py
pip install -e .
```
or, in a simpler way:
```bash
pip install git+https://github.com/planning-team/oo-ctrl-py.git
```

## Usage
See the [examples](examples) directory for more information.
