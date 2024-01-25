# autograd.cpp

A scalar value [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) engine in C++. (WIP)

## What is automatic differentiation?
Automatic differentiation is a computational technique used to efficiently compute derivates of functions. This process is vital in gradient-based optimization methods like stochastic gradient descent. There are two modes in auto differentiation, forward mode and reverse mode. Forward mode evaluates the intermediate variables and stores the expression tree (also called a computational graph) in memory. Then, in the reverse mode, we compute the partial derivates of the output w.r.t the intermediate variables.

![auto](https://se.mathworks.com/help/optim/ug/computational_graph.png)

## Motivation
The reason why I wanted to build this project was because I wanted learn the C++ language. Most high performance machine learning packages are written in C/C++, such as [pytorch](https://github.com/pytorch/pytorch) and [numpy](https://github.com/numpy/numpy). So I took it upon myself to create this project in order to understand the technology that drives these libraries.

## :handshake: Contributing
### Clone the repo
```bash
git clone https://github.com/<username>/autograd.cpp.git
cd autograd.cpp
```

### Submit a pull request

If you'd like to contribute, please fork the repository and open a pull request.

