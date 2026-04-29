---
layout: ../../layouts/BlogPost.astro
title: "Deep Learning"
date: 2023-12-25
---

Before delving into the realm of deep learning, it's essential to understand its fundamental aspects: What is it, and what problems does it aim to solve?

## What is Deep Learning?

Deep learning can be defined in two ways: verbally and mathematically.

### Verbal Definition

Simply put, deep learning is the study of enabling computers or algorithms to learn from experience without being explicitly programmed.

Consider, for example, writing an algorithm to predict handwritten digits. If we were to manually write such an algorithm, we might rely on many if-else statements:

```python
if there is a curve on the top
    print("maybe it is 6, 3, 2, 9")

elif there are two lines
    print("maybe it is 1, 7")
```

This is just pseudocode—imagine having to write a complete algorithm like this! There must be a much better, more efficient, and more intelligent way of doing this. That’s where deep learning comes in.

Given sufficient training data, deep learning algorithms can learn underlying patterns and generalize to new data. For example, the algorithm might learn that if there are two curves on top of each other, the digit is most likely an eight. Therefore, even if it encounters a new handwritten eight, it should be able to predict the correct digit.

### Mathematical Definition

The main idea is that a neural network tries to approximate a function. Given a dataset of inputs and outputs, we assume there is an underlying function that produces the output values from the inputs. The goal of the neural network is to approximate this function as closely as possible.

Given some function $y = f^*(x)$ a neural network defines a mapping $y = f(x)$ which is a close approximation to $f^*(x)$.

This falls under the branch of mathematics known as function approximation. In fact, the Universal Approximation Theorem states that a neural network with an output layer and at least one hidden layer that uses a non-linear activation function can approximate any function. In other words, neural networks are universal approximators—they can be used to approximate virtually any function.

Now that we understand what deep learning is and what problems it solves, let’s move on to understanding forward propagation.

## Forward Propagation 

We all know the general flow of a neural network: inputs are multiplied by weights (and have biases added) to produce an output; then, a loss function measures the error between the prediction and the true value; finally, we backpropagate this error to adjust the weights and biases.

For the forward propagation part, I assume you already know the basics. However, to truly grasp the concept, it’s important to understand the rationale behind the following topics.

### Activation function

<div style="text-align:center;">
  <img src="/assets/images/feedforward_no_activation.svg" alt="feedforward no activation" style="display:inline-block;">
</div>

If we don't include activation (non-linear) functions, forward propagation would look like this:

$$
\begin{aligned}
z_1 &= w_1 \cdot x_1 + b \\
z_2 &= w_2 \cdot x_1 + b \\
z_3 &= (w_3 \cdot z_1 + w_5 \cdot z_2) + b \\
z_4 &= (w_4 \cdot z_1 + w_6 \cdot z_2) + b \\
o &= (w_7 \cdot z_3 + w_8 \cdot z_4) + b
\end{aligned}
$$

The final output is just a linear weighted sum in the form $y = mx + b$. Because a combination of linear functions remains linear—a straight line—the model would not be useful for learning complex, non-linear mappings between inputs and outputs, which are common in real-world scenarios.

### Bias 

Let’s consider the formula without a bias term:

$$z_1 = w_1 \cdot x_1$$

This is of the form $y = mx$, which is a straight line through the origin. Without bias, the activation function is restricted to passing through the origin. By using a bias term, we can shift the activation function, which is crucial for the model to learn more flexible representations.

## Backpropagation

Let’s recap. Inputs propagate forward through the hidden layers—being multiplied by weights and having biases added along the way—until they reach the output layer. 

### Loss Function

The loss function measures how well the model is performing, and it is this function that we aim to minimize during training.

#### Backpropagation or Gradient Descent or Automatic Differentiation? 

#### Backpropagation

Backpropagation is the process of calculating the gradients of a loss function with respect to the weights.

*Derivative and gradient*

The derivative of a function tells us how to change the input in order to increase or decrease the output, which helps us move closer to the function's minimum or maximum [source](https://machinelearningmastery.com/gradient-in-machine-learning/). In essence, the derivative at a point indicates the direction of steepest ascent; taking its negative gives the direction of steepest descent, which is what we want.

A gradient is simply the derivative of a multivariable function such as a loss function and is represented as a vector of partial derivatives. Thus, backpropagation is the algorithm that calculates these gradients. The gradient tells us the direction of steepest increase in the loss function. If we take the negative of the gradient, we get the direction of steepest decrease. This is the most efficient direction to move to reduce the loss quickly.

#### Gradient Descent

While backpropagation calculates the gradient (i.e., the direction and magnitude of steepest descent), gradient descent is the process of applying that gradient to the model parameters to move them toward the minimum of the loss function.

*Recap*

- **Forward Pass:** Inputs are multiplied by the weights.
- **Loss Calculation:** The loss function measures the error.
- **Backpropagation:** Gradients of the loss are computed wrt all the parameters using the backpropagation algorithm.
- **Parameter Update:** The weights are updated by the gradient in a process called gradient descent.

#### Automatic differentiation (AD)

There are several methods for computing derivatives: numerical differentiation, symbolic differentiation, and automatic differentiation (AD). The first two have disadvantages that make them less suitable for deep learning [source](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf). AD is an efficient, algorithmic way to compute derivatives during backpropagation, utilizing computational graphs to manage the calculations.

### Loss

#### Cross-Entropy Loss

$$L = -\sum_{i=1}^C y_i \cdot \log(p_i)$$

- Measures the difference between the true and predicted probabilities.
- $y_i$, 1 for the true class and 0 for the others (one-hot encoded).
- $p_i$, is softmax output, where the predicted probabilities for all classes sum to 1.
- $\log(p_i)$, $\log(1)$ is 0 and $\log(0)$ is a high negative number, therefore, it is a good function to calculate the loss.
- The negative is to cancel out any negative number generated by the $\log$ function.

The above is theoretical definition, however, in practice and in libraries like PyTorch, this is how cross-entropy is calculated:

- First compute log softmax,

$$\log(p_i) = p_i - \log\left(\sum \exp(p_i)\right)$$

- Then choose only the relevant class, negate it and sum them, and this value should be the same as the one you get using the theoretical formula

$$L = \sum -\log(p)[\text{target}]$$

Lets understand it with a simple example:

```python
# BS=2 and 3 classes
logits = [[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]]
targets = [0, 1]

# First we compute the log softmax
m = logits.exp() # (bs, 3)
ss = m.sum(dim=1) # (bs, 1)
log_softmax_output = logits - ss.log() # (bs, 3)

# Second we pick the output of only the relevant class, negate it and sum (across the batch)
cross_entropy_loss = -log_softmax_output[target].sum() # ()
```

There is also one additional trick used to make the logsoftmax numerically stable. Since exponentials of large numbers are very high, we just subtract the maximum of that sample from each element before taking the exponent.

$$\log(p_i) = (p_i - \max(p_i)) - \log\left(\sum \exp(p_i - \max(p_i))\right)$$

```python
# BS=2 and 3 classes
logits = [[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]]
targets = [0, 1]

# First we compute the log softmax
t = logits.max(dim=1, keepdim=True)
m = logits - t # (bs, 3)
e = m.exp() # (bs, 3)
ss = e.sum(dim=1, keepdim=True) # (bs, 1)
log_softmax_output = m - ss.log() # (bs, 3)

# Second we pick the output of only the relevant class, negate it and sum (across the batch)
cross_entropy_loss = -log_softmax_output[target].sum() # ()

# Taking the first sample
# t = 2
# m = [0, -1, -1.9]
# e = [1.0000, 0.3679, 0.1496]
# ss = [1.5174]
# log_softmax_output = [-0.4170, -1.4170, -2.3170]
# cross_entropy_loss = -[-0.4170] = 0.4170
```

### Optimizers

Algorithms like gradient descent, stochastic gradient descent, Adam are called optimizers. Optimizers are algorithms that adjust the parameters of a neural network to minimize a loss function. The formula for gradient descent is: 

$$\theta_{new} = \theta_{old} - \alpha \cdot \nabla J(\theta)$$
*(Where $\alpha$ is the learning rate and $\nabla J(\theta)$ is the gradient of the parameter).*

#### Gradient Descent, Stochastic Gradient Descent and Mini-batch Gradient Descent

Regular gradient descent computes the gradient using the entire dataset before making a parameter update. SGD updates parameters in batches with batch size equal to 1, and mini-batch GD updates the parameters in batches with batch size equal to $n$. I think SGD and mini-batch GD are used interchangeably.

#### GD with momentum

The formula is as follows:

$$
\begin{aligned}
v_{new} &= \gamma \cdot v_{old} + \nabla J(\theta) \\
\theta_{new} &= \theta_{old} - \alpha \cdot v_{new}
\end{aligned}
$$

As the name suggests, momentum used with GD boosts the step we take. Lets assume at each step we are going in one direction (downward), given this formula:

$$v_{new} = \gamma \cdot v_{old} + \nabla J(\theta)$$

Each step accumulates more and more momentum because we keep adding $\gamma \cdot v_{old}$ (momentum * old velocity) to the gradient. The result is that the updates get larger, making the optimization faster.
The value of $\gamma$ (momentum) is usually 0.9, which means we want to use 0.9 times the previous velocity. Think of it maybe like friction?

#### Adam

[Adam](https://paperswithcode.com/method/adam) builds on top of GD with momentum by adding a term ($v$) that adjusts the learning rate for each parameter.

$$
\begin{aligned}
m_{new} &= \beta_1 \cdot m_{old} + (1 - \beta_1) \cdot \nabla J(\theta) \quad \text{(same as momentum)} \\
v_{new} &= \beta_2 \cdot v_{old} + (1 - \beta_2) \cdot (\nabla J(\theta))^2 \\
\\
\theta_{new} &= \theta_{old} - \alpha \cdot \frac{m_{new}}{\sqrt{v_{new}} + \epsilon}
\end{aligned}
$$

If the gradients are consistently large, $v$​ will be large. If the gradients are small, $v$ will be small. By dividing by $\sqrt{v}$​, Adam makes sure that big updates get smaller and small updates get bigger.

