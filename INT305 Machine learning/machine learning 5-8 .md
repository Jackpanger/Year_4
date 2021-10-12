# Machine Learning 5-8

## Lecture 5

### Content

>TBD

### Neural network

**Neural Network**: without the brain stuff
(**Before**) Linear score function: $\quad f=W x$
(**Now**) 2-layer Neural Network  $\quad f=W_{2} \max \left(0, W_{1} x\right)$ 

or 3-layer Neural Network              $f=W_{3} \max \left(0, W_{2} \max \left(0, W_{1} x\right)\right)$



<img src="images\image-20211011165014240.png" alt="image-20211011165014240" style="zoom:80%;" />

#### Activation functions

<img src="images\image-20211011165146078.png" alt="image-20211011165146078" style="zoom:80%;" />

#### Neural network

<img src="images\image-20211011165204113.png" alt="image-20211011165204113" style="zoom:80%;" />

##### Example Feed-forward computation of a Neural Network 

```python
class Neuron:
 	#... 
	def neuron_tick(inputs):
	""" assume inputs and weights are $1-D$ numpy arrays and bias is a number """
    cell_body_sum = np.sum(inputs * self.weights) + self.bias
    firing_rate =1.0 /(1.0 + math.exp(-cell_body_sum)) # signoid activation function
    return firing_rate
```

<img src="images\image-20211011165949553.png" alt="image-20211011165949553"  />

```python
# forward-pass of a 3-layer neural network:
f = lambda x: 1.0 /(1.0 + np.exp(-x)) #activation function (use sigmoid)
x = np.random.randn(3,1) # randoin input vector of three numbers (3x1)
h1 = f(np.dot(W1,x) + b1) # calculate first hidden layer activations (4x1)
h2 = f(np.dot(W2,h1) + b2) # calculate second hidden layer activations (4x1)
out = np.dot(W3,h2) + b3 # output neuron (1x1)
```

#### Gradient Descent

Where we are...
$$
\begin{align*}
&s=f(x ; W)=W x \quad &\text { scores function } \\&L_{i}=\sum_{j \neq y_{i}} \max \left(0, s_{j}-s_{y_{i}}+1\right) & \text { SVM loss } \\&L=\frac{1}{N} \sum_{i=1}^{N} L_{i}+\sum_{k} W_{k}^{2} &\text { data loss + regularization } \\&\text { want } \nabla_{W} L
\end{align*}
$$

#### Computational Graph

<img src="images\image-20211011170927311.png" alt="image-20211011170927311" style="zoom:80%;" />

<img src="images\image-20211011170942228.png" alt="image-20211011170942228" style="zoom:80%;" />

<img src="images\image-20211011170949534.png" alt="image-20211011170949534" style="zoom:80%;" />

#### Example 1

<img src="images\image-20211011171346007.png" alt="image-20211011171346007" style="zoom:80%;" />

<img src="images\image-20211011171409386.png" alt="image-20211011171409386" style="zoom:80%;" />

<img src="images\image-20211011171418982.png" alt="image-20211011171418982" style="zoom:80%;" />

<img src="images\image-20211011171427629.png" alt="image-20211011171427629" style="zoom:80%;" />

<img src="images\image-20211011171435744.png" alt="image-20211011171435744" style="zoom:80%;" />

#### Chain rule

<img src="images\image-20211011171627662.png" alt="image-20211011171627662" style="zoom:80%;" />

**Another example:** $$f(w, x)=\frac{1}{1+e^{-\left(w_{0} x_{0}+w_{1} x_{1}+w_{2}\right)}}$$

<img src="images\image-20211011173117268.png" alt="image-20211011173117268"  />

<img src="images\image-20211011173152544.png" alt="image-20211011173152544"  />

<img src="images\image-20211011173213342.png" alt="image-20211011173213342"  />

<img src="images\image-20211011173236350.png" alt="image-20211011173236350"  />

<img src="images\image-20211011173301017.png" alt="image-20211011173301017"  />

<img src="images\image-20211011173717525.png" alt="image-20211011173717525"  />

#### Sigmoid

$$
\begin{align*}
f(w, x)=\frac{1}{1+e^{-\left(w_{0} x_{0}+w_{1} x_{1}+w_{2}\right)}} \quad \sigma(x)=\frac{1}{1+e^{-x}} \quad 
\text{sigmoid function}  \\ \frac{d \sigma(x)}{d x}=\frac{e^{-x}}{\left(1+e^{-x}\right)^{2}}=\left(\frac{1+e^{-x}-1}{1+e^{-x}}\right)\left(\frac{1}{1+e^{-x}}\right)=(1-\sigma(x)) \sigma(x)
\end{align*}
$$

<img src="images\image-20211011173919106.png" alt="image-20211011173919106"  />

#### Pattern in backward flow

<img src="images\image-20211011173959045.png" alt="image-20211011173959045"  />

### Exercise 1

Pooling units take $n$ values $x_{i}, \mathrm{i} \in[1, \mathrm{n}]$ and compute a scalar output whose value is invariant to permutations of the inputs. 

1. The Lp-pooling module takes positive inputs and computes $\mathrm{y}=\left(\sum_{i} x_{i}^{p}\right)^{\frac{1}{p}}$, assuming we know that $y^{\prime}=\frac{\partial L}{\partial y}$, what is $x_{i}^{\prime}=\frac{\partial L}{\partial x_{i}} ?$
   $$
   \begin{align*}
   x_{i}^{\prime}&=\frac{\partial L}{\partial x_{i}} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x} = y'\vdot \frac{\partial y} {\partial x} = y'\vdot\frac{1}{p}(\sum_i x^p_i)^{\frac{1-p}{p}}px_i^{p-1}\\
   &=y'x_i^{p-1}(\sum_i x_i^p)^{\frac{1-p}{p}}
   \end{align*}
   $$

2. The log-average module computes $\mathrm{y}=$ $\frac{1}{\beta} \ln \left(\frac{1}{n} \sum_{i} \exp \left(\beta x_{i}\right)\right)$, assuming we know that $y^{\prime}=\frac{\partial L}{\partial y}$, what is $x_{i}^{\prime}=\frac{\partial L}{\partial x_{i}}$ ?
   $$
   \begin{align*}
   x_{i}^{\prime}&=\frac{\partial L}{\partial x_{i}} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x} = y'\vdot \frac{\partial y} {\partial x} = y'\vdot\frac{1}{\beta}\frac{1}{\frac{1}{n}\sum_i\exp(\beta x_i)}\vdot\frac{1}{n}\vdot\exp(\beta x_i)\vdot \beta \\&= y'\vdot\frac{\exp(\beta x_i)}{\sum_i\exp(\beta x_i)}
   \end{align*}
   $$

#### Gradients for vector

<img src="images\image-20211011175901061.png" alt="image-20211011175901061"  />

<img src="images\image-20211011175921931.png" alt="image-20211011175921931"  />

<img src="images\image-20211011175955200.png" alt="image-20211011175955200"  />

<img src="images\image-20211011180008723.png" alt="image-20211011180008723"  />

A vectorized example: $f(x, W)=\|W \cdot x\|^{2}=\sum_{i=1}^{n}(W \cdot x)_{i}^{2}$$,x\in \mathbb{R}^{n}, W\in \mathbb{R}^{n \times n}$

<img src="images\image-20211011180206284.png" alt="image-20211011180206284"  />

<img src="images\image-20211011180303151.png" alt="image-20211011180303151"  />

<img src="images\image-20211012184040583.png" alt="image-20211012184040583" style="zoom:67%;" />
$$
\begin{align*}
q = W\vdot x = \sum_{i,j}w_{i,j}x_{j}, &\quad\text{size of }q \text{ is }(2,1)\\
q_k = q_i=\sum_{j}w_{i,j}x_{j} &\quad\text{size of }q_k \text{ is }(1,1)\\
\frac{\partial q_k}{\partial W_{i,j}} = \frac{\partial q_i}{\partial W_{i,j}}&=1_{k=i}x_{j} = x_j \\ 
\frac{\partial f}{\partial W_{i,j}}= \frac{\partial f}{\partial q_{i}} \frac{\partial q_i}{\partial W_{i,j}} &= 2q_ix_j
\end{align*}
$$
<img src="images\image-20211011180329875.png" alt="image-20211011180329875"  />

