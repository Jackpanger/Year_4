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

## Lecture 6

### Content

>TBD

### The problem

<img src="images\image-20211018163747766.png" alt="image-20211018163747766"  />

### Challenges

Challenges: Viewpoint Variation 

<img src="images\image-20211018163843534.png" alt="image-20211018163843534"  />

**Challenges**: Illumination

<img src="images\image-20211018163916058.png" alt="image-20211018163916058"  />

**Challenges**: Deformation

<img src="images\image-20211018163945273.png" alt="image-20211018163945273"  />

**Challenges**: Occlusion

<img src="images\image-20211018164009661.png" alt="image-20211018164009661"  />

**Challenges**: Background Clutter

<img src="images\image-20211018164032839.png" alt="image-20211018164032839"  />

**Challenges**: Interclass variation

<img src="images\image-20211018164058012.png" alt="image-20211018164058012"  />

### An image classifier

```python
def predict(image):
    # ???
	return class__labet
```

Unlike e.g. sorting a list of numbers,
**no obvious way** to hard-code the algorithm for recognizing a cat, or other classes.

### Attempts

Attempts have been made

<img src="images\image-20211018164300538.png" alt="image-20211018164300538"  />

<img src="images\image-20211018164317155.png" alt="image-20211018164317155"  />

### Data-driven approach

**Data-driven approach:**

1. Collect a dataset of images and labels
2. Use Machine Learning to train an image classifier
3. Evaluate the classifier on a withheld set of test images

```python
def train(train_images, train_labels):
	# build a model for images -> labels...
	return model
def predict(model, test_images):
	# predict test_labeIs using the model...
	return test_labels
```

<img src="images\image-20211018164445056.png" alt="image-20211018164445056" style="zoom:150%;" />

### CNN

Convolutional Neural Networks

<img src="images\image-20211018164515917.png" alt="image-20211018164515917"  />

<img src="images\image-20211018164539066.png" alt="image-20211018164539066" style="zoom:80%;" />

<img src="images\image-20211018164556747.png" alt="image-20211018164556747" style="zoom:80%;" />

<img src="images\image-20211018164610994.png" alt="image-20211018164610994" style="zoom:80%;" />

<img src="images\image-20211018164633341.png" alt="image-20211018164633341" style="zoom:80%;" />

<img src="images\image-20211018164718597.png" alt="image-20211018164718597" style="zoom:80%;" />

For example, if we had $65 \times 5$ filters, we'll get 6 separate activation maps:

<img src="images\image-20211018164915779.png" alt="image-20211018164915779" style="zoom:80%;" />

We stack these up to get a "new image" of size $28 \times 28 \times 6 !$

Preview: ConvNet is a sequence of Convolution Layers, interspersed with activation functions

<img src="images\image-20211018164951056.png" alt="image-20211018164951056" style="zoom:80%;" />

<img src="images\image-20211018165007989.png" alt="image-20211018165007989" style="zoom:80%;" />

#### **Preview**

<img src="images\image-20211018165039487.png" alt="image-20211018165039487"  />

<img src="images\image-20211018165058252.png" alt="image-20211018165058252"  />

<img src="images\image-20211018165117629.png" alt="image-20211018165117629"  />

<img src="images\image-20211018165130595.png" alt="image-20211018165130595"  />

A closer look at spatial dimensions:

<img src="images\image-20211018165224027.png" alt="image-20211018165224027"  />

A closer look at spatial dimensions:

<img src="images\image-20211018165422309.png" alt="image-20211018165422309" style="zoom:80%;" />

<img src="images\image-20211018165445015.png" alt="image-20211018165445015" style="zoom:80%;" />

<img src="images\image-20211018165534690.png" alt="image-20211018165534690" style="zoom:80%;" />

<span style="color:red">$7 \times 7$ input (spatially) assume $3 \times 3$ filter applied with stride $3 ?$</span>
doesn't fit! cannot apply $3 \times 3$ filter on $7 \times 7$ input with stride 3 .

Output size:
**$(N-F) /$ stride $+1$**
e.g. $N=7, F=3$ :
stride $1=>(7-3) / 1+1=5$
stride $2=>(7-3) / 2+1=3$
stride $3=>(7-3) / 3+1=2.33$

#### **In practice: Common to zero pad the border**

<img src="images\image-20211018165736671.png" alt="image-20211018165736671"  />

e.g. input $7 \times 7$
$3 \times 3$ filter, applied with **stride 1**
**pad with 1 pixel** border $=>$ what is the output?
**$7 \times 7$ output!**

in general, common to see CONV layers with stride 1, filters of size FxF, and zero-padding with $(F-1) / 2$. (will preserve size spatially)
e.g. $F=3=>$ zero pad with 1
$F=5=>$ zero pad with 2
$F=7 \Rightarrow$ zero pad with 3

**Remember back to...**
E.g. $32 \times 32$ input convolved repeatedly with $5 \times 5$ filters shrinks volumes spatially! (32 -> 28 -> $24 \ldots$..). Shrinking too fast is not good, doesn't work well.

<img src="images\image-20211018165902205.png" alt="image-20211018165902205"  />

Examples time:
Input volume: $32 \times 32 \times 3$
$105 \times 5$ filters with stride 1, pad 2
Output volume size:
$\left(32+2^{*} 2-5\right) / 1+1=32$ spatially, so $32 \times 32 \times 10$

Examples time:
Input volume: $32 \times 32 \mathbf{x} 3$
$105 \times 5$ filters with stride 1, pad 2
Number of parameters in this layer?
each filter has $5^{*} 5^{\star} 3+1=76$ params $\Rightarrow 76^{*} 10=760$

#### **Summary**

To summarize, the Conv Layer:

- Accepts a volume of size $W_{1} \times H_{1} \times D_{1}$

- Requires four hyperparameters:
  - Number of filters $K$,

  - their spatial extent $F$,

  - the stride $S$,

  - the amount of zero padding $P$.

    >Common settings:
    >K= powers of 2, e.g.  32,64,128,512) 
    >
    >+ F=3, S=1, P=1 
    >
    >+ F=5, S=1, P=2 
    >+ F=5, S=2, P=?  (whatever fits)
    >+ F=1, S=1, P=0 

- Produces a volume of size $W_{2} \times H_{2} \times D_{2}$ where:

  - $W_{2}=\left(W_{1}-F+2 P\right) / S+1$
  - $H_{2}=\left(H_{1}-F+2 P\right) / S+1$ (i.e. width and height are computed equally by symmetry)
  - $D_{2}=K$

- With parameter sharing, it introduces $F \cdot F \cdot D_{1}$ weights per filter, for a total of $\left(F \cdot F \cdot D_{1}\right) \cdot K$ weights and $K$ biases.

- In the output volume, the $d$-th depth slice (of size $\left.W_{2} \times H_{2}\right)$ is the result of performing a valid convolution of the $d$-th filter over the input volume with a stride of $S$, and then offset by $d$-th bias.

(btw, $1 \times 1$ convolution layers make perfect sense)

<img src="images\image-20211018170437642.png" alt="image-20211018170437642"  />

<img src="images\image-20211018170832888.png" alt="image-20211018170832888"  />

<img src="images\image-20211018170841760.png" alt="image-20211018170841760"  />

The brain/neuron view of CONV Layer

<img src="images\image-20211018170914030.png" alt="image-20211018170914030"  />

<img src="images\image-20211018170942293.png" alt="image-20211018170942293"  />

<img src="images\image-20211018171036484.png" alt="image-20211018171036484"  />

<img src="images\image-20211018171053627.png" alt="image-20211018171053627"  />

#### Pooling layer

- makes the representations smaller and more manageable 
- operates over each activation map independently:

<img src="images\image-20211018171158100.png" alt="image-20211018171158100"  />

##### MAX POOLING

<img src="images\image-20211018171227529.png" alt="image-20211018171227529" style="zoom:80%;" />

- Accepts a volume of size $W_{1} \times H_{1} \times D_{1}$
- Requires three hyperparameters:
  - their spatial extent $F$,

  - the stride $S$.

    >Common settings:
    >
    >F=2, S=2 
    >F=3, S=2
- Produces a volume of size $W_{2} \times H_{2} \times D_{2}$ where:
  - $W_{2}=\left(W_{1}-F\right) / S+1$
  - $H_{2}=\left(H_{1}-F\right) / S+1$
  - $D_{2}=D_{1}$
- Introduces zero parameters since it computes a fixed function of the input
- Note that it is not common to use zero-padding for Pooling layers

### Fully Connected Layer (FC layer)

- Contains neurons that connect to the entire input volume, as in ordinary Neural Networks

#### Case Study: LeNet-5

![image-20211018171747101](C:\Users\ADMIN\AppData\Roaming\Typora\typora-user-images\image-20211018171747101.png)

#### Case Study: AlexNet 

[Krizhevsky etal. 2012]

<img src="images\image-20211018171809893.png" alt="image-20211018171809893"  />

Input: $227 \times 227 \times 3$ images
**First layer** (CONV1): $9611 \times 11$ filters applied at stride 4 =>
<span style="color:blue">$\mathrm{Q}$ : what is the output volume size? Hint: $(227-11) / 4+1=55$</span>

Output volume **[55x55x96]**
<span style="color:blue">Q: What is the total number of parameters in this layer?</span>

Parameters: $\left(11^{*} 11^{*} 3\right)^{*} 96=35 \mathrm{~K}$

After CONV1: 55x55x96
**Second layer** (POOL1): $3 \times 3$ filters applied at stride 2 

Output volume: 27x27x96

<span style="color:blue">Q: what is the number of parameters in this layer?</span>

Parameters: 0!



Full (simplified) AlexNet architecture: 

$[227\times227\times3]$ INPUT
$[55 \times 55 \times 96]$ <span style="color:red">CONV1</span>: $9611 \times 11$ filters at stride 4, pad 0 

$[27\times27\times96]$ <span style="color:blue">MAX POOL1</span>: $3 \times 3$ filters at stride 2 

$[27x27x96]$ :<span style="color:green">NORM1</span> Normalization layer 

$[27\times27\times256]$ <span style="color:red">CONV2:</span> $2565 \times 5$ filters at stride 1, pad 2 

$[13\times13\times256]$ <span style="color:blue">MAX POOL2</span>: $3 \times 3$ filters at stride 2 

$[13x13\times256]$ <span style="color:green">NORM2</span>: Normalization layer.
$[13x13\times384]$ <span style="color:red">CONV3</span>: $3843 \times 3$ filters at stride 1, pad 1 

$[13 \times 13 \times 384]$ <span style="color:red">CONV4</span>: $3843 \times 3$ filters at stride 1, pad 1 

$[13\times 13\times256]$ <span style="color:red">CONV5</span>: $2563 \times 3$ filters at stride 1, pad 1 

$[6 \times 6 \times 256]$ <span style="color:blue">MAX POOL3</span>: $3 \times 3$ filters at stride 2
[4096] <span style="color:orange">FC6</span>: 4096 neurons
[4096] <span style="color:orange">FC7</span>: 4096 neurons
[1000] <span style="color:orange">FC8</span>; 1000 neurons (class scores)

>Details/Retrospectives:
>- first use of $\operatorname{ReLU}$
>- used Norm layers (not common anymore)
>- heavy data augmentation
>- dropout $0.5$
>- batch size 128
>- SGD Momentum $0.9$
>- Learning rate $1 \mathrm{e}-2$, reduced by 10
>manually when val accuracy plateaus
>- L2 weight decay $5 \mathrm{e}-4$
>- 7 CNN ensemble: $18.2 \%=>15.4 \%$

<img src="images\image-20211018172655398.png" alt="image-20211018172655398"  />

#### Case Study: VGGNet 

[Simonyan and Zisserman, 2014]

<img src="images\image-20211018172711105.png" alt="image-20211018172711105"  />

INPUT: $[224\times224\times3]$  <span style="color:red">memory: $224^{*} 224^{\star} 3=150 \mathrm{~K}$</span> <span style="color:blue">params: 0 (not counting biases) </span>

CONV3-64: $[224 \times 224 \times 64]$ <span style="color:red">memory: $224^{*} 224^{*} 64=3.2 \mathrm{M}$</span> <span style="color:blue">params: $\left(3^{*} 3^{*} 3\right)^{*} 64=1,728$</span>
CONV3-64: $[224 \times 224 \times 64]$ <span style="color:red">memory: $224^{2} 224^{*} 64=3.2 \mathrm{M}$</span>  <span style="color:blue">params: $\left(3^{*} 3^{*} 64\right)^{*} 64=36,864$</span>

POOL2: $[112\times112\times64]$ <span style="color:red">memory: $112^{*} 112^{\circ} 64=800 \mathrm{~K}$</span> <span style="color:blue">params: 0</span>
CONV3-128: $\left[112 \times 112 \times 1281\right.$ <span style="color:red">memory: $112^{*} 112^{*} 128=1.6 \mathrm{M}$</span> <span style="color:blue">params: $\left(3^{*} 3^{*} 64\right)^{*} 128=73,728$</span>
CONV3-128: $[112\times112\times128]$ <span style="color:red">memory: $112^{*} 112^{*} 128=1.6 \mathrm{M}$ </span><span style="color:blue">params: $\left(3^{*} 3^{*} 128\right)^{*} 128=147,456$ </span>

POOL2: $[56\times56\times128]$ <span style="color:red">memory: $56^{*} 56^{*} 128=400 \mathrm{K}$ </span><span style="color:blue">params: 0</span>
CONV3-256: $[56\times56\times256]$ <span style="color:red">memory: $56^{*} 56^{*} 256=800 \mathrm{K}$</span> <span style="color:blue">params: $\left(3^{*} 3^{*} 128\right)^{*} 256=294,912$</span>
CONV3-256: $[56\times56\times256]$ <span style="color:red">memory: $56^{*} 56^{*} 256=800 \mathrm{K}$ </span><span style="color:blue">params: $\left(3^{*} 3^{*} 256\right)^{*} 256=589,824$</span>
CQNY3-256: $[56 \times 56 \times 256]$ <span style="color:red">memory: $56^{*} 56^{*} 256=800 \mathrm{K}$ </span><span style="color:blue">params: $\left(3^{*} 3^{*} 256\right)^{*} 256=589,824$</span>
POOL2: $[28\times28\times256]$ <span style="color:red">memory: $28^{*} 28^{*} 256=200 \mathrm{K}$ </span><span style="color:blue">params: 0</span>
CONV3-512: $[28 \times 28 \times 512]$ <span style="color:red">memory: $28^{*} 28^{*} 512=400 \mathrm{K}$ </span><span style="color:blue">params: $\left(3^{*} 3^{*} 256\right)^{*} 512=1,179,648$</span>
CONV3-512: $[28\times28\times512]$ <span style="color:red">memory: $28^{*} 28^{*} 512=400 \mathrm{K}$</span> <span style="color:blue">params: $\left(3^{*} 3^{*} 512\right)^{*} 512=2,359,296$</span>
CONV3-512: $[28\times28\times512]$ <span style="color:red">memory: $28^{*} 28^{*} 512=400 \mathrm{K}$ </span><span style="color:blue">params: $\left(3^{*} 3^{*} 512\right)^{*} 512=2,359,296$</span>
POOL2: $[14\times 14\times512]$ <span style="color:red">memory: $14^{*} 14^{*} 512=100 \mathrm{K}$ </span><span style="color:blue">params: 0</span>
CONV3-512: $[14 \times 14 \times 512]$ <span style="color:red">memory: $14^{*} 14^{*} 512=100 \mathrm{K}$</span> <span style="color:blue">params: $\left(3^{*} 3^{*} 512\right)^{*} 512=2,359,296$</span>
CONV3-512: $[14 \times 14 \times 512]$ <span style="color:red">memory: $14^{*} 14^{*} 512=100 \mathrm{K}$</span> <span style="color:blue">params: $\left(3^{*} 3^{*} 512\right)^{*} 512=2,359,296$</span>
CONV3-512: $[14\times14\times512]$ <span style="color:red">memory: $14^{*} 14^{*} 512=100 \mathrm{K}$</span> <span style="color:blue">params: $\left(3^{*} 3^{*} 512\right)^{*} 512=2,359,296$</span>
POOL2: $[7 \times 7 \times 512]$ <span style="color:red">memory: $7^{*} 7^{*} 512=25 \mathrm{K}$</span> <span style="color:blue">params: 0</span>
FC: $[1 \times 1 \times 4096]$ <span style="color:red">memory: 4096</span> <span style="color:blue">params: $7^{*} 7^{*} 512^{*} 4096=102,760,448$</span>
FC: $[1\times1\times4096]$ <span style="color:red">memory: 4096</span> <span style="color:blue">params: $4096^{*} 4096=16,777,216$</span>
FC: $[1\times1\times1000]$ <span style="color:red">memory; 1000</span> <span style="color:blue">params: $4096^{*} 1000=4,096,000$</span>
<span style="color:red">TOTAL memory: $24 \mathrm{M} * 4$ bytes $\sim=93 \mathrm{MB} /$ image</span> (only forward! $\sim^{\star} 2$ for bwd) 

<span style="color:blue">TOTAL params: $138 \mathrm{M}$ parameters</span>

#### Case Study: GoogLeNet

[Szegedy et al., 2014]

<img src="images\image-20211018174522833.png" alt="image-20211018174522833"  />

<img src="images\image-20211018174545009.png" alt="image-20211018174545009"  />

#### Case Study: ResNet

[He et al., 2015]
<span style="color:blue">ILSVRC 2015 winner $(3.6 \%$ top 5 error)</span>

<img src="images\image-20211018174648345.png" alt="image-20211018174648345"  />

<img src="C:\Users\ADMIN\AppData\Roaming\Typora\typora-user-images\image-20211018174717466.png" alt="image-20211018174717466"  />

##### CIFAR-10 experiments

<img src="images\image-20211018174753352.png" alt="image-20211018174753352"  />

<img src="images\image-20211018174813550.png" alt="image-20211018174813550"  />

<img src="images\image-20211018174831913.png" alt="image-20211018174831913"  />

<img src="images\image-20211018174847996.png" alt="image-20211018174847996"  />

### Summary

- ConvNets stack CONV,POOL,FC layers
- Trend towards smaller filters and deeper architectures
- Trend towards getting rid of POOL/FC layers (just CONV)
- Typical architectures look like
  **[(CONV-RELU) $^{*}$ N-POOL?] $^{*}$ M-(FC-RELU) ${ }^{\star}$ K,SOFTMAX** where $N$ is usually up to $\sim 5$, M is large, $0<=k<=2$.
  - but recent advances such as ResNet/GoogLeNet challenge this paradigm

## Lecture 8

### Content

>TBD

### Outline

- <span style="color:blue">Decision Trees</span>
  - imple but powerful learning algorithm
  - Used widely in Kaggle competitions
  - Lets us motivate concepts from information theory (entropy, mutual information, etc.)
- <span style="color:blue">Bias-variance decomposition</span>
  - Lets us motivate methods for combining different classifiers.

### Decision Trees

- Make predictions by splitting on features according to a tree structure.

  <img src="images\image-20211101163643574.png" alt="image-20211101163643574" style="zoom:80%;" />

- Split continuous features by checking whether that feature is greater than or less than some threshold.
- Decision boundary is made up of axis-aligned planes.

  <img src="images\image-20211101163717108.png" alt="image-20211101163717108"  />

- <span style="color:blue">Internal nodes</span> test a <span style="color:blue">feature</span>
- <span style="color:blue">Branching</span> is determined by the <span style="color:blue">feature value</span>
- <span style="color:blue">Leaf nodes</span> are <span style="color:blue">outputs</span> (predictions)

#### Classification and Regression

- Each path from root to a leaf defines a region $R_{m}$ of input space
- Let $\left\{\left(x^{\left(m_{1}\right)}, t^{\left(m_{1}\right)}\right), \ldots,\left(x^{\left(m_{k}\right)}, t^{\left(m_{k}\right)}\right)\right\}$ be the training examples that fall into $R_{m}$
- <span style="color:blue">Classification tree</span> (we will focus on this):
  - discrete output
  - leaf value $y^{m}$ typically set to the most common value in $\left\{t^{\left(m_{1}\right)}, \ldots, t^{\left(m_{k}\right)}\right\}$
- <span style="color:blue">Regression tree:</span>
  - continuous output
  - leaf value $y^{m}$ typically set to the mean value in $\left\{t^{\left(m_{1}\right)}, \ldots, t^{\left(m_{k}\right)}\right\}$

#### Discrete Features

- Will I eat at this restaurant?

  <img src="images\image-20211101164035341.png" alt="image-20211101164035341"  />

- Split discrete features into a partition of possible values.

  <img src="images\image-20211101164103342.png" alt="image-20211101164103342"  />

#### Attempts

The drawing below shows a dataset. Each example in the dataset has two inputs features 𝑥 and 𝑦, and maybe classified as a positive example (labelled +) or a negative example (labelled −). Draw a decision tree which correctly classifies each example in the dataset.

<img src="images\image-20211101164238207.png" alt="image-20211101164238207"  />

<img src="images\image-20211101164251827.png" alt="image-20211101164251827"  />

#### Learning Decision Trees

- For any training set we can construct a decision tree that has exactly the one leaf for every training point, but it probably won't generalize.
  - Decision trees are universal function approximators.
- But, finding the smallest decision tree that correctly classifies a training set is NP complete.
  - If you are interested, check: Hyafil \& Rivest'76.
- So, how do we construct a useful decision tree?

- Resort to a greedy heuristic:
  - Start with the whole training set and an empty decision tree.
  - Pick a feature and candidate split that would most reduce the loss.
  - Split on that feature and recurse on subpartitions.
- Which loss should we use?
  - Let's see if misclassification rate is a good loss.

#### Choosing a Good Split

- Consider the following data. Let's split on width.

  <img src="images\image-20211101164504718.png" alt="image-20211101164504718"  />

- Recall: classify by majority.

  <img src="images\image-20211101164530831.png" alt="image-20211101164530831"  />

- A and B have the same misclassification rate, so which is the best split? 

  **Left. Because the left one has one part with certainty.**

- A feels like a better split, because the left-hand region is very certain about whether the fruit is an orange.

  <img src="images\image-20211101164631733.png" alt="image-20211101164631733"  />

- Can we quantify this?

- How can we quantify uncertainty in prediction for a given leaf node?
  - If all examples in leaf have same class: good, low uncertainty
  - If each class has same amount of examples in leaf: bad, high uncertainty
- **Idea**: Use counts at leaves to define probability distributions; use a probabilistic notion of uncertainty to decide splits.
- A brief detour through information theory...

#### Quantifying Uncertainty

- The <span style="color:blue">entropy</span> of a discrete random variable is a number that quantifies the <span style="color:blue">uncertainty</span> inherent in its possible outcomes.
- The mathematical definition of entropy that we give in a few slides may seem arbitrary, but it can be motivated axiomatically.
  - If you're interested, check: Information Theory by Robert Ash.
- To explain entropy, consider flipping two different coins...

#### We Flip Two Different Coins

<img src="images\image-20211101164807010.png" alt="image-20211101164807010" style="zoom:80%;" />

- The entropy of a loaded coin with probability $p$ of heads is given by
  $$
  \begin{align*}
  -p \log _{2}(p)-(1-p) \log _{2}(1-p)
  \end{align*}
  $$
  <img src="images\image-20211101164911108.png" alt="image-20211101164911108"  />

- Notice: the coin whose outcomes are more certain has a lower entropy.
- In the extreme case $p=0$ or $p=1$, we were certain of the outcome before observing. So, we gained no certainty by observing it, i.e., entropy is 0 .

#### Quantifying Uncertainty

- Can also think of <span style="color:blue">entropy</span> as the expected information content of a random draw from a probability distribution.

  <img src="images\image-20211101165039202.png" alt="image-20211101165039202"  />

- Claude Shannon showed: you cannot store the outcome of a random draw using fewer expected bits than the entropy without losing information,
- So units of entropy are <span style="color:blue">bits</span>; a fair coin flip has 1 bit of entropy.

### Entropy

- More generally, the <span style="color:blue">entropy</span> of a discrete random variable $Y$ is given by
  $$
  \begin{align*}
  H(Y)=-\sum_{y \in Y} p(y) \log _{2} p(y)
  \end{align*}
  $$

- **"High Entropy":**

  - Variable has a uniform like distribution over many outcomes
  - Flat histogram
  - Values sampled from it are less predictable

- **"Low Entropy"**

  - Distribution is concentrated on only a few outcomes
  - Histogram is concentrated in a few areas
  - Values sampled from it are more predictable

- Suppose we observe partial information $X$ about a random variable $Y$
  - For example, $X=\operatorname{sign}(Y)$.
- We want to work towards a definition of the expected amount of information that will be conveyed about $Y$ by observing $X$.
  - Or equivalently, the expected reduction in our uncertainty about $Y$ after observing $X$.

##### Entropy of a Joint Distribution

- Example: $X=\{$ Raining, Not raining $\}, Y=\{$ Cloudy, Not cloudy $\}$

  |             | Cloudy | Not Cloudy |
  | :---------: | :----: | :--------: |
  |   Raining   | 24/100 |   1/100    |
  | Not Raining | 25/100 |   50/100   |

  $$
  \begin{align*}
  H(X, Y) &=-\sum_{x \in X} \sum_{y \in Y} p(x, y) \log _{2} p(x, y) \\
  &=-\frac{24}{100} \log _{2} \frac{24}{100}-\frac{1}{100} \log _{2} \frac{1}{100}-\frac{25}{100} \log _{2} \frac{25}{100}-\frac{50}{100} \log _{2} \frac{50}{100} \\
  & \approx 1.56 \mathrm{bits}
  \end{align*}
  $$

#### Specific Conditional Entropy

- What is the entropy of cloudiness $Y$, **given that it is raining**?
  $$
  \begin{align*}
  H(Y \mid X=x) &=-\sum_{y \in Y} p(y \mid x) \log _{2} p(y \mid x) \\
  &=-\frac{24}{25} \log _{2} \frac{24}{25}-\frac{1}{25} \log _{2} \frac{1}{25} \\
  & \approx 0.24 \mathrm{bits}
  \end{align*}
  $$

- We used: $p(y \mid x)=\frac{p(x, y)}{p(x)}$, and $p(x)=\sum_{y} p(x, y) \quad$ (sum in a row)

#### Conditional Entropy

- The expected conditional entropy:
  $$
  \begin{align*}
  H(Y \mid X) &=\sum_{x \in X} p(x) H(Y \mid X=x) \\
  &=-\sum_{x \in X} \sum_{y \in Y} p(x, y) \log _{2} p(y \mid x)
  \end{align*}
  $$

- What is the entropy of cloudiness, given the knowledge of whether or not it is raining?
  $$
  \begin{align*}
  H(Y \mid X) &=\sum_{x \in X} p(x) H(Y \mid X=x) \\
  &=\frac{1}{4} H(\text { cloudy } \mid \text { is raining })+\frac{3}{4} H(\text { cloudy } \mid \text { not raining }) \\
  & \approx 0.75 \text { bits }
  \end{align*}
  $$
  

- Some useful properties:
  - $H$ is always non-negative
  - Chain rule: $H(X, Y)=H(X \mid Y)+H(Y)=H(Y \mid X)+H(X)$
  - If $X$ and $Y$ independent, then $X$ does not affect our uncertainty about $Y: H(Y \mid X)=H(Y)$
  - But knowing $Y$ makes our knowledge of $Y$ certain: $H(Y \mid Y)=0$
  - By knowing $X$, we can only decrease uncertainty about $Y$ : $H(Y \mid X) \leq H(Y)$

#### Information Gain

|             | Cloudy | Not Cloudy |
| :---------: | :----: | :--------: |
|   Raining   | 24/100 |   1/100    |
| Not Raining | 25/100 |   50/100   |

- How much more certain am I about whether it's cloudy if I'm told whether it is raining? My uncertainty in $Y$ minus my expected uncertainty that would remain in $Y$ after seeing $X$.

- This is called the <span style="color:blue">information gain</span> $I G(Y \mid X)$ in $Y$ due to $X$, or the <span style="color:blue">mutual information</span> of $Y$ and $X$
  $$
  \begin{align*}
  I G(Y \mid X)=H(Y)-H(Y \mid X)
  \end{align*}
  $$

- If $X$ is completely uninformative about $Y: I G(Y \mid X)=0$

- If $X$ is completely informative about $Y: I G(Y \mid X)=H(Y)$

#### Revisiting Our Original Example

- Information gain measures the informativeness of a variable, which is exactly what we desire in a decision tree split!

- The information gain of a split: how much information (over the training set) about the class label $Y$ is gained by knowing which side of a split you're on.

  

- What is the information gain of split B? Not terribly informative...

  <img src="images\image-20211101171458654.png" alt="image-20211101171458654"  />

- Root entropy of class outcome: $H(Y)=-\frac{2}{7} \log _{2}\left(\frac{2}{7}\right)-\frac{5}{7} \log _{2}\left(\frac{5}{7}\right) \approx 0.86$
- Leaf conditional entropy of class outcome: $H(Y \mid l e f t) \approx 0.81$, $H(Y \mid$ right $) \approx 0.92$
- $I G($ split $) \approx 0.86-\left(\frac{4}{7} \cdot 0.81+\frac{3}{7} \cdot 0.92\right) \approx 0.006$



- What is the information gain of split A? Very informative!

  <img src="images\image-20211101171803281.png" alt="image-20211101171803281"  />

  - Root entropy of class outcome: $H(Y)=-\frac{2}{7} \log _{2}\left(\frac{2}{7}\right)-\frac{5}{7} \log _{2}\left(\frac{5}{7}\right) \approx 0.86$
  - Leaf conditional entropy of class outcome: $H(Y \mid$ left $)=0$, $H(Y \mid$ right $) \approx 0.97$
  - $I G($ split $) \approx 0.86-\left(\frac{2}{7} \cdot 0+\frac{5}{7} \cdot 0.97\right) \approx 0.17 ! !$

### Constructing Decision Trees

<img src="images\image-20211101171855395.png" alt="image-20211101171855395"  />

- At each level, one must choose:
  1. Which feature to split.
  2. Possibly where to split it.

- Choose them based on how much information we would gain from the decision! (choose feature that gives the highest gain)

#### Decision Tree Construction Algorithm

- Simple, greedy, recursive approach, builds up tree node-by-node
  1. pick a feature to split at a non-terminal node
  2. split examples into groups based on feature value
  3. for each group:
     + if no examples - return majority from parent
     + else if all examples in same class - return class
     + else loop to step 1
- Terminates when all leaves contain only examples in the same class or are empty.

#### Back to Our Example

<img src="images\image-20211101172028355.png" alt="image-20211101172028355"  />

<img src="images\image-20211101172043348.png" alt="image-20211101172043348"  />
$$
I G(Y)=H(Y)-H(Y \mid X) \\
I G(\text {type})=1-\left[\frac{2}{12} H(Y \mid \mathrm{Fr}.)+\frac{2}{12} H(Y \mid \mathrm{It} .)+\frac{4}{12} H(Y \mid \text{Thai})+\frac{4}{12} H(Y \mid \text {Bur.})\right]=0 \\
I G(\text {Patrons})=1-\left[\frac{2}{12} H(0,1)+\frac{4}{12} H(1,0)+\frac{6}{12} H\left(\frac{2}{6}, \frac{4}{6}\right)\right] \approx 0.541
$$

### What Makes a Good Tree?

- Not too small: need to handle important but possibly subtle distinctions in data
- Not too big:
  - Computational efficiency (avoid redundant, spurious attributes)
  - Avoid over-fitting training examples
  - Human interpretability
- <span style="color:blue">"Occam's Razor"</span>: find the simplest hypothesis that fits the observations
  - Useful principle, but hard to formalize (how to define simplicity?)
  - See Domingos, 1999, "The role of Occam's razor in knowledge discovery"
- We desire small trees with informative nodes near the root

### Decision Tree Miscellany

- Problems:
  - You have exponentially less data at lower levels
  - Too big of a tree can overfit the data
  - Greedy algorithms don't necessarily yield the global optimum
- Handling continuous attributes
  - Split based on a threshold, chosen to maximize information gain

#### Comparison to some other classifiers

Advantages of decision trees over KNNs and neural nets
- Simple to deal with discrete features, missing values, and poorly scaled data
- Fast at test time
- More interpretable

Advantages of KNNs over decision trees

- Few hyperparameters
- Can incorporate interesting distance measures (e.g. shape contexts)

Advantages of neural nets over decision trees

- Able to handle attributes/features that interact in very complex ways (e.g. pixels)



- We've seen many classification algorithms.
- We can combine multiple classifiers into an ensemble, which is a set of predictors whose individual decisions are combined in some way to classify new examples
  - E.g., (possibly weighted) majority vote
- For this to be nontrivial, the classifiers must differ somehow, e.g.
  - Different algorithm
  - Different choice of hyperparameters
  - Trained on different data
  - Trained with different weighting of the training examples
- Next lecture, we will study some specific ensembling techniques.



- Today, we deepen our understanding of generalization through a bias-variance decomposition.
  - This will help us understand ensembling methods.

### Bias-Variance Decomposition

- Recall that overly simple models underfit the data, and overly complex models overfit

  <img src="images\image-20211101172926546.png" alt="image-20211101172926546"  />

- We can quantify this effect in terms of the <span style="color:blue">bias/variance decomposition</span>.
  - Bias and variance of what?

####  Basic Setup

- Suppose the training set $\mathcal{D}$ consists of pairs $\left(\mathbf{x}_{i}, t_{i}\right)$ sampled <span style="color:blue">independent and identically distributed (i.i.d.)</span> from a single <span style="color:blue">data generating distribution</span> $p_{\text {sample }}$.
- Pick a fixed query point $\mathrm{x}$ (denoted with a green $\mathrm{x}$ ).
- Consider an experiment where we sample lots of training sets independently from $p_{\text {sample. }}$

  <img src="images\image-20211101173057668.png" alt="image-20211101173057668"  />

- Let's run our learning algorithm on each training set, and compute its prediction $y$ at the query point $\mathbf{x}$.
- We can view $y$ as a random variable, where the randomness comes from the choice of training set.
- The classification accuracy is determined by the distribution of $y$.

  <img src="images\image-20211101173121165.png" alt="image-20211101173121165"  />

Here is the analogous setup for regression:

<img src="images\image-20211101173142530.png" alt="image-20211101173142530"  />

Since $y$ is a random variable, we can talk about its expectation, variance, etc.

- Recap of basic setup:
  - Fix a query point $\mathrm{x}$.
  - Repeat:
    - Sample a random training dataset $\mathcal{D}$ i.i.d. from the data generating distribution $p_{\text {sample. }}$
    - Run the learning algorithm on $\mathcal{D}$ to get a prediction $y$ at $\mathbf{x}$.
    - Sample the (true) target from the conditional distribution $p(t \mid \mathbf{x})$.
    - Compute the loss $L(y, t)$
- Notice: $y$ is independent of $t$
- This gives a distribution over the loss at $\mathbf{x}$, with expectation $\mathbb{E}[L(y, t) \mid \mathbf{x}]$.
- For each query point $\mathbf{x}$, the expected loss is different. We are interested in minimizing the expectation of this with respect to $\mathbf{x} \sim p_{\text {sample }}$.

#### Bayes Optimality

- For now, focus on squared error loss, $L(y, t)=\frac{1}{2}(y-t)^{2}$.

- A first step: suppose we knew the conditional distribution $p(t \mid \mathbf{x}) .$ What value $y$ should we predict?

- Here, we are treating $t$ as a random variable and choosing $y$.

- **Claim**: $y_{*}=\mathbb{E}[t \mid \mathbf{x}]$ is the best possible prediction.

- **Proof**:
  $$
  \begin{align*}
  \mathbb{E}\left[(y-t)^{2} \mid \mathbf{x}\right] &=\mathbb{E}\left[y^{2}-2 y t+t^{2} \mid \mathbf{x}\right] \\
  &=y^{2}-2 y \mathbb{E}[t \mid \mathbf{x}]+\mathbb{E}\left[t^{2} \mid \mathbf{x}\right] \\
  &=y^{2}-2 y \mathbb{E}[t \mid \mathbf{x}]+\mathbb{E}[t \mid \mathbf{x}]^{2}+\operatorname{Var}[t \mid \mathbf{x}] \\
  &=y^{2}-2 y y_{*}+y_{*}^{2}+\operatorname{Var}[t \mid \mathbf{x}] \\
  &=\left(y-y_{*}\right)^{2}+\operatorname{Var}[t \mid \mathbf{x}]
  \end{align*}
  $$

$$
\mathbb{E}\left[(y-t)^{2} \mid \mathbf{x}\right]=\left(y-y_{*}\right)^{2}+\operatorname{Var}[t \mid \mathbf{x}]
$$
- The first term is nonnegative, and can be made 0 by setting $y=y_{*}$.

- The second term corresponds to the inherent unpredictability, or <span style="color:blue">noise</span>, of the targets, and is called the <span style="color:blue">Bayes error</span>.
  - This is the best we can ever hope to do with any learning algorithm. An algorithm that achieves it is <span style="color:blue">Bayes optimal</span>.
  - Notice that this term doesn't depend on $y$.

- This process of choosing a single value $y_{*}$ based on $p(t \mid \mathbf{x})$ is an example of <span style="color:blue">decision theory</span>.

  

- Now return to treating $y$ as a random variable (where the randomness comes from the choice of dataset).
- We can decompose out the expected loss (suppressing the conditioning on $\mathrm{x}$ for clarity):

$$
\begin{aligned}
\mathbb{E}\left[(y-t)^{2}\right] &=\mathbb{E}\left[\left(y-y_{\star}\right)^{2}\right]+\operatorname{Var}(t) \\
&=\mathbb{E}\left[y_{\star}^{2}-2 y_{\star} y+y^{2}\right]+\operatorname{Var}(t) \\
&=y_{\star}^{2}-2 y_{\star} \mathbb{E}[y]+\mathbb{E}\left[y^{2}\right]+\operatorname{Var}(t) \\
&=y_{\star}^{2}-2 y_{\star} \mathbb{E}[y]+\mathbb{E}[y]^{2}+\operatorname{Var}(y)+\operatorname{Var}(t) \\
&=\underbrace{\left(y_{\star}-\mathbb{E}[y]\right)^{2}}_{\text {bias }}+\underbrace{\operatorname{Var}(y)}_{\text {variance }}+\underbrace{\operatorname{Var}(t)}_{\text {Bayes error }}
\end{aligned}
$$

$$
\mathbb{E}\left[(y-t)^{2}\right]=\underbrace{\left(y_{\star}-\mathbb{E}[y]\right)^{2}}_{\text {bias }}+\underbrace{\operatorname{Var}(y)}_{\text {variance }}+\underbrace{\operatorname{Var}(t)}_{\text {Bayes error }}
$$
- We just split the expected loss into three terms:
  - <span style="color:blue">bias</span>: how wrong the expected prediction is (corresponds to underfitting)
  - <span style="color:blue">variance</span>: the amount of variability in the predictions (corresponds to overfitting)
  - Bayes error: the inherent unpredictability of the targets
- Even though this analysis only applies to squared error, we often loosely use "bias" and "variance" as synonyms for "underfitting" and "overfitting".

#### Bias and Variance

- Throwing darts $=$ predictions for each draw of a dataset

  <img src="images\image-20211101173700110.png" alt="image-20211101173700110"  />

- Be careful, what doesn't this capture?
  - We average over points $\mathbf{x}$ from the data distribution.

