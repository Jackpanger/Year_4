# Bio-Computation 5-8

## Lecture 5

### Content

>TBD

### Multilayer Perceptrons

+ The <span style="color:blue">**multilayer perceptron**</span> (<span style="color:red">**MLP**</span>) is a hierarchical structure of several perceptrons, which overcomes the shortcomings of the single-layer networks. 

+ The MLP neural network is able to learn nonlinear function mappings.
  + learning a rich variety of nonlinear decision surfaces.

+ Nonlinear functions can be represented by MLPs with units that use nonlinear activation functions.
  + <span style="color:green">Multiple layers of cascaded linear units still produce only linear mappings!</span>

#### Differentiable Activation Functions

- Training algorithms for MLP require differentiable, continuous nonlinear activation functions.

- Such a function is the sigmoid function:
  $$
  \begin{align*}
  o=\sigma(s)=\frac{1}{1+e^{-s}}
  \end{align*}
  $$
  <img src="images\image-20211012165110187.png" alt="image-20211012165110187" style="zoom:80%;" />

  where $s$ is the sum:
  $$
  \begin{align*}
  s=\sum_{i=0}^{d} w_{i} x_{i}
  \end{align*}
  $$
  which is the products from the weights $w_{i}$ and the inputs $x_{i}$

  >```matlab
  >You can use fplot('1/(1+exp(-x))',[-5 5]) to plot sigmoid
  >```

- Another nonlinear function often used in practice is the hyperbolic tangent.
  $$
  o=\tanh (s)=\frac{e^{s}-e^{-s}}{e^{s}+e^{-s}}
  $$
  <img src="images\image-20211012165307909.png" alt="image-20211012165307909" style="zoom:80%;" />

- The mean of tanh is 0

### Multilayer Network Structure

<img src="images\image-20211012165329552.png" alt="image-20211012165329552" style="zoom:80%;" />

- A two-layer neural network implements the function:
  $$
  \begin{align*}
  f(x)=\sigma\left(\sum_{j=1}^{J} w_{j k} \sigma\left(\sum_{i=1}^{I} w_{i j} x_{i}+w_{o j}\right)+w_{o k}\right)
  \end{align*}
  $$
  where: $\mathbf{x}$ is the input vector,
  Surpat trom hidden layer
  $w_{o j}$ and $w_{o k}$ are the bias terms,
  $w_{i j}$ are the weights connecting the input with the hidden nodes $w_{j k}$ are the weights connecting the hidden with output nodes $\sigma$ is the sigmoid activation function.

- The hidden units enable the multilayer network to learn complex tasks by extracting <span style="color:green">**progressively more meaningful information**</span> from the input examples.
- The MLP has a highly connected topology since every input is connected to all nodes in the first hidden layer, every unit in the hidden layers is connected to all nodes in the next layer, and so on.
- The input signals, initially these are the input examples, propagate through the neural network in a forward direction on a layer-by-layer basis, that is why they are often called **<span style="color:red">feedforward multilayer networks</span>.**

### Representation Power of MLP

- Properties concerning the representational power of MLP:
  - **<span style="color:blue">learning arbitrary functions:</span>** any function can be learned with an arbitrary accuracy by a two-layer network;
  - **<span style="color:blue">learning continuous functions:</span>** every bounded continuous function can be learned with a small error by a two-layer network (the number of hidden units depends on the function to be approximated);
  - **<span style="color:blue">learning Boolean functions:</span>** every Boolean function can be learned exactly by a two-layer network although the number of hidden units grows exponentially with the input dimension.

### Backpropagation Learning Algorithm

- MLP became applicable on practical tasks after the discovery of a supervised training algorithm, the error backpropagation Iearning algorithm.
- The error backpropagation algorithm includes two passes through the network:
  - <span style="color:blue">**forward pass**</span>, and
  - <span style="color:blue">**backward pass**</span> .

- During the backward pass the weights are adjusted in accordance with the <span style="color:blue">**error correction rule**</span>. The actual network output is subtracted from the given output in the example and the weights are adjusted so as to make the network output close to the desired one.
- The backpropagation algorithm does gradient descent as it moves in direction opposite to the gradient of the error, that is in direction of the steepest decrease of the error.
- This is the direction of most rapid error decrease by varying all the weights simultaneously:

$$
\nabla \mathrm{E}(\mathrm{w})=\left[\frac{\partial \mathrm{E}}{\partial \mathrm{w}_{0}}, \frac{\partial \mathrm{E}}{\partial \mathrm{w}_{1}}, \ldots, \frac{\partial \mathrm{E}}{\partial \mathrm{w}_{\mathrm{d}}}\right]
$$

- By gradient descent search, backpropagation training algorithm minimizes a cost function $E$ (the mean square difference between the desired and actual net outputs).
- The network is trained initially selecting small random weights and then presenting all training data incrementally.
- Weights are adjusted after every trial using side information specifying the correct class until weights converge and the cost function is reduced to an acceptable value.

- <span style="color:blue">**Initialization**</span>: Examples $\left\{\left(x_{e}, y_{e}\right)\right\}_{e=1}^{N}$, initial weights $w_{i}$ set to small random values, learning rate $\eta$
- <span style="color:blue">**Repeat**</span>

  - For each training example $(x, y)$

  Forward

  - calculate the outputs using the sigmoid function:
    $o_{j}=\sigma\left(s_{j}\right)=\frac{1}{1+e^{-s_{j}}}, s_{j}=\sum_{i=0}^{d} w_{i j} o_{i} \quad \text { at the hidden units j} \\$
    where $o_{i}=x_{i}$
    $o_{k}=\sigma\left(s_{k}\right)=\frac{1}{1+e^{-s_{k}}}, s_{k}=\sum_{i=0}^{d} w_{j k} o_{j} \quad \text { at the output units k} \\$

  Backward

  - compute the <span style="color:blue">benefit $\beta_{k}$</span> at the node $k$ in the output layer:
    $$
    \beta_{k}=o_{k}\left(1-o_{k}\right)\left[y_{k}-o_{k}\right] \quad\text{effects from the output nodes}
    $$

  + compute <span style="color:blue">the changes for weights $j \rightarrow k$</span> on connections to nodes in the output layer:
    $$
    \begin{align*}
    &\Delta w_{j k}=\eta \beta_{k} o_{j} \quad\text{effects from the output of the neuron}\\
    &\Delta w_{0 k}=\eta \beta_{k}
    \end{align*}
    $$

  + compute the <span style="color:blue">benefit $\beta_{\mathrm{j}}$</span> for the hidden node $j$ with the formula:
    $$
    \begin{align*}
    \beta_{j}=o_{j}\left(1-o_{j}\right)\left[\sum_{k} \beta_{k} w_{j k}\right] \quad\text{effects from nultiple nodes in the next layer}\\
    \end{align*}
    $$

  + compute the changes for the weights $i \rightarrow j$ on connections to nodes in the hidden layer:
    $$
    \begin{align*}
    &\Delta w_{i j}=\eta \beta_{j} o_{i} \\
    &\Delta w_{0 j}=\eta \beta_{j}
    \end{align*}
    $$

  + update the weights by the computed changes:
    $$
    \begin{align*}
    w=w+\Delta w
    \end{align*}
    $$

  <span style="color:blue">**until** </span> termination condition is satisfied.

### On-line Training

Revision by example is called <span style="color:blue">**on-line (incremental) learning**</span>.
- **Initialization**: Examples $\left\{\left(x_{e}, y_{e}\right)\right\}_{e=1}^{N}$, initial weights $w_{i}$ set to small random values, learning rate $\eta$
- **Repeat**
  pick a training example $(x, y)$

  - forward propagate the example and calculate the outputs using the sigmoid function

  - backward propagate the error to calculate the benefits

  - update the weights by the computed changes:
    $$
    \begin{align*}
    \mathrm{w}=\mathrm{w}+\Delta \mathrm{w}
    \end{align*}
    $$

+ **until** termination condition is satisfied. 

### Derivation of Backpropagation Algorithm

- The BP training algorithm for MLP is a generalized gradient descent rule, according to which with each training example every weight is updated as:
  $$
  \quad w=w+\Delta w
  $$
  where: 
  $$
  \begin{align*}
  \quad \Delta \mathrm{w}=-\eta \frac{\partial \mathrm{E}_{\mathrm{e}}}{\partial \mathrm{w}}, \quad \mathrm{E}_{\mathrm{e}}=\frac{1}{2} \sum_{k}\left(y_{k}-o_{k}\right)^{2}
  \end{align*}
  $$

- The implementation of the generalized gradient descent rule requires to derive an expression for the computation of the derivatives $\partial \mathrm{E}_{\mathrm{e}} / \partial \mathrm{w}$
  $$
  \begin{align*}
  \frac{\partial E_{e}}{\partial w}=\frac{\partial E_{e}}{\partial s} \cdot \frac{\partial s}{\partial w}
  \end{align*}
  $$

- The first part $\partial \mathrm{E}_{\mathrm{e}} /$ os reflects the change of the error as a function of the change in the network weighted input to the unit.

- The second part $\partial s / \partial w$ reflects the change in the network weighted input as a function of the change of particular weight $w$ to that node.

- Since: 
  $$
  \begin{align*}
  \quad \frac{\partial s}{\partial w}=\frac{\partial\left(\sum_{l} w_{l} o_{l}\right)}{\partial w}=o
  \end{align*}
  $$

- The expression is reduced as follows:
  $$
  \begin{align*}
  \frac{\partial E_{e}}{\partial w}=\frac{\partial E_{e}}{\partial s} \cdot o
  \end{align*}
  $$

- For weights $j \rightarrow k$ on connections to nodes in the output layer:
  $$
  \begin{align*}
  &\frac{\partial E_{e}}{\partial w_{j k}}=\frac{\partial E_{e}}{\partial s_{k}} \cdot o_{j} \\
  &\frac{\partial E_{e}}{\partial s_{k}}=\frac{\partial E_{e}}{\partial o_{k}} \cdot \frac{\partial o_{k}}{\partial s_{k}} \\
  &\frac{\partial E_{e}}{\partial o_{k}}=\frac{\partial\left(\frac{1}{2} \sum_{k}\left(y_{l}-o_{l}\right)^{2}\right)}{\partial o_{k}}=\frac{\partial\left(\frac{1}{2}\left(y_{k}-o_{k}\right)^{2}\right)}{\partial o_{k}} \\
  &\quad=\frac{1}{2} \cdot 2 \cdot\left(y_{k}-o_{k}\right) \frac{\partial\left(y_{k}-o_{k}\right)}{\partial o_{k}} \\
  &\quad=-\left(y_{k}-o_{k}\right) \\
  &\frac{\partial o_{k}}{\partial s_{k}}=\frac{\partial \sigma\left(s_{k}\right)}{\partial s_{k}}=o_{k}\left(1-o_{k}\right)
  \end{align*}
  $$

- Therefore:
  $$
  \begin{align*}
  \frac{\partial E_{e}}{\partial s_{k}}=-\left(y_{k}-o_{k}\right) o_{k}\left(1-o_{k}\right) \quad \frac{\partial E_{e}}{\partial w_{j k}}=\frac{\partial E_{e}}{\partial s_{k}} \cdot o_{j}
  \end{align*}
  $$

- Then we substitute:
  $$
  \begin{align*}
  \Delta w_{j k}=-\frac{\partial E_{e}}{\partial w_{j k}}=\eta \beta_{k} o_{j} \quad \beta_{k}=\left(y_{k}-o_{k}\right) o_{k}\left(1-o_{k}\right)
  \end{align*}
  $$

+ The gradient descent rule in previous lecture:
  $$
  \begin{align*}
  \Delta w_{i}=\Delta w_{i}+\eta\left(y_{e}-o_{e}\right) \sigma(s)(1-\sigma(s)) x_{i e}
  \end{align*}
  $$

- For weights $i \rightarrow j$ on connections to nodes in the hidden layer
$$
\frac{\partial E_{e}}{\partial w_{i j}}=\frac{\partial E_{e}}{\partial s_{j}} \cdot o_{i}
$$
+ In this case the error depends on the errors committed by all output units:
  $$
  \begin{align*}
  &\frac{\partial E_{e}}{\partial s_{j}}=\sum_{k}\frac{\partial E_{e}}{\partial s_{k}} \cdot \frac{\partial s_{k}}{\partial s_{j}}=\sum_{k}-\beta_{k} \cdot \frac{\partial s_{k}}{\partial s_{j}} &\frac{\partial E_{e}}{\partial s_{k}}=-\left(y_{k}-o_{k}\right) o_{k}\left(1-o_{k}\right)\\
  &=\sum_{k}-\beta_{k} \cdot \frac{\partial s_{k}}{\partial o_{j}} \cdot \frac{\partial o_{j}}{\partial s_{j}} &o_{k}=\sigma\left(s_{k}\right)=\frac{1}{1+e^{-s_{k}}}, s_{k}=\sum_{i=0}^{d} w_{j k} o_{j}\\
  &=\sum_{k}\left(-\beta_{k}\right) \cdot w_{j k} \cdot \frac{\partial o_{j}}{\partial s_{j}}=\sum_{k}\left(-\beta_{k}\right) \cdot w_{j k} \cdot o_{j}\left(1-o_{j}\right) &o_{j}=\sigma\left(s_{j}\right)=\frac{1}{1+e^{-s_{j}}}, s_{j}=\sum_{i=0}^{d} w_{i j} o_{i}
  \end{align*}
  $$

- For the hidden units:
  $$
  \begin{align*}
  \begin{aligned}
  &\Delta w_{i j}=\eta \beta_{j} o_{i} \\
  &\Delta w_{0 j}=\eta \beta_{j}
  \end{aligned} \quad \beta_{j}=-\frac{\partial E_{e}}{\partial s_{j}}=o_{j}\left(1-o_{j}\right)\left[\sum_{k} \beta_{k} w_{j k}\right]
  \end{align*}
  $$
  Note: This analysis was made for a single training pattern, but it can be generalized so that:
  $$
  \begin{align*}
  \frac{\partial E_{\text {total }}}{\partial w_{i j}}=\sum_{e} \frac{\partial E_{e}}{\partial w_{i j}}
  \end{align*}
  $$
  Thus, we just need to sum out weight changes over the examples.

### Batch Backpropagation Algorithms

<span style="color:blue">**Revision by Epoch**</span>

- From mathematical point of view the error derivatives should be computed after each epoch, i.e., after all examples in the training set have been processed.
  - This means that the error derivative is taken to be the sum of the error derivatives for all examples.
  - While this revision by epoch may have stronger theoretical motivation, revision by a particular example may yield better results and is more commonly used.
- Revision by epoch is called <span style="color:blue">**Batch Learning**</span>.

#### Batch version of the backpropagation algorithm

- <span style="color:blue">**Initialization**</span>:
Examples $\left\{\left(\mathrm{x}_{e}, y_{e}\right)\right\}_{e=1}^{N}$, initial weights $w_{i}$ set to small random values, learning rate $\eta$
- <span style="color:blue">**Repeat**</span>

  + for each training example $(x, y)$

    - forward propagate the example and calculate the outputs using the sigmoid function
    - backward propagate the error to calculate the benefits

  + after processing all examples update the weights by the computed changes:
    $$
    \begin{align*}
    w=w+\Delta w
    \end{align*}
    $$

+ <span style="color:blue">**until** </span> termination condition is satisfied.

