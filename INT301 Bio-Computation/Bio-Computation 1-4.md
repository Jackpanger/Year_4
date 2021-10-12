# Bio-Computation 1-4

>Assessment 1 
>
>+ week 8
>+ open book 
>+ lab
>
>Assessment 2
>
>+ week 14
>+ open book
>+ lab
>
>Final exam
>
>+ 2 hour
>+ One-page open book (probably)

## Lecture 0

### Content

>[Biological Neural Networks Overview](#Biological Neural Networks Overview): [Abstract neuron](#Abstract neuron)

#### Bio-computation

+ **Bio-computation**: a field devoted to tackling complex problems using computational methods modeled after principles encountered in Nature. 

+ **Goal**: to produce informatics tools with enhanced robustness, scalability, flexibility and reliability
+ The main content is **Artificial Neural Networks**.

#### Artificial intelligence (AI), deep learning, and neural networks

<img src="images\image-20210907100728112.png" alt="image-20210907100728112" style="zoom: 67%;" />

**AI (Artificial Intelligence)** - any technique which enables computer to mimic human behavior

**ML (Machine Learning)** - subset of AI techniques which use statistical methods to enable machines to improve with experience

**Neural network ** -- also known as "artificial" neural network -- is one type of machine learning that's loosely based on how neurons work in the brain

**DL (Deep Learning) **- subset of ML which makes the computation of multi-layer neural network feasible

#### ANN (Artificial neural networks) : a brief history [ignored]

#### Biological Neural Network Approach

+ Human brain is an intelligent system. By studying how the brain works we can learn what intelligence is and what properties of the brain are essential for any intelligent system. 

+ Other essential attributes include that <span style="color:red">**memory**</span> is primarily a sequences of patterns, that behavior is an essential part of all learning, and that learning must be continuous. 

+ In addition, biological neurons are far more sophisticated than the simple neurons used in the simple neural network approach. 

### Biological Neural Networks Overview

+ The inner-workings of the human brain are often modeled around the concept of neurons and the networks of <span style="color:red">**neurons**</span> known as <span style="color:red">**biological neural networks**</span>
  + It’s estimated that the human brain contains roughly 100 billion neurons, which are connected along pathways throughout these networks.

<img src="images\image-20210907101849632.png" alt="image-20210907101849632" style="zoom:67%;" />

+ At a very high level, neurons communicate with one another through an interface consisting of <span style="color:red">**axon terminals**</span> that are connected to <span style="color:red">**dendrites**</span> across a gap (<span style="color:red">**synapse**</span>)

#### Abstract neuron

+ In plain English, a single neuron will pass a message to another neuron across this interface if the sum of weighted input signals from one or more neurons (summation) into it is great enough (exceeds a threshold) to cause the message transmission. 
+ This is called activation when the threshold is exceeded and the message is passed along to the next neuron.

<img src="images\image-20210907102121948.png" alt="image-20210907102121948" style="zoom:80%;" />

#### Further on Simple Neural Network

+ Neural networks are mathematical models inspired by the human brain. 

+ Neural networks, and machine learning in general, engage in two different phases. 
  + First is the <span style="color:red">**learning phase**</span>, where the model trains to perform a specific task. It could be learning how to describe photos to the blind or how to do language translations. 
  + The second phase is the <span style="color:red">**application phase**</span>, where the finished model is used. 

#### Neural Network

+ In a biological system, learning involves adjustments to the synaptic connections between neurons 
  + same for artificial neural network (ANN)
+ Neural networks are configured for specific applications, such as <span style="color:blue">**prediction**</span> or forecasting, <span style="color:blue">**pattern recognition**</span> or data classification, through a <span style="color:red">**learning process** </span>

#### What is Machine Learning

+ Webster's definition of "**to learn**" "To gain **knowledge** or **understanding** of, or **skill** in **by study**, **instruction** or **experience**" 
+ Simon's definition of "**machine learning**" "Learning denotes **changes** in the system that are **adaptive** in the sense that they enable the system to do the same task or tasks drawn from the same population **more effectively the next time**" -- Machine Learning I, 1993, Chapter 2. 
+ <span style="color:red">**Machine learning**:</span> programming computers to <span style="color:green">**optimize a performance criterion using example data**</span> or past experience. 
  + There is no need to "learn" to calculate payroll
+ <span style="color:red">**Learning is used when**: </span>
  + Human expertise does not exist (e.g.,navigating on Mars), 
  + Humans are unable to explain their expertise (e.g., speech recognition) 
  + Solution changes in time (e.g., forecasting stock market) 
  + Solution needs to be adapted to particular cases (e.g., user biometrics)

#### General Illustration of Machine Learning

<img src="images\image-20210907103205113.png" alt="image-20210907103205113" style="zoom: 50%;" />

#### Typical Learning Machines

+ Basic ML
  + Models from statistics for regression and classification
  + Decision trees 
  + Bayesian networks 
  + <span style="color: red">**Artificial neural networks (focus of INT301)**</span> 
  + Support vector machines 
  + Latent variable models 
  + Unsupervised learning
  + Manifold learning ̶ Reinforcement learning 
  + Transfer learning

## Lecture 1

### Content

> [The McCulloch-Pitts Neuron](#The McCulloch-Pitts Neuron")
>
> [Activation function](#Activation function)
>
> [Hebb’s Rule](#Hebb’s Rule): [Hebb’s rule in practice](#Hebb’s rule in practice)

#### Abstract model of a neuron

An abstract neuron $j$ with $n$ inputs: 

+ Each input $i$ transmits a real value $a_j $
+ Each connection is assigned with the weight $w_{ji}$<span style="color:red">($w_{ij}$ on PPT)</span>

The total input $S$, i.e., the sum of the products of the inputs with the corresponding weights, is compared with the threshold (equal to 0 in this case), and the outcome $X_j$<span style="color:red">($X_i$ on PPT) </span>is produced consequently

<img src="images\image-20210907103705762.png" alt="image-20210907103705762" style="zoom: 80%;" />
$$
S = \sum_i^n w_{ji}x_i
$$

#### The McCulloch-Pitts Neuron

 (1943) The authors modelled the neuron as 

+ A binary discrete-time element; 

+ With <span style="color:blue">**excitatory**</span> and <span style="color:blue">**inhibitory**</span> inputs and an excitation threshold; 

+ The network of such elements was the first model to tie the study of neural networks to the idea of computation in its modern sense.

  <img src="images\image-20210907104222797.png" alt="image-20210907104222797"  />

+ The input values $a_i^t$ from the $i$-th presynaptic neuron at any instant $t$ may be <span style="color:red">**equal either to 0 or 1 only** </span>

+ The weights of connections $w_i$ are <span style="color:red">**+1**</span> for <span style="color:red">**excitatory**</span> type connection and <span style="color:blue">**-1**</span> for <span style="color:blue">**inhibitory**</span> type connection 

+ There is an excitation threshold $\theta$ associated with the neuron.

+ Output $x^{t+1}$ of the neuron at the following instant $t+1$ is defined according to the rule
  $$
  x^{t+1} =1 \,\mbox{ if and only if } \,S^t = \sum_i w_ia_i^t \geq \theta
  $$
  
+ In the MP neuron, we shall call the instant total input $S_t$ - <span style="color:red">**instant state of the neuron**</span>

+ The <span style="color:red">**state**</span> $S^t$ of the MP neuron does not depend on the previous state of the neuron itself, but is simply 

+ The <span style="color:blue">**neuron output**</span> $x^{t+1}$ is function of its state $S^t$, therefore the output also can be written as function of discrete time 
  $$
  x(t)=g(S^t) =g( f (t))
  $$
  

### Activation function

+ The neuron output $x^{t+1}$ can be written as 
  $$
  x(t)=g(S^t) =g( f (t))
  $$
  <Span style="color:rgb(130,50,150)">***PN: which means* **</span>
  $$
  x^{t+1} = x(t)=g(S^t) =g( f (t))
  $$
  where $g$ is the <span style="color:blue">**threshold activation function**</span>
  $$ {math}
  g(S^t) = H(S^t-\theta) = 
  \begin{cases} 
  1,  & \text{if }S^t\geq\theta; \\
  0, & \text{if }S^t<\theta.
  \end{cases}
  $$
  Here $H$ is the Heaviside (unit step) function: 0.
  $$
  H(X) = \begin{cases}
  1, &x\geq\theta;\\
  0,&x<\theta.
  \end{cases}
  $$

<img src="images\image-20210907105042875.png" alt="image-20210907105042875"  />

#### MP-neuron vs brain function

+ M-P neuron made a base for a machine (network of units) capable of 
  + storing information and 
  + producing logical and arithmetical operations 
+ The next step is to realize another important function of the brain, which is 
  + to acquire <span style="color:red">**new knowledge**</span> through experience, i.e., <span style="color:red">**learning**</span>
+ These correspond to the main functions of the brain 
  + to store knowledge, and 
  + to apply the knowledge stored to solve problems

### ANN learning rules

<span style="color:red">**Learning**</span> means <span style="border-bottom:1px solid;">to change in response to experience.</span>

In a network of MP-neurons, binary weights of  connections and thresholds are fixed. The only change can be the change of pattern of connections,  which is technically expensive.

=> <span style="color:red">**Some easily changeable free parameters are needed.**</span>

<img src="images\image-20210910134126042.png" alt="image-20210910134126042"  />
$$
S_j = \sum_{i=0}^n w_{ji}a_i\\
x_j = \begin{cases}
0 \text{ if } s_j\leq0\\
1 \text{ if } s_j > 0
\end{cases}
$$

+ <span style="color:blue">**The ideal free parameters to adjust**</span>, and so to  resolve learning without changing patterns of  connections, are the <span style="color:blue">**weights of connections $w_{ji}$**</span>

#### Definition:

+ <span style="color:red">**ANN learning rule**</span>: how to adjust the  weights of connections to get desirable output. 

+ Much work in artificial neural networks focuses on  the learning rules that define

  > <span style="color:blue">**how to change the weights of connections  between neurons to better adapt a network  to serve some overall function.**</span>

+ For the first time the problem was formulated in  1940s 

  > When experimental neuroscience was limited, the  classic definitions of these learning rules came not  from biology, but from *psychological studies* of **Donald Hebb** and **Frank Rosenblatt** 

+ Hebb proposed that 

  > a particular type of <span style="color:red">use-dependent modification</span> of the connection strength of synapses might  underlie learning in the nervous system

#### Hebb’s Rule

+ (1949) Hebb proposed a <span style="color:blue">**neurophysiological postulate:**</span>

  > <span style="color:red">**" …when an axon of a cell A  **</span>
  >
  > + <span style="color:red">**is near enough to excite a cell B and **</span>
  > + <span style="color:red">**repeatedly and persistently takes part in firing it **</span>
  >
  > <span style="color:red">**some growth process or metabolic change takes  place in one or both cells such that A’s efficiency as one of the cells firing B,  is increased."**</span>

+ The simplest formalization of Hebb’s rule is <span style="color:red">**to increase weight of connection  at every next instant in the way:**</span>
  $$
  w_{ji}^{k+1} = w_{ji}^k +\Delta w_{ji}^k\\
  \text{where } \Delta w_{ji}^k=Ca_i^kx_j^k 
  $$
  here

  + $w_{ji}^ k$ is the weight of connection at instant $k $
  + $w_{ji}^ k+1$ is the weight of connection at the following instant $k+1 $
  + $w_{ji}^ k$  is increment by which the weight of connection is  enlarged  
  + $C$ is positive coefficient which determines learning rate 
  + $a_i^k$ is input value from the presynaptic neuron at instant $k $
  + $x_j^k$ is output of the postsynaptic neuron at the same instant $k$

+ Thus the weight of connection changes at the next  instant only if both preceding input via this  connection and the resulting output  simultaneously are not equal to 0
+ Equations emphasize the <span style="color:red">**correlation**</span>  <span style="color:red">nature of a Hebbian synapse</span>

+ Hebb’s original learning rule 
  + referred exclusively to excitatory synapses, and  
  + has the <span style="border-bottom:1px solid;">unfortunate property</span> that it can only  increase synaptic weights, thus washing out the  distinctive performance of different neurons in a  network, as the connections drive into  saturation ...

+ However, when the Hebbian rule is augmented by a  formalization rule, e.g., keep constant the total strength of  synapses upon a given neuron, it tends to “sharpen” a  neuron’s predisposition “without a teacher”, causing its  firing to become better correlated with a cluster of  stimulus patterns. 
+ For this reason, Hebb's rule plays an important role in  studies of many ANN algorithms, such as unsupervised  learning or self-organization, which we will study later.

#### Hebb’s rule in practice

**Input unit** 

<img src="images\image-20210910113615207.png" alt="image-20210910113615207" style="zoom:80%;" />

**Input unit** 

<img src="images\image-20210910113737442.png" alt="image-20210910113737442" style="zoom: 67%;" />

**Input unit** 

<img src="images\image-20210910113809177.png" alt="image-20210910113809177" style="zoom: 67%;" />

**Input unit**

<img src="images\image-20210910113859520.png" alt="image-20210910113859520" style="zoom: 67%;" />

**Input unit**

<img src="images\image-20210910114049883.png" alt="image-20210910114049883" style="zoom: 67%;" />

**Input unit**

<img src="images\image-20210910114213034.png" alt="image-20210910114213034" style="zoom: 67%;" />

**Input unit** 

<img src="images\image-20210910114240377.png" alt="image-20210910114240377" style="zoom: 67%;" />

**Input unit** 

<img src="images\image-20210910114302602.png" alt="image-20210910114302602" style="zoom: 67%;" />

**Input unit** 

<img src="images\image-20210910114331930.png" alt="image-20210910114331930" style="zoom: 67%;" />

**Input unit** 

<img src="images\image-20210910133151581.png" alt="image-20210910133151581" style="zoom: 67%;" />

**Input unit** 

<img src="images\image-20210910133235332.png" alt="image-20210910133235332" style="zoom: 67%;" />

**Input unit** 

<img src="images\image-20210910133306227.png" alt="image-20210910133306227" style="zoom: 67%;" />

## Lecture 2

**Supervised learning model: Perceptron**

### Content

>[Perceptron (1958)](#Perceptron)

### Recall: Machine learning and ANN

+ Like human learning from past experiences. 
+ A computer does not have “experiences”. 
+ <span style="color:blue">**A computer system learns from data**</span>, which represent some “past experiences” of an application domain. 
+ <span style="color:red">**Our focus**</span>: learn <span style="color:blue">**a target function**</span> that can be used to predict the values of a discrete class attribute, e.g., <span style="color:blue">**yes**</span> or <span style="color:blue">**no**</span>, and <span style="color:blue">**high**</span> or <span style="color:blue">**low**</span>. 
+ The task is commonly called: <span style="color:red">**supervised learning**</span>.

### The data and the goal

+ <span style="color:red">**Data:**</span> A set of data records (also called examples, instances or cases) described by 
  + <span style="color:blue">**k attributes**</span>: $A_1, A_2,...A_k$. 
  + <span style="color:blue">**a class**</span>: Each example is labelled with a predefined class.
+ <span style="color:red">**Data:**</span> To learn a <span style="color:blue">**classification model**</span> from the data that can be used to predict the classes of new (future, or test) cases/instances.

#### An example: data (loan application)

<span style="color:red">**Approve or not**</span>

<img src="images\image-20210914091611497.png" alt="image-20210914091611497" style="zoom:80%;" />

#### An example: the learning task

+ <span style="color:red">**Learn a classification model from the data**</span> from the data
+ Use the model to classify future loan applications into
  + <span style="color:blue">**Yes (approve)** and</span> 
  + <span style="color:blue">**No (disapprove)**</span>

+ What is the class for following case/instance?

  <img src="images\image-20210914091906607.png" alt="image-20210914091906607" style="zoom:80%;" />

### Supervised vs. unsupervised Learning

+ <span style="color:red">**Supervised learning**</span>: classification is seen as supervised learning from examples. 
  + <span style="color:blue">**Supervision**</span>: The data (observations, measurements, etc.) are labeled with predefined classes. It is like that a “teacher” gives the classes (<span style="color:blue">**supervision**</span>). 
  + Test data are classified into these classes too. 
+ <span style="color:red">**Unsupervised learning** (e.g. clustering)</span> 
  + <span style="color:blue">**Class labels of the data are unknown**</span> 
  + Given a set of data, the task is to establish the existence of classes or clusters in the data

#### Supervised learning process: two steps

+ <span style="color:red">**Learning (training)**</span>: Learn a model using the <span style="color:blue">**Learning (training)**</span>

+ <span style="color:red">**Testing**:</span> Test the model using <span style="color:blue">**unseen test data**</span> to assess the model accuracy
  $$
  \begin{align*}
  Accuracy=\frac{\text{Number of correct classifications}}{\text{Total number of test cases}}.
  \end{align*}
  $$
  <img src="images\image-20210914092505874.png" alt="image-20210914092505874" style="zoom:80%;" />

#### What do we mean by learning?

+ <span style="color:red">**Given**</span> 

  + <span style="color:blue">**a data set D**</span> 
  + <span style="color:blue">**a task T**</span> 
  + **<span style="color:blue">a performance measure M</span>**

  a computer system is said to <span style="color:red">**learn**</span> from D to perform the task T if after learning the system’s performance on T improves as measured by M. 

+ In other words, the learned model helps the system to perform T better as <span style="color:blue">**compared to no learning**</span>.

##### Fundamental assumption of learning

> <span style="color:red">**Assumption**</span>: The distribution of training examples is <span style="color:blue">**identical**</span> to the distribution of test examples (including future unseen examples).

+ In practice, this assumption is often violated to certain degree. 
+ Strong violations will clearly result in poor classification accuracy. 
+ <span style="color:blue">**To achieve good accuracy on the test data, training examples must be sufficiently representative of the test data.**</span>

### Perceptron (1958)

+ Rosenblatt (1958) explicitly considered the problem of <span style="color:red">**pattern recognition**</span>, where a “teacher” is essential. 
+ Perceptrons are neural networks that change with “experience” using <span style="color:red">**error-correcting rule**</span>. 
+ According to the rule, <span style="color:blue">**weight of a response unit changes when it makes erroneous response to stimuli presented to the network**</span>.

#### ANN for Pattern Recognition

+ <span style="color:rgb(0,112,192)">**Training data**</span>: set of sample pairs $(x, y)$. 

+ Network (model, classifier) **<span style="color:red">adjusts its connection weights</span> <span style="color:blue">according to the errors</span>** between target and network output

  <img src="images\image-20210914093643692.png" alt="image-20210914093643692" style="zoom:80%;" />

  **Supervised learning** is mainly applied in classification/prediction. 

#### Perceptron

+ The simplest architecture of perceptron comprises two layers of idealized “neurons”, which we shall call <span style="color:blue">***"units" of the network***</span>

  <img src="images\image-20210914094058954.png" alt="image-20210914094058954"  />

+ There are

  + one layer of input units, and 
  + one layer of output units

  in the perceptron

+ The two layers are fully interconnected, i.e., every input unit is connected to every output unit
+ Thus, <span style="color:blue">***processing elements***</span> of the perceptron are the <span style="color:rgb(51,51,153)">***abstract neurons***</span>
+ Each processing element has the same input comprising total input layer, but individual outputs with individual connections and therefore different weights of connections.

The total input to the output unit $j$ is
$$
\begin{align*}
S_j = \sum_{i=0}^nw_{ji}a_i
\end{align*}
$$
$a_i$: input value from the $i$-th input unit

$w_{ji}$: the weight of connection btw $i$-th input and $j$-th output units

+ The sum is taken <span style="color:red">**over all n+1**</span> inputs units connected to the output unit <span style="color:red">**$j$**</span>. 
+ There is special <span style="color:red">**bias input**</span> unit <span style="color:red">**number 0**</span> in the input layer

+ The bias unit always produces inputs $a_0$ of the fixed values of +1. 
+ The input $a_0$ of <span style="color:red">**bias unit**</span> functions as a constant value in the sum. 
+ The <span style="color:red">**bias unit connection**</span> to output unit $j$ has a weight $w_{j0}$ adjusted in the same way as all the other weights 

+ The output value of $X_j$ the output unit $j$ depends on whether the weighted sum is above or below the unit's threshold value.

+ $X_j$ is defined by the unit's threshold activation function.
  $$
  \begin{align*}
  X_j =f(S_j)=
  \begin{cases}
  1, S_j\geq\theta_j\\
  0,S_j<\theta_j
  \end{cases}
  \end{align*}
  $$

**Definition:** 

the ordered set of instant outputs of all units in the output layer $X=\{X_0,X_1,...,X_n\}$ constitutes an <span style="color:red">**output vector**</span> of the network

+ The instant output $X_j$ of the $j$-th unit in the output layer constitutes the $j$-th component of the output vector.
+ <span style="color:blue">**Weight $w_{ji}$**</span> of connections between the two layers <span style="color:blue">**are changed**</span> according to <span style="color:red">**perceptron learning rule**</span>, so the network is more likely to produce the desired output in response to certain inputs.
+ **<span style="color:rgb(0,128,0)">The process of weights adjustment is called</span> <span style="color:red">perceptron learning</span> <span style="color:rgb(0,128,0)">(or training)</span>.**

#### Perceptron Training

Every processing element computes an output according its state and threshold:
$$
\begin{align*}
S_j = \sum_{i=0}^nw_{ji}a_i \to X_j =f(S_j)=
\begin{cases}
1, S_j\geq\theta_j\\
0,S_j<\theta_j
\end{cases} \to e_j=(t_j-X_j)
\end{align*}
$$
The network instant outputs $X_j$ are then compared to the desired outputs specified in the training set.

> <span style="color:blue">**The error of an output unit is the difference between the target output and the instant one.**</span>

> <span style="color:blue">**The error are computed and used to re-adjust the values of the weights of connections.**</span>

**<span style="color:rgb(25,25,78)">The weights re-adjustment is done in such a way that the network is – on the whole – more likely to give the desired response next time.</span>**

##### Perceptron Updating of the Weights

The goal of the training session is to arrive at a single set of weights that allow each of the mappings in the training set to be done successfully by the network.

1. Compute <span style="color:rgb(255,0,102)">**error**</span> of every output unit
   $$
   \begin{align*}
   e_j=(t_j-X_j)
   \end{align*}
   $$
   where $t_j$ is the target value for output unit $j$ 

   $X_j$ is the instant output produced by output unit $j$

Having the errors computed, 

 2. Update the weights
    $$
    \begin{align*}
    w_ji = w_ji+\triangle w_{ji}
    \end{align*}
    $$
    where  (**<span style="color:blue">Perceptron learning rule</span>**)
    $$
    \begin{align*}
    \triangle w_{ji} =Ce_ja_i=C(t_j-X_j)a_i
    \end{align*}
    $$

+ A sequential learning procedure for updating the weights. 

+ Perceptron training algorithm (<span style="color:red">**delta rule**</span>)

  <img src="images\image-20210914101119552.png" alt="image-20210914101119552" style="zoom: 67%;" />

##### Example

+ Define our “features”:

  | Taste | Sweet = 1, Not_Sweet = 0   |
  | ----- | -------------------------- |
  | Seeds | Edible = 1, Not_Edible = 0 |
  | Skin  | Edible = 1, Not_Edible = 0 |

  For output:

  ```matlab
  Good_Fruit = 1 
  Not_Good_Fruit = 0
  ```

  Let’s start with no knowledge:

  <img src="images\image-20210914101455240.png" alt="image-20210914101455240" style="zoom: 67%;" />

+ To train the perceptron, we will show it each example and have it categorize each one.
+ Since it’s starting with no knowledge, it is going to make mistakes. When it makes a mistake, we are going to adjust the weights to make that mistake less likely in the future.

+ When we adjust the weights, we’re going to take relatively small steps to be sure we don’t over-correct and create new problems.

+ We’re going to learn the category “good fruit” defined as anything that is sweet.

  ```matlab
  Good fruit = 1
  Not good fruit = 0
  ```

  **Show it a banana:**

  <img src="images\image-20210914101722693.png" alt="image-20210914101722693" style="zoom: 67%;" />

  + In this case we have: 

    (1 X 0) = 0 

    \+ (1 X 0) = 0 

    \+ (0 X 0) = 0 

  + It adds up to 0.0. 

  + Since that is less than the threshold (0.40), we responded “no.” 

  + Is that correct? No.

+ Since we got it wrong, we need to change the weights. We’ll do that using the delta rule (delta for change).
  $$
  \begin{align*}
  ∆w = \text{learning rate} \times \text{(teacher \bf{- output)}} \times \text{\bf{input}}
  \end{align*}
  $$

The three parts of that are: 

+ <span style="color:red">**Learning rate:**</span> We set that ourselves. Set large enough that learning happens in a reasonable amount of time; and also small enough to avoid too fast. Here pick 0.25. 
+ <span style="color:red">**(teacher - output):**</span> The teacher knows the correct answer (e.g., that a banana should be a good fruit). In this case, the teacher says 1, the output is 0, so (1 - 0) = 1. 
+ <span style="color:red">**Input:**</span> That’s what came out of the node whose weight we’re adjusting. For the first node, 1.

+ To pull it together: 

  − Learning rate: 0.25. 

  − (teacher - output): 1. 

  − input: 1.
  $$
  \begin{align*}
  ∆w = 0.25 \times 1 \times 1 = 0.25.
  \end{align*}
  $$

+ Since it’s a ∆w, it’s telling us how much to change the first weight. In this case, we’re adding 0.25 to it.

Let’s think about the delta rule:  <span style="color:red">**(teacher - output)**</span>

+ If we get the categorization right, (teacher - output) will be zero (the right answer minus itself).
+ In other words, if we get it right, we won’t change any of the weights. As far as we know we have a good solution, why would we change it?
+ If we get the categorization wrong, (teacher - output) will either be -1 or +1. 
  + If we said “yes” when the answer was “no”, we’re too high on the weights and we will get a (teacher - output) of -1 which will result in reducing the weights. 
  + If we said “no” when the answer was “yes”, we’re too low on the weights and this will cause them to be increased.

+ Input: 
  + If the node whose weight we’re adjusting sent in a 0, then it didn’t participate in making the decision. In that case, it shouldn’t be adjusted. Multiplying by zero will make that happen. 
  + If the node whose weight we’re adjusting sent in a 1, then it did participate and we should change the weight (up or down as needed) if the corresponding output wrong.

+ How do we change the weights for banana?

  | Feature: | Learning rate: | (teacher - output): | Input: | ∆w    |
  | -------- | -------------- | ------------------- | ------ | ----- |
  | taste    | 0.25           | 1                   | 1      | +0.25 |
  | seeds    | 0.25           | 1                   | 1      | +0.25 |
  | skin     | 0.25           | 1                   | 0      | 0     |

  **Here it is with the adjusted weights:**

  <img src="images\image-20210914102819927.png" alt="image-20210914102819927" style="zoom: 67%;" />

+ To continue training, we show it the next example, adjust the weights…

+ We will keep cycling through the examples until we go all the way through one time without making any changes to the weights. At that point, the concept is learned.

  **Show it a pear:**

  <img src="images\image-20210914102934394.png" alt="image-20210914102934394" style="zoom: 67%;" />

+ How do we change the weights for pear?

  | Feature: | Learning rate: | (teacher - output): | Input: | ∆w    |
  | -------- | -------------- | ------------------- | ------ | ----- |
  | taste    | 0.25           | 1                   | 1      | +0.25 |
  | seeds    | 0.25           | 1                   | 0      | 0     |
  | skin     | 0.25           | 1                   | 1      | +0.25 |

  **Here it is with the adjusted weights:**

  <img src="images\image-20210914103102386.png" alt="image-20210914103102386" style="zoom: 67%;" />

  **Show it a lemon:**

  <img src="images\image-20210914103119188.png" alt="image-20210914103119188" style="zoom: 67%;" />

+ How do we change the weights for lemon?

  | Feature: | Learning rate: | (teacher - output): | Input: | ∆w   |
  | -------- | -------------- | ------------------- | ------ | ---- |
  | taste    | 0.25           | 0                   | 0      | 0    |
  | seeds    | 0.25           | 0                   | 0      | 0    |
  | skin     | 0.25           | 0                   | 0      | 0    |

  **Show it a strawberry:**

  <img src="images\image-20210914103211665.png" alt="image-20210914103211665" style="zoom: 67%;" />

+ How do we change the weights for strawberry?

  | Feature: | Learning rate: | (teacher - output): | Input: | ∆w   |
  | -------- | -------------- | ------------------- | ------ | ---- |
  | taste    | 0.25           | 0                   | 1      | 0    |
  | seeds    | 0.25           | 0                   | 1      | 0    |
  | skin     | 0.25           | 0                   | 1      | 0    |

  **Here it is with the adjusted weights:**

  <img src="images\image-20210914103354323.png" alt="image-20210914103354323" style="zoom: 67%;" />

  **Show it a green apple:**

  <img src="images\image-20210914103417211.png" alt="image-20210914103417211" style="zoom: 67%;" />

+ If you keep going, you will see that this perceptron can correctly classify the examples that we have.

## Lecture 3

### Content

>TBD

### Recall: Perceptron

<img src="images\image-20210920150549671.png" alt="image-20210920150549671" style="zoom: 67%;" />

#### Review of Perceptron rule

+ <span style="color:blue">**Perceptron rule**</span>: a sequential learning procedure for updating the weights.

+ Perceptron Learning Algorithm

  <img src="images\image-20210920150707941.png" alt="image-20210920150707941" style="zoom: 70%;" />

### Perceptron Rule: Further Discussion

+ A weight of connection changes <span style="color:blue">**only if**</span> both the input value and the error of the output unit are not equal to 0. 
  +  If the output is correct ($y_e=o_e$) the weights $w_i$ are not changed 
  + If the output is incorrect ($y_e≠o_e$) the weights $w_i$ are changed such that the output of the perceptron for the new weights is closer to $y_e$. 
+ The algorithm <span style="color:red">**converges**</span> to the correct classification, if 
  + the training data is <span style="color:red">**linearly separable**</span>; 
  + and the learning rate is sufficiently small, usually set below 1, which <span style="color:blue">**determines the amount of correction made in a single iteration**.</span>

#### Perceptron convergence Theorem

> For any data set that’s <span style="color:red">**linearly separable**</span>, the learning rule is guaranteed to find a solution in a finite number of steps.

**Assumptions:**

+ At least one such set of weights, $w^*$, exists 
+ There are a finite number of training patterns. 
+ The threshold function is uni-polar (output is 0 or 1).

### Network Performance for Perceptron 

The network performance during training session can be measured by a <span style="color:red">**root-mean-square (RMS) error value**</span>
$$
\begin{align*}
R M S=\sqrt{\frac{\sum_{p=0}^{n_{p}} \sum_{j=0}^{n_{o}} e_{j p}^{2}}{n_{p} n_{o}}}=\sqrt{\frac{\sum_{p=0}^{n_{p}} \sum_{j=0}^{n_{o}}\left(t_{j p}-X_{j p}\right)^{2}}{n_{p} n_{o}}}
\end{align*}
$$
where 

$n_p$ is the number of patterns in the training set and 

$n_o$ is the number of units in the output layer

+ As the target output values $t_{jp}$ and $n_p$ and $n_o$ numbers are constants, <span style="color:green">the RMS error is a function of the instant output values Xjp only</span>
  $$
  \begin{align*}
  R M S=\sqrt{\frac{\sum_{p=0}^{n_{p}} \sum_{j=0}^{n_{o}}\left(t_{j p}-X_{j p}\right)^{2}}{n_{p} n_{o}}}
  \end{align*}
  $$

+ In turn, the instant outputs $X_{jp}$ are functions of the input values $a_{ip}$ , which are also constants, and of the weights of connections $w_{ji}$
  $$
  \begin{align*}
  X_{j p}=f\left(S_{j p}=\sum_{i=0}^{n_{i}} w_{j i} a_{i p}\right)=\tilde{f}\left(w_{j i}, a_{i p}\right)
  \end{align*}
  $$
  So the <span style="color:red">**performance of the network**</span> measured by the RMS error also <span style="color:red">**is function of the weights of connections only**</span>
  $$
  \begin{align*}
  \textit{RMS} = \textit{F}(w_{ji},a_{ip})
  \end{align*}
  $$

+ Performance of the network measured by the RMS error *is function of the weights of connections only.*

+ >The <span style="color:red">best performance of the network</span> corresponds to the <span style="color:red">minimum of the RMS error</span>, and we adjust the weights of connections in order to get that minimum.

#### RMS on Training Set

<img src="images\image-20210920152832469.png" alt="image-20210920152832469" style="zoom:80%;" />
$$
\begin{align*}
\textit{RMS} = \textit{F}(w_{ji},a_{ip})
\end{align*}
$$
Shown is a <span style="color:red">learning curve</span>, i.e., <span style="color:blue">dependence of the RMS error on the number of iterations for the training set.</span>

+ Initially, the adaptable weights are all set to small random values, and the network does not perform very well. 
+ As weights are adjusted during training, performance improves; when the error rate is low enough, training stops and the network is said to have <span style="color:red">converged</span>.

#### RMS on the Training/Testing Data

<span style="color:blue">**RMS on the Training Data**</span>
$$
\begin{align*}
R M S^{training}=\sqrt{\frac{\sum_{p=0}^{n_{p}} \sum_{j=0}^{n_{o}}\left(t_{j p}-X_{j p}^{trained}\right)^{2}}{n_{p} n_{o}}}
\end{align*}
$$
**<span style="color:blue">RMS on the Testing Data</span>**
$$
\begin{align*}
R M S^{testing}=\sqrt{\frac{\sum_{p=0}^{n_{p}} \sum_{j=0}^{n_{o}}\left(t_{j p}-X_{j p}^{predicted}\right)^{2}}{n_{p} n_{o}}}
\end{align*}
$$

### Recall: Perceptron Convergence Theorem

> <span style="color:red">If</span> <span style="color:blue">a set of weights that allow the perceptron to respond correctly to all of the training patterns exists,</span> <span style="color:red">then</span> <span style="color:blue">the perceptron’s learning method will find the set of weights, and it will do it in a finite number of iterations.</span>**(Rosenblatt, 1962)**

<img src="images\image-20210920153529158.png" alt="image-20210920153529158" style="zoom:80%;" />

### More on Perceptron Convergence

+ There might be another possibility during a training session: 
  + eventually performance stops improving, and the RMS error does not get smaller regardless of number of iterations. 
+ That means the network has <span style="color:red">**failed**</span> to learn all of the answers correctly. 
+ <span style="color:blue">**If the training is successful**</span>, the perceptron is said 
  + to have gone through the supervised learning, and 
  + is able to classify patterns similar to those of the training set.

### Perceptron As a Classifier

For $d$-dimensional data, perceptron consists of d-weights, a bias, and a thresholding activation function. For 2D data example，we have:

<img src="images\image-20210920153755934.png" alt="image-20210920153755934" style="zoom:80%;" />

If we group the weights as a vector w , the net output y can be expressed as:
$$
\begin{align*}
y = g(w\vdot x+w_0 )
\end{align*}
$$

#### Further Discussion Perceptron As a Classifier

+ A perceptron training is to compute weight vector: $W=[w_0,w_1,w_2,...,w_p]$ to correctly classify all the training examples.

  E.g., consider when p=2

  <img src="images\image-20210920154026573.png" alt="image-20210920154026573" style="zoom:80%;" />

  W. X is a <span style="color:green">**hyperplane**</span>, which in 2d is a straight line.

+ For 2 classes, view net output as a <span style="color:red">discriminant function y(x, w)</span>, where: 

  y(x, w) = 1 , if x in class 1 (C1) 

  y(x, w) = -1, if x in class 2 (C2)

  **Example** 

  <img src="images\image-20210920154225351.png" alt="image-20210920154225351" style="zoom: 80%;" />

+ For m classes, a classifier should partition the feature space into <span style="color:red">m **decision regions**</span> 
  + The <span style="color:blue">**line or curve**</span> separating the classes is the <span style="color:red">**decision boundary**</span>. 
  + In more than 2 dimensions, this is a hyperplane.

#### Further on Perceptron Decision Boundary

A perceptron represents a <span style="color:blue">**hyperplane decision surface**</span> in d-dimensional space, for example, a line in 2D, a plane in 3D, etc.

<span style="color:red">The equation of the hyperplane is $w\vdot x^T = 0$</span>

This is the equation for points in x-space that are <span style="color:blue">**on**</span> the boundary

#### Decision boundary of Perceptron

<img src="images\image-20210920154613124.png" alt="image-20210920154613124" style="zoom: 67%;" />

+ Perceptron is able to represent some useful functions 
+ But functions that are not linearly separable (e.g. XOR) are not representable

##### Example of Perceptron Decision Boundary

> Decision surface is the surface at which the output of the unit is precisely equal to the threshold, i.e. $∑w_i x_i =θ$

<img src="images\image-20210920155008350.png" alt="image-20210920155008350" style="zoom: 50%;" />

<img src="images\image-20210920155105792.png" alt="image-20210920155105792" style="zoom: 50%;" />

<img src="images\image-20210920155212617.png" alt="image-20210920155212617" style="zoom: 50%;" />

<img src="images\image-20210920155408911.png" alt="image-20210920155408911" style="zoom: 50%;" />

### Linear Separability Problem

+ If two classes of patterns can be separated by a decision boundary, represented by the linear equation
  $$
  \begin{align*}
  b+\sum_{i=1}^nx_iw_i=0
  \end{align*}
  $$
  then they are said to be <span style="color:red">**linearly separable**</span> and the perceptron can correctly classify any patterns 

  NOTE: <span style="color:red">**without the bias term, the hyperplane will be forced to intersect origin.**</span>

+ Decision boundary (i.e., *W, b* ) of linearly separable classes can be determined either by some learning procedures, or by solving linear equation systems based on representative patterns of each classes 
+ <span style="color:red">If such a decision boundary does not exist, then the two classes are said to be linearly inseparable. </span>
+ Linearly inseparable problems cannot be solved by the simple perceptron network, more sophisticated architecture is needed.

+ Examples of linearly inseparable classes 

  <span style="color:blue">**Logical XOR (exclusive OR) function**</span> 

  patterns (bipolar) decision boundary 

  <img src="images\image-20210920155844043.png" alt="image-20210920155844043" style="zoom:67%;" />
  $$
  \begin{align*}
  \begin{array}{ccc}
  x_{1} & x_{2} & y \\
  -1 & -1 & -1 \\
  -1 & 1 & 1 \\
  1 & -1 & 1 \\
  1 & 1 & -1
  \end{array}
  \end{align*}
  $$
  No line can separate these two classes, as can be seen from the fact that the following linear inequality system has no solution 
  $$
  \left\{\begin{array}{l}
  \boldsymbol{b}-\boldsymbol{w}_{1}-\boldsymbol{w}_{2}<0 \quad(1)\\
  \boldsymbol{b}-\boldsymbol{w}_{1}+\boldsymbol{w}_{2} \geq 0 \quad(2)\\
  \boldsymbol{b}+\boldsymbol{w}_{1}-\boldsymbol{w}_{2} \geq 0 \quad(3)\\
  \boldsymbol{b}+\boldsymbol{w}_{1}+\boldsymbol{w}_{2}<0 \quad(4)
  \end{array}\right.
  $$
  because we have b < 0 from (1) + (4), and b >= 0 from (2) + (3), which is a contradiction 

+ Examples of linearly separable classes 

  -<span style="color:blue"> **Logical AND function**</span>

  patterns (bipolar) decision boundary
  $$
  \begin{align*}
  \begin{array}{rrrr}
  x_{1} & x_{2} & y & w_{1}=1 \\
  -1 & -1 & -1 & w_{2}=1 \\
  -1 & 1 & -1 & b=-1 \\
  1 & -1 & -1 & \theta=0 \\
  1 & 1 & 1 & \color{green}{-1+x_{1}+x_{2}=0}
  \end{array}
  \end{align*}
  $$
  <img src="images\image-20210920160010048.png" alt="image-20210920160010048" style="zoom:67%;" />

  -<span style="color:blue">**Logical OR function** </span>

  patterns (bipolar) decision boundary

  <img src="images\image-20210920160051626.png" alt="image-20210920160051626" style="zoom:67%;" />
  $$
  \begin{align*}
  \begin{array}{rrrr}
  x_{1} & x_{2} & y & w_{1}=1 \\
  -1 & -1 & -1 & w_{2}=1 \\
  -1 & 1 & -1 & b=1 \\
  1 & -1 & 1 & \theta=0 \\
  1 & 1 & 1 & \color{green}{1+x_{1}+x_{2}=0}
  \end{array}
  \end{align*}
  $$

### Tips for Building ANN

Formulating neural network solutions for particular problems is a multi-stage process: 

1. Understand and specify the problem in terms of inputs and required outputs 
2. Take the simplest form of network you think might be able to solve your problem 
3. Try to find the appropriate connection weights (including neuron thresholds) so that the network produces the right outputs for each input in its training data 
4. Make sure that the network works on its training data and test its generalization by checking its performance on new testing data 
5. If the network doesn’t perform well enough, go back to stage 3 and try harder 
6. If the network still doesn’t perform well enough, go back to stage 2 and try harder 
7. If the network still doesn’t perform well enough, go back to stage 1 and try harder 
8. Problem solved – or not

## Lecture 4

### Content

>TBD

### The Gradient Descent Rule

- Perceptron rule fails <span style="color:orange">**if data is not linearly separable**</span>
- <span style="color:red">**Idea:** </span>uses gradient descent to search the <span style="color:blue">**hypothesis space**</span>
  - perceptron rule cannot be used (not differentiable)
  - hence, an <span style="color:blue">**unthresholded linear unit**</span> is an appropriate error measure:
$$
E(w)=\frac{1}{2} \sum_{e}\left(y_{e}-o_{e}\right)^{2}
$$
- To understand gradient descent, it is helpful to visualize the entire hypothesis space with
  - all possible weight vectors
  - associated E values

The objective is to minimize the following error:
$$
E(\mathrm{w})=\frac{1}{2} \sum_{e}\left(y_{e}-o_{e}\right)^{2}
$$
- The training is a process of minimizing the error $E(w)$ in the steepest direction (most rapid decrease), <span style="color:red">**that is in direction opposite to the gradient**</span>
$$
\nabla E(w)=\left[\partial E / \partial w_{0}, \partial E / \partial w_{1}, \ldots, \partial E / \partial w_{d}\right]
$$
​		which leads to the <span style="color:blue">**that is in direction opposite to the gradient**</span>:
$$
w_{i}=w_{i}-\eta \partial E / \partial w_{i}
$$

#### Error Surface

+ the axes $w_0,w_1$ represent possible values for the two weights of a simple linear unit

  <img src="images\image-20210928162248479.png" alt="image-20210928162248479" style="zoom:80%;" />

error surface must be **parabolic** with a **single global minimum**

#### Moving Downhill: Move in direction of negative derivative

<img src="images\image-20210928162337542.png" alt="image-20210928162337542" style="zoom: 67%;" />

<img src="images\image-20210928162434035.png" alt="image-20210928162434035" style="zoom: 67%;" />

#### Illustration of Gradient Descent

<img src="images\image-20210928162508604.png" alt="image-20210928162508604" style="zoom: 67%;" />

<img src="images\image-20210928162528226.png" alt="image-20210928162528226" style="zoom: 67%;" />

<img src="images\image-20210928162549390.png" alt="image-20210928162549390" style="zoom: 67%;" />

<img src="images\image-20210928162620541.png" alt="image-20210928162620541" style="zoom: 67%;" />

- The weight update can be derived as follows:
$$
\begin{aligned}
\partial E / \partial w &=\partial\left(\frac{1}{2} \sum_{e}\left(y_{e}-o_{e}\right)^{2}\right) / \partial w_{i} \\
&=\frac{1}{2} \sum_{e} \partial\left(y_{e}-o_{e}\right)^{2} / \partial w_{i} \\
&=\frac{1}{2} \sum_{e} 2\left(y_{e}-o_{e}\right) \partial\left(y_{e}-o_{e}\right) / \partial w_{i} \\
&=\sum_{e}\left(y_{e}-o_{e}\right) \partial\left(y_{e}-w_{i} x_{i e}\right) / \partial w_{i} \\
&=\sum_{e}\left(y_{e}-o_{e}\right)\left(-x_{i e}\right)
\end{aligned}
$$
where $x_{i e}$ denotes the $i$-th component of the example $e$. The gradient descent training rule becomes:
$$
w_{i}=w_{i}+\eta \sum_{e}\left(y_{e}-o_{e}\right) x_{i e}
$$

### Gradient Descent Learning Algorithm

- Initialization: Examples $\left\{\left(x_{e}, y_{e}\right)\right\}_{e=1}^{N}$, initial weights $w_{i}$ set to small random values, learning rate parameter $\eta$
- **Repeat**
  for each training example $\left(x_{e}, y_{e}\right)$
  - calculate the network output: $o_{e}=\sum_{i=0}^{d} w_{i} x_{i e}$
  - if the Perceptron does not respond correctly, compute weight corrections:
$$
\Delta w_{i}=\Delta w_{i}+\eta\left(y_{e}-o_{e}\right) x_{i e}
$$
​		update the weights with the <span style="color:red">**accumulated error**</span> from all examples
$$
w_{i}=w_{i}+\Delta w_{i} \quad \begin{array}{l}
\text{ Gradient Descent Rule} \\
\end{array}
$$
​		 **until** termination condition is satisfied.

#### Example

- Suppose an example of Perceptron which accepts two inputs $x_{1}$ and $x_{2}$ with weights $w_{1}=0.5$ and $w_{2}=0.3$ and $w_{0}=-1$, learning rate $=1$.
- Let the example is given: $x_{1}=2, x_{2}=1, y=0$ The network output of the Perceptron is :
$$
o=2 * 0.5+1 * 0.3-1=0.3
$$
- The weight updates according to the gradient descent algorithm will be:

$$
\begin{array}{l}
\Delta w_{1}=(0-0.3) * 2=-0.6 \\
\Delta w_{2}=(0-0.3) * 1=-0.3 \\
\Delta w_{0}=(0-0.3) * 1=-0.3
\end{array}
$$

#### Example

- Let another example is given: $x_{1}=1, x_{2}=2, y=1$
- The network output of the Perceptron is :
$$
o=1 * 0.5+2 * 0.3-1=0.1
$$
The weight updates according to the gradient descent algorithm will be:
$$
\begin{array}{l}
\Delta w_{1}=-0.6+(1-0.1) * 1=0.3 \\
\Delta w_{2}=-0.3+(1-0.1) * 2=1.5 \\
\Delta w_{0}=-0.3+(1-0.1) * 1=0.6
\end{array}
$$
If there are no more examples, the weights will be modified as follows:
$$
\begin{array}{l}
w_{1}=0.5+0.3=0.8 \\
w_{2}=0.3+1.5=1.8 \\
w_{n}=-1+0.6=1.6
\end{array}
$$

### Incremental gradient descent

The gradient descent rule faces two difficulties in practice: 

\- it converges very slowly

\- if there are multiple local minima in the error surface, then there is no guarantee that it will find the global minimum

- That is why, a <span style="color:red">**stochastic version**</span> called <span style="color:red">**incremental gradient descent**</span> rule is developed to overcome these difficulties. <span style="color:blue">**Whereas the gradient descent rule updates the weights after calculating the whole error accumulated from all examples, the incremental version approximates the gradient descent error decrease by updating the weights after each training example.**</span>

- Incremental gradient descent is implemented
$$
w_{i}=w_{i}+\eta\left(y_{e}-o_{e}\right) x_{i e} \quad \text { where } \quad o_{e}=\sum_{i=0}^{d} w_{i} x_{i e}
$$
#### Incremental Gradient Descent Learning Algorithm

Initialization: Examples $\left\{\left(x_{e}, y_{e}\right)\right\}_{e=1}^{N}$, initial weights $w_{i}$ set to small random values, learning rate parameter $\eta$

**Repeat**
for each training example $\left( x_{e}, y_{e}\right)$

- calculate the network output:
$$
o_{e}=\sum_{i=0}^{d} w_{i} x_{i e}
$$
- If the Perceptron does not respond correctly update the weights:
$$
w_{i}=w_{i}+\eta\left(y_{e}-o_{e}\right) x_{i e}
$$
**until** termination condition is satisfied.

### Sigmoidal Perceptrons

- The simple single-layer Perceptrons with threshold or linear activation functions are not generalizable to more powerful learning mechanisms like multilayer neural networks.
That is why, single-layer Perceptrons with sigmoidal activation functions are developed.
- The sigmoidal Perceptron produces output:

$$
o=\sigma(S)=\frac{1}{1+e^{-S}}
$$

<img src="images\image-20210928165957967.png" alt="image-20210928165957967" style="zoom: 80%;" />

​	 	where:
$$
\begin{align*}
S=\sum_{i=0}^{d} w_{i} x_{x}
\end{align*}
$$

#### Training Sigmoidal Perceptrons

- The gradient descent rule for training sigmoidal Perceptrons is again:
$$
w_{i}=w_{i}-\eta \partial E / \partial w_{i}
$$
- The difference is in the error derivative $\partial E / \partial w_{i}$ which due to the use of the sigmoidal function $\sigma(s)$ becomes:
$$
\begin{array}{l}
\partial E / \partial w_{i}=\partial\left((1 / 2) \Sigma_{e}\left(y_{e}-o_{e}\right)^{2}\right) / \partial w_{i} \\
=(1 / 2) \Sigma_{e} \partial\left(y_{e}-o_{e}\right)^{2} / \partial w_{i} \\
=(1 / 2) \Sigma_{e} 2\left(y_{e}-o_{e}\right) \partial\left(y_{e}-o_{e}\right)/\partial w_{i} \\
=\Sigma_{e}\left(y_{e}-o_{e}\right) \partial\left(y_{e}-\sigma(s)\right) / \partial w_{i} \\
=\Sigma_{e}\left(y_{e}-o_{e}\right) \sigma^{\prime}(s)\left(-x_{i e}\right)
\end{array}
$$
where $x_{i e}$ denotes the $i$-th component of the example

- The Gradient descent training rule for training sigmoidal Perceptrons is:
$$
w_{i}=w_{i}+\eta \sum_{e}\left(y_{e}-o_{e}\right) \sigma^{\prime}(S) x_{i e}
$$
where:
$$
\sigma^{\prime}(S)=\sigma(S)(1-\sigma(S))
$$

#### Gradient Descent Learning Algorithm for Sigmoidal Perceptrons

- Initialization: Examples $\left\{\left(x_{e}, y_{e}\right)\right\}_{e=1}^{N}$, initial weights $w_{i}$ set to small random values, learning rate parameter $\eta$
- **Repeat**
  for each training example $\left(x_{e}, y_{e}\right)$
  - calculate the network output: $o=\sigma(s)$ where $s=\sum_{i=0}^{d} w_{i} x_{i e}$
  - if the Perceptron does not respond correctly compute weight corrections:
$$
\Delta w_{i}=\Delta w_{i}+\eta\left(y_{e}-o_{e}\right) \sigma(s)(1-\sigma(s)) x_{i e}
$$
​		update the weights with the <span style="color:red">**accumulated error**</span> from all examples $w_{i}=w_{i}+\Delta w_{i}$
​        **until** termination condition is satisfied.

##### Example

- Suppose an example of Perceptron which accepts two inputs $x_{1}$ and $x_{2}$ with weights $w_{1}=0.5$ and $w_{2}=0.3$ and $w_{0}=-1$ learning rate $=1$
- Let the following example is given: $x_{1}=2, x_{2}=1, y=0$ The output of the Perceptron is :
$$
o=\sigma(-1+2 * 0.5+1 * 0.3)=\sigma(0.3)=0.5744
$$
- The weight updates according to the gradient descent algorithm will be:

$$
\begin{array}{l}
\Delta w_{0}=(0-0.5744) * 0.5744 *(1-0.5744) * 1=-0.1404 \\
\Delta w_{1}=(0-0.5744) * 0.5744 *(1-0.5744) * 2=-0.2808 \\
\Delta w_{2}=(0-0.5744) * 0.5744 *(1-0.5744) * 1=-0.1404
\end{array}
$$

##### Example

Let another example is given: $x_{1}=1, x_{2}=2, y=1$ The output of the Perceptron is :
$$
o=\sigma(-1+1 * 0.5+2 * 0.3)=\sigma(0.1)=0.525
$$
The weight updates according to the gradient descent algorithm will be:
$$
\begin{array}{l}
\Delta W_{0}=-0.1404+(1-0.525) * 0.525 *(1-0.525) * 1=-0.0219 \\
\Delta W_{1}=-0.2808+(1-0.525) * 0.525 *(1-0.525) * 1=-0.1623 \\
\Delta W_{2}=-0.1404+(1-0.525) * 0.525 *(1-0.525) * 2=0.0966
\end{array}
$$
If there are no more examples in the batch, the weights will be modified as follows:
$$
\begin{array}{l}
w_{0}=-1+(-0.0219)=-1.0219 \\
w_{1}=0.5+(-0.1623)=0.3966 \\
w_{2}=0.3+0.0966=0.3966
\end{array}
$$

#### Incremental Gradient Descent Learning Algorithm for Sigmoidal Perceptrons

**Initialization:**
Examples $\left\{\left(x_{e} ,y_{e}\right)\right\}$, initial weights $w_{i}$ set to small random values, learning rate parameter $\eta$
**Repeat**
for each training example $\left(x_{e} ,y_{e}\right)$

- calculate the network output: $o=\sigma(s)$
  where $s=\sum_{i=0}^{d} w_{i} x_{i e}$

  If the Perceptron does not respond correctly update the weights:  $\quad w_{i}=w_{i}+\eta\left(y_{e}-o_{e}\right) \sigma(s)(1-\sigma(s)) x_{i e}$

**until** termination condition is satisfied.

### Perceptron vs. Gradient Descent

- Gradient descent finds the decision boundary which minimizes the <span style="color:blue">**sum squared error**</span> of the (target - net) value rather than the (target - output) value
  - Perceptron rule will find the decision boundary which minimizes the classification error<span style="color:red"> **$-$ if the problem is linearly separable**</span>
  - Gradient descent decision boundary may leave more instances misclassified as compared to the perceptron rule: <span style="color:blue">**could have a higher misclassification rate than with the perceptron rule**</span>
- Perceptron rule (target - thresholded output) guaranteed to converge to a separating hyperplane if the problem is linearly separable.

#### The error surface

- The error surface lies in a space with a horizontal axis for each weight and one vertical axis for the error.
  - For a linear neuron, it is a quadratic bowl.
  - Vertical cross-sections are parabolas.
  - Horizontal cross-sections are ellipses.

  <img src="images\image-20210928171446376.png" alt="image-20210928171446376" style="zoom:80%;" />

#### Batch vs incremental learning

<img src="images\image-20210928171512831.png" alt="image-20210928171512831" style="zoom: 80%;" />

### Summary

- Perceptron training:
  - uses thresholded unit
  - converges after a finite number of iterations
  - output hypothesis classifies training data perfectly
  - linearly separability necessary
- Gradient descent
  - uses unthresholded linear unit
  - converges asymptotically toward a minimum error hypothesis
  - termination is not guaranteed
  - linear separability not necessary

### The fall of the Perceptron

- Researchers begun to discover the Perceptron's limitations.
- Unless input categories were "linearly separable", a perceptron could not learn to discriminate between them.
- Unfortunately, it appeared that many important categories were not linearly separable.
- E.g., those inputs to an XOR gate that give an output of 1 (namely 10 \& 01) are not linearly separable from those that do not $(00 \& 11)$.
