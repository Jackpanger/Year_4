# Bio-Computation

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
