# Bio-Computation

## Lecture 0

### Content

>[Biological Neural Networks Overview](#Biological Neural Networks Overview): [Abstract neuron](#Abstract neuron)

**Important test time:**

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

<img src="images\image-20210910113737442.png" alt="image-20210910113737442" style="zoom:80%;" />

**Input unit** 

<img src="images\image-20210910113809177.png" alt="image-20210910113809177" style="zoom:80%;" />

**Input unit**

<img src="images\image-20210910113859520.png" alt="image-20210910113859520" style="zoom:80%;" />

**Input unit**

<img src="images\image-20210910114049883.png" alt="image-20210910114049883" style="zoom:80%;" />

**Input unit**

<img src="images\image-20210910114213034.png" alt="image-20210910114213034" style="zoom:80%;" />

**Input unit** 

<img src="images\image-20210910114240377.png" alt="image-20210910114240377" style="zoom:80%;" />

**Input unit** 

<img src="images\image-20210910114302602.png" alt="image-20210910114302602" style="zoom:80%;" />

**Input unit** 

<img src="images\image-20210910114331930.png" alt="image-20210910114331930" style="zoom:80%;" />

**Input unit** 

<img src="images\image-20210910133151581.png" alt="image-20210910133151581" style="zoom:80%;" />

**Input unit** 

<img src="images\image-20210910133235332.png" alt="image-20210910133235332" style="zoom:80%;" />

**Input unit** 

<img src="images\image-20210910133306227.png" alt="image-20210910133306227" style="zoom:80%;" />

ss
