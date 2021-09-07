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

> TBD

#### Abstract model of a neuron

An abstract neuron $j$ with $n$ inputs: 

+ Each input $i$ transmits a real value $a_j $
+ Each connection is assigned with the weight $w_{ji}$<span style="color:red">($w_{ij}$ on PPT)</span>

The total input $S$, i.e., the sum of the products of the inputs with the corresponding weights, is compared with the threshold (equal to 0 in this case), and the outcome $X_j$<span style="color:red">($X_i$ on PPT) </span>is produced consequently

<img src="images\image-20210907103705762.png" alt="image-20210907103705762" style="zoom: 80%;" />
$$
S = \sum_i^n w_{ji}x_i
$$

#### The McCulloch-Pitts Neuron (1943)

The authors modelled the neuron as 

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