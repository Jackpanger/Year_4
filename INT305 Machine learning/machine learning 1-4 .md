# Machine Learning 1-4

1. *Coursework 1 : 15%* 

   *○ The coursework requires no lab practice.* 

2. *Coursework 2 : 15%*

   *○ This coursework requires lab practice.*

3. *Final Exam : 70%* 

   *○ Final Exam is the most important part for assessment. It will be a open  book exam.* 

## Lecture 1 

### **Contents**

> [Nearest Neighbor](#Nearest Neighbor) (all)

### **What is machine learning**

<img src="images\image-20210906165229750.png" alt="image-20210906165229750"  />

#### Relations to AI

##### Relations to Human Learning

#### Types of machine learning 

+ **Supervised learning**: have labeled examples of the correct behavior
+ **Reinforcement learning**: learning system (agent) interacts with the world and learns to maximize a scalar reward signal
+ **Unsupervised learning**: no labeled example - instead, looking for "interesting" patterns in the data

#### **History of machine learning**

#### Machine learning in computer vision

> Object detection
>
> Semantic segmentation
>
> Pose estimation

#### Machine learning in speech processing

>Speech to text
>
>Personal assistants
>
>Speaker identifications

#### Machine learning in NLP

>Machine translation
>
>Sentiment analysis
>
>Topic modeling
>
>Spam filtering

#### Machine learning in game play

>AlphaGo

#### Machine learning in E-commserce

#### ML workflow

##### **ML workflow sketch** 

1. Should I use ML on this problem? 
   + *Is there a pattern to detect?* 
   + *Can I solve it analytically?* 
   + *Do I have data?* 
2. Gather and organize data. 

+ *Preprocessing, cleaning, visualizing.*

3. Establishing a baseline. 
4. Choosing a model, loss, regularization,… 
5. Optimization (could be simple, could be a Phd…)， 
6. Hyperparameter search. 
7. Analyze performance & mistakes, and iterate back to step 4 (or 2).

#### Implementing machine learning systems

> Array processing (NumPy)
>
> Neural net frameworks: PyTorch, TensorFlow, JAX, etc. 
>
> + automatic differentiation 
> + compiling computation graphs 
> + libraries of algorithms and network primitives 
> + support for graphics processing units (GPUs)

### <span id="anchor1" style="color:rgb(200,10,0)">Nearest Neighbor</span>

##### **Preliminaries and Nearest Neighbor Methods**

> supervised learning: This means we are given a training set consisting of inputs and corresponding labels. eg.
>
> |          Task           |     Inputs     |      Labels       |
> | :---------------------: | :------------: | :---------------: |
> |   object recognition    |     image      |  object category  |
> |    image captioning     |     image      |      caption      |
> | document classification |      text      | document category |
> |     speech-to-text      | audio waveform |       text        |

##### **Input Vector**

Common strategy: represent the input as an input vector in $R^d$

+ **Representation** = mapping to another space that's easy to manipulate 
+ Vectors are a great representation since we can do linear algebra!

<img src="images\image-20210906173139003.png" alt="image-20210906173139003" />

Mathematically, training set consists of a collection of pairs of an input vector $x\in R^d$ and its corresponding **target**, or **label**, $t$ 

+ **Regression**: $t$ is a real number (e.g. stock price) 
+ **Classification**: $t$ is an element of a discrete set $\{1,…,C\} $
+ These days,  $t$ is often a highly structured object (e,g. image) 

 Denote the training set $\{(x^{(1)},t^{(1))},...,(X^{(N)},T^{(N)})\}$

+ Note: these superscripts have nothing to do with exponentiation!

#### Nearest Neighbors

+ Suppose we’re given a novel input vector $x$ we’d like to classify. 
+ The idea: find the nearest input vector to $x$ in the training set and copy 
  its label. 
+ Can formalize "nearest" in terms of Euclidean distance

$$
||x^{(a)}-x^{(b)}||_2 = \sqrt{\sum_{j=1}^d(x_j^{(a)}-x_j^{(b)})^2}
$$

> **Algorithm:**
>
> 1. Find example $(x^*, t^*) $ (from the stored training set) closest $ x$. That is:
>    $$
>    x^* = \underset {x^{(i)}\in{train\,set}}{\operatorname {arg\,min} }\,distance(x^{(i)},x)
>    $$
>
> 2. Output $y = t^*$

##### Decision Boundaries

We can visualize the behavior in the classification setting using a **Voronoi diagram**.

<img src="images\\image-20210906194744348.png" alt="image-20210906194744348" />

<img src="images\image-20210906194912474.png" alt="image-20210906194912474" />

+ Nearest neighbors **sensitive to noise or mis-labeled data** (“class noise”)
+ Smooth by having k nearest neighbors vote

### k-Nearest Neighbors

<img src="images\image-20210906195111261.png" alt="image-20210906195111261" />

>**Algorithm:**
>
>1. Find $k$ example $(x^{(i)}, t^{(i)}) $ (from the stored training set) closest to the test instance $x$. 
>
>2. Classification output is majority class
>
>$$
>y = \underset {t^{(z)}}{\operatorname {arg\,max} }\,\sum_{i=1}^k Ⅱ(t^{(z)}=t^{(i)})
>$$
>
>*$Ⅱ${statement} is the identity function and is equal to one whenever the statement is true. We could also write this as $\delta(t^{(z)},t^{(i)})$, with $\delta(a,b)=1$ if $a=6,0$ otherwise.* 

**$k=1$**

<img src="images\image-20210906200944380.png" alt="image-20210906200944380" style="zoom: 50%;" />

**$k=15$**

<img src="images\image-20210906201006702.png" alt="image-20210906201006702" style="zoom: 50%;" />

##### **Tradeoffs in choosing $k$?** 

> **Small $k$** 
>
> + Good at capturing fine-grained patterns
> + May **overfit**, i.e. be sensitive to random idiosyncrasies in the training data 
>
> **Large $k $**
>
> + Makes stable predictions by averaging over lots of examples 
> + May **underfit**, i.e. fail to capture important regularities 
>
> **Balancing $ k $**
>
> + Optimal choice of k depends on number of data points $n$. 
>
> + Nice theoretic al properties if $k\to\infty$ and $\frac{k}{n}\to0 $
>
>   <Span style="color:rgb(130,50,150)">***PN: which means make k as much as large and $\frac{k}{n}$ still tends to 0*  **</span>
>
> + Rule of thumb: choose $k < \sqrt{n}$
>
> + We can choose k using validation set (next slides)・

+ We would like our algorithm to **generalize** to data it hasn't seen before. 

+ We can measure the generalization error (error rate on new examples) 
  using a **test set**.

  <Span style="color:rgb(130,50,150)">***PN: choose the k at the point of lowest error rate in test set* **</span>

<img src="images\image-20210906202105173.png" alt="image-20210906202105173" style="zoom: 200%;" />

+ $k$ is an example of a **hyperparameter**, something we can’t, fit as part of the learning algorithm itself 
+ We can tune hyperparameters using a **validation set**:

<img src="images\image-20210906202227254.png" alt="image-20210906202227254" style="zoom:80%;" />

+ The test set is used only at the very end, to measure the generalization performance of the final configuration.

#### Pitfalls: The Curse of Dimensionality

+ Low-dimensional visualizations are misleading! In high dimensions, "most" points are far apart.
+ If we want the nearest neighbor to be closer than $\epsilon$. how many points do we need to guarantee it? 
+ The volume of a single ball of radius $\epsilon$ is $O(\epsilon^d) $
+ The total volume of $[0,1]^d$ d is 1.

Therefore $O((\frac{1}{\epsilon})^d)$ balls are needed to cover the volume.

<Span style="color:rgb(130,50,150)">***PN: which means* **</span>

1. <Span style="color:rgb(130,50,150)">***Assuming demension is d, there needs $O((\frac{1}{\epsilon})^d)$ balls to cover the total volume(given it is the unit 1). $\epsilon<1$, the number is increasing when d goes up.* **</span>
2. <Span style="color:rgb(130,50,150)">***as d goes up, the ratio of the volume of a ball to the total volumn goes down. To obtain a same ratio, $\epsilon$ should be larger to get more point.* **</span>
3. <Span style="color:rgb(130,50,150)">***Hence, we can see in the right image, when d goes to 10, in same fraction of Volumn, d=10 got the highest distance. Namely, in high dimensions, most points are far apart.* **</span>

<img src="images\image-20210906202731235.png" alt="image-20210906202731235"  />

+ In high dimensions. "most" points are approximately the same distance. 
+ We can show this by applying the rules of expectation and covariance of random variables in surprising ways. 
+ Picture to keep in mind:

<img src="images\image-20210906202903453.png" alt="image-20210906202903453"  />

+ Saving grace: some datasets (e.g. images) may have low **intrinsic dimension**, i.e. lie on or near a low-dimensional manifold.

  <Span style="color:rgb(130,50,150)">***PN: which means the number of features required to approximate it.* **</span>

<img src="images\image-20210906203006640.png" alt="image-20210906203006640"  />

+ The neighborhood structure (and hence the Curse of Dimensionality) depends on the intrinsic dimension. 
+ The space of megapixel images is 3 million-dimensional. The true number of degrees of freedom is much smaller.

<img src="images\image-20210906203051252.png" alt="image-20210906203051252" style="zoom:150%;" />

+ Nearest neighbors can be sensitive to the ranges of different features. 
+ Often, the units are arbitrary:

<img src="images\image-20210906203126229.png" alt="image-20210906203126229"  />

+ Simple fix: **normalize** each dimension to be zero mean and unit variance. I.e., compute the mean $\mu_j$ and standard deviation $\sigma_j$)，and take
  $$
  \tilde{a}_j = \frac{x_j-\mu_j}{\sigma_j}
  $$

+ Caution: depending on the problem, the scale might be important!

#### Pitfalls: Computational Cost

+  Number of computations at **training time**: 0 
+  Number of computations at **test time**，per query (naïve algorithm) 
   + Calculate D-dimensional Euclidean distances with $N$ data points: $O(ND)$ 
   + Sort the distances: $O(N logN)$ 
+  This must be done for each query, which is very expensive by the standards of a learning algorithm! 
+  Need to store the entire dataset in niemorv! 
+  Tons of work lias gone into algorithms and data structures for efficient nearest neighbors with high dimensions and/or large datasets.

##### Example: Digit Classification

+ KNN can perform a lot better with a good similarity measure. 
+ Example: shape contexts for object recognition. In order to achieve invariance to image transformations, they tried to warp one image to match the other image. 
  + Distance measure: average distance between corresponding points on warped images 
+ Achieved $0.63\%$ error on MNIST, compared with $3\%$ for Euclidean KNN. 
+ Competitive with conv nets at the time, but required careful engineering.

<img src="images\image-20210906204157608.png" alt="image-20210906204157608"  />

### Conclusion

+ Simple algorithm that does all its work at test time $--$ in a sense, no learning! 
+ Can control the complexity by varying $k $
+ Suffers from the Curse of Dimensionality 
+ Next time: parametric models, which learn a compact summary of the data rather than referring back to it at test time.

## Lecture 2

**Linear Methods for Regression, Optimization**

### Contents

>[Supervised Learning Setup](#Supervised Learning Setup)
>
>[Linear Regression](#Linear Regression - Model)
>
>+ [Loss Function](#Linear Regression - Loss Function)
>+ [Vectorization](#Vectorization)
>
>[Solving the Minimization Problem](#Solving the Minimization Problem)
>
>+ [Direct Solution I: Linear Algebra](#Direct Solution I: Linear Algebra)
>+ [Direct Solution II: Calculus](#Direct Solution II: Calculus)
>
>[Feature Mapping](#Feature Mapping (Basis Expansion))
>
>+ [Polynomial Feature Mapping](#Polynomial Feature Mapping)
>
>[Regularization](#Regularization)
>
>+ ##### [L2 Regularization](#L2 Regularization)
>
>+ [L2 Regularized Least Squares: Ridge regression](#L2 Regularized Least Squares: Ridge regression)
>
>[Gradient Descent]()
>
>+ [Gradient Descent for Linear Regression](#Gradient Descent for Linear Regression)
>+ [L2 Regularized Least Squares: Ridge regression](#L2 Regularized Least Squares: Ridge regression)
>+ [Learning Rate (Step Size)](#Training Curves)
>+ [Training Curves](#Training Curves)
>
>[Stochastic Gradient Descent](#Stochastic Gradient Descent)
>
>+ [SGD Learning Rate](#SGD Learning Rate)

### **Overview**

+ Second learning algorithm of the course: <span style="color:blue">linear regression</span>. 

  + <span style="color:blue">Task</span>: predict scalar-valued targets (e.g. stock prices) 

    <Span style="color:rgb(130,50,150)">***PN: A scalar valued function is  a function that takes one or more values but returns a single value.***  </span>

  + <span style="color:blue">Architecture</span>: linear function of the inputs  

+ While KNN was a complete algorithm, linear regression exemplifies a modular approach that will be used throughout this course: 

  + choose a <span style="color:blue">model</span> describing the relationships between variables of interest 
  + define a <span style="color:blue">loss function</span> quantifying how bad the fit to the data is 
  + choose a <span style="color:blue">regularizer</span> saying how much we prefer different candidate models (or explanations of data) 
  + fit a model that minimizes the loss function and satisfies the constraint/penalty imposed by the regularizer, possibly using an <span style="color:blue">optimization algorithm </span>

+ Mixing and iiiatcliing these modular components give us a lot of new ML methods.

### Supervised Learning Setup

<img src="images\image-20210913164046006.png" alt="image-20210913164046006"  />

In supervised learning:   (data driven approach)

+ There is input $x\in \mathcal{X}$ typically a vector of features (or covariates) 
+ There is target $t\in \mathcal{T}$ (also called response, outcome, output, class) 
+ Objective is to learn a function $f : \mathcal X \to \mathcal{T}$ such that $t\approx y = f(x)$ based on some data $\mathcal{D} = \{(x^{(i)}，t^{(i)}) \text{ for } i = 1, 2,...,N\}.$

### Linear Regression - Model

+ <span style="color:blue">Model:</span> In linear regression, we use a linear function of the features $x = (x_1,...,x_D)\in \Bbb R^D$ to make predictions $y$ of the target value $t \in \Bbb R$:
  $$
  y = f(x) = \sum_jw_jx_j+b
  $$

  + $y$ is the <span style="color:blue">prediction  </span>
  + $w $ is the <span style="color:blue">weights  </span>
  + $b$ is the <span style="color:blue">bias </span> (or <span style="color:blue">intercept </span>)

+ $w$ and $b$ together are the <span style="color:blue">parameters</span>

+ We hope that our prediction is close to the target: $ y\approx t$.

#### What is Linear? 1 feature vs D features

<img src="images\image-20210913164751281.png" alt="image-20210913164751281"  />

+ If we have only 1 feature:  $y = wx + b$ where $w, x, b \in \Bbb R. $
+ $y$ is linear in $x$.

<img src="images\image-20210913164834154.png" alt="image-20210913164834154"  />

+ If we have $D$ features: $ y = w^Tx +b $ where $w,x\in \Bbb R^D, b\in \Bbb R$

  <Span style="color:rgb(130,50,150)">***PN:***</span>

  1. <Span style="color:rgb(130,50,150)">***$w^T:$ T means transpose. ps: vector is vertical by default***</span>
     $$
     \begin{align*}
     w =\begin{pmatrix}
     w_1\\
     w_2\\
     \vdots\\
     w_D
     \end{pmatrix}
     \quad
     w^T=
     \begin{pmatrix}
     w_1,
     w_2,
     \cdots,
     w_D
     \end{pmatrix}
     \end{align*}
     $$

  2. <Span style="color:rgb(130,50,150)">***$w^Tx$ is inner product***</span>

+ $y$ is linear in $x$.

Relation between the prediction y and inputs x is linear in both cases.

#### Linear Regression

We have a dataset $\mathcal{D} = \{(x^{(i)}, t^{(i)})$  for  $ i = 1,2,..., N\} $ where, 

+ $x^{(i)} = (x_1^{(i)}, x_2^{(i)},..., x_D^{(i)})^T \in \Bbb{R}^D $ are the inputs (e.g. age, height) 

+ $t^{(i)} \in \Bbb{R}$ is the target or response (e.g. income) 

+ predict $t^{(i)}$ with a linear function of $x^{(i)}$:

  <img src="images\image-20210913182206188.png" alt="image-20210913182206188" style="zoom:80%;" />

+ $t^{(i)}\approx y_{(i)}=w^Tx^{(i)}+b$

+ Different $(w,b)$ define different lines.

+ We want the "best" line $(w,b)$.

+ How to quantify "best"?

##### Linear Regression - Loss Function

+ A <span style="color:blue">loss function</span> $\mathcal{L}(y,t)$ defines how bad it is if，for some example $x$, the algorithm predicts $y$, but the target is actually $t$.

+ <span style="color:blue">Squared error loss function</span>:
  $$
  \mathcal L(y,t) = \frac{1}{2}(y-t)^2
  $$

+ $y-t$ is the <span style="color:blue">residual</span>, and we want to make this small in magnitude 

+ The $\frac{1}{2}$ factor is just to make the calculations convenient. 

+ <span style="color:blue">Cost function</span>: loss function averaged over all training examples
  $$
  \begin{align*}
  \mathcal J(w,b) &= \frac{1}{2N}\sum_{t=1}^N(y^{(i)}-t^{(t)})^2 \\
  &=\frac{1}{2N}\sum_{i=1}^N(w^Tx^{(i)}+b-t^{(i)})^2
  \end{align*}
  $$

+ Terminology varies. Some call "cost" *empirical* or *average* loss.

##### Vectorization

+ Notation-wise, $\frac{1}{2N}\sum^N_{i=1}(y^{(i)}-t^{(i)})^2$ gets messy if we expand $y^{(i)}$
  $$
  \frac{1}{2N}\sum_{i=1}^N(\sum_{j=1}^D(w_jx_j^{(i)}+b)-t^{(i)})^2
  $$

+ The code equivalent is to compute the prediction using a for loop:

  ```python
  y = b
  for j in range(M):
      y+=w[j]*x[j]
  ```

+ Excessive super/sub scripts are hard to work with, and Python loops are slow, so we <span style="color:blue">vectorize</span> algorithms by expressing them in terms of vectors and matrices.
  $$
  w=(w_1,...w_D)^T \text{ \,\,\,} w = (x_1,...,x_D)^T\\
  y = w^Tx+b
  $$

+ This is simpler and executes much faster

  ```python
  y = np.dot(w,x)+b
  ```

Why vectorize?

+ The equations, and the code, will be simpler and more readable. 
  Gets riel of dummy variables$/$indices! 

+ Vectorized code is much faster 

  + Cut down on Python interpreter overhead 
  + Use highly optimized linear algebra libraries (hardware support)
  + Matrix multiplication very fast on GPU (Graphics Processing Unit) 

+ Switching in and out of vectorized form is a skill you gain with practice 

  + Some derivations are easier to do element-wise 
  + Some algorithms are easier to write$ /$understand using for-loops 
    and vectorize later for performance

+ We can organize all the training examples into a <span style="color:blue">design matrix</span> $X$ with one row per training example, and all the targets into the <span style="color:blue">target vector</span> $t$.

  <img src="images\image-20210913165946384.png" alt="image-20210913165946384" style="zoom:80%;" />

+ Computing the predictions for the whole dataset:
  $$
  Xw+b1 = 
  \begin{pmatrix}
  w^Tx^{(1)}+b\\
  \vdots\\
  w^Tx^{(N)}+b
  \end{pmatrix}
  =\begin{pmatrix}
  y^{(1)}\\
  \vdots\\
  y^{(N)}
  \end{pmatrix}
  =y
  $$
  <Span style="color:rgb(130,50,150)">***PN: b1 might mean b with all paramaters initialized as 1***</span>

+ Computing the squared error cost across the whole dataset:
  $$
  y = Xw +b1\\
  \mathcal J=\frac{1}{2N}||y-t||^2
  $$

+ Sometimes we may use $\mathcal J = \frac{1}{2}||y — t||^2$, without a normalizer. This would correspond to the sum of losses, and not the averaged loss. The minimizer does not depend on $N$ (but optimization might!). 

+ We can also add a column of 1’s to design matrix, combine the bias and the weights, and conveniently write
  $$
  X = 
  \begin{bmatrix}
  1 & [x^{(1)}]^T\\
  1 & [x^{(2)}]^T\\
  1 & \vdots
  \end{bmatrix}
  \in \Bbb R^{N \times (D+1)} \qquad
  w = 
  \begin{bmatrix}
  b\\
  w_1\\
  w_2\\
  \vdots
  \end{bmatrix}
  \in \Bbb R^{D+1}
  $$
  Then, our predictions reduce to $y = Xw$

#### Solving the Minimization Problem

We defined a cost function. This is what we’d like to minimize.

Two commonly applied mathematical approaches:

+ Algebraic, e.g., using inequalities:
  + to show $z^*$ minimizes $f(z)$, show that $\forall z, f(z) \geq f(z^*)$
  + to show that $a=b$, show that $b\geq a$ and $b\geq a$
+ Calculus: minimum of a smooth function (if it exists) occurs at a <span style="color:blue">critical point</span>, i.e. point where the derivative is zero. 
  + multivariate generalization: set the partial derivatives to zero (or equivalently the gradient).

Solutions may be direct or iterative 

+ Sometimes we can directly find provably optimal parameters (e.g. set the gradient to zero and solve in closed form). We call this a <span style="color:blue">direct solution</span>. 
+ We may also use optimization techniques that iteratively get us closer to the solution. We will get back to this soon.

##### Direct Solution I: Linear Algebra

+ We seek $w$ to minimize $||Xw — t||^2$, or equivalently $||Xw — t|| $

+ range$(X) = \{Xw | w \in \Bbb R^D\}$ is a $D$-dimensional subspace of $\Bbb R^N$. 

+ Recall that the closest point $y^* = Xw^*$ in subspace range$(X)$ of $\Bbb R^N$ to arbitrary point $t\in \Bbb R^N$ is found by orthogonal projection.

  <Span style="color:rgb(130,50,150)">***PN:***</span>

  1. <Span style="color:rgb(130,50,150)">***For $w\in \Bbb R^D$, Xw constitute a subspace, like a plane.***</span>
  2. <Span style="color:rgb(130,50,150)">***Find the shortest path from t to the plane is by orthogonal projection***</span>
  3. <Span style="color:rgb(130,50,150)">***Hence, vector <t,y'> is orthogonal to the subspace***</span>

<img src="images\image-20210913170948703.png" alt="image-20210913170948703"  />

​	We have $(y^*-t)丄 Xw, \forall w \in \Bbb R^D$

+ Why is $y^*$ the closest point to $t$?

  + Consider any $z = Xw$
  + By Pythagorean theorem and the trivial inequality $x^2\geq 0$:

  $$
  \begin{align*}
  ||z-t||^2 &= ||y^*-t||^2 +||y^*-z||^2\\
  &\geq ||y^*-t||^2
  \end{align*}
  $$

+ From the previous slide, we have $(y* - t)丄 Xw, \forall w\in \Bbb R^D$

+ Equivalently, the columns of the design matrix X are all orthogonal to $(y^* - t)$, and we have that:
  $$
  \begin{align*}
  X^T(y^*-t) &= 0\\ 
  X^TXw^* - X^Tt &= 0\\ 
  X^TXw^* &= X^Tt \\
  w^* &= (X^TX)^{-1}X^Tt\\
  \end{align*}
  $$

+ While this solution is clean and the derivation easy to remember, like many algebraic solutions, it is somewhat ad hoc. 

+ On the hand, the tools of calculus are broadly applicable to differentiable loss functions...

##### Direct Solution II: Calculus

+ <span style="color:blue">Partial derivative</span>: derivative of a multivariate function with respect to one of its arguments.
  $$
  \frac{\partial}{\partial x_1} f(x_1,x_2) = \lim_{h\to 0}\frac{f(x_1+h,x_2)-f(x_1,x_2)}{h}
  $$

+ To compute, take the single variable derivative, pretending the other arguments are constant. 

+ Example: partial derivatives of the prediction $y$

  + $$
    \begin{align*}
    \frac{\partial y}{\partial w_j} &= \frac{\partial}{\partial w_j}
    \begin{bmatrix}
    \sum_{j^{'}}w_{j^{'}}x_{j^{'}} + b
    \end{bmatrix}\\
    &=x_j\\
    \end{align*}
    $$

  + $$
    \begin{align*}
    \frac{\partial y}{\partial b} &= \frac{\partial}{\partial b}
    \begin{bmatrix}
    \sum_{j^{'}}w_{j^{'}}x_{j^{'}} + b
    \end{bmatrix}\\
    &=1
    \end{align*}
    $$

+ For loss derivatives, apply the <span style="color:blue">chain rule</span>:
  $$
  \begin{align*}
  \frac{\partial \mathcal L}{\partial w_j} &= \frac{\mathrm{d}\mathcal L}{\mathrm{d}y}\frac{\partial y}{\partial w_j}\\
  &=\frac{\mathrm{d}}{\mathrm{d} y}[\frac{1}{2}(y-t)^2] \cdot x_j\\
  &= (y-t)x_j
  \end{align*}
  $$

  $$
  \begin{align*}
  \frac{\partial \mathcal L}{\partial b} &= \frac{\mathrm{d}\mathcal L}{\mathrm d y}\frac{\partial y}{\partial b}\\
  &= y-t
  \end{align*}
  $$

+ For cost derivatives, use <span style="color:blue">linearity </span> and average over data points:
  $$
  \frac{\partial \mathcal J}{\partial w_j} = \frac{1}{N}\sum_{i=1}^N(y^{(i)}-t^{(i)})x_j^{(i)} \qquad
  \frac{\partial \mathcal J}{\partial b} = \frac{1}{N}\sum_{i=1}^Ny^{(i)}-t^{(i)}
  $$

+ Minimum must occur at a point where partial derivatives are zero.
  $$
  \frac{\partial \mathcal J}{\partial w_j} = 0 \,(\forall j), \qquad
  \frac{\partial \mathcal J}{\partial b} = 0.
  $$
  (if $\partial \mathcal J/\partial w_j \neq 0$, you could reduce the cost by changing $w_j$)

+ The derivation on the previous slide gives a system of linear equations, which we can solve efficiently.

+ As is often the case for models and code, however, the solution is easier to characterize if we vectorize our calculus. 

+ We call the vector of partial derivatives the <span style="color:blue">gradient</span>

+ Thus, the "gradient of $f:\Bbb R^D \to \Bbb R$"，denoted $\nabla f(w)$, is:
  $$
  (\frac{\partial}{\partial w_1}f(w),...,\frac{\partial}{\partial w_D}f(w))
  $$

+ The gradient points in the direction of the greatest rate of increase. 

+ Analogue of second derivative (the "Hessian" matrix):

  $\nabla^2f(w)\in \Bbb R^{D\times D}$ is a matrix with $[\nabla^2f(w)]_{ij}=\frac{\partial ^2}{\partial w_i \partial w_j}f(w)$.

+ We seek $w$ to minimize $\mathcal J(w) = \frac{1}{2}| |Xw — t||^2$ 

+ Taking the gradient with respect to **w (see course notes for additional details)** we get:
  $$
  \begin{align*}
  \nabla_w\mathcal J(w) = X^TXw-X^Tt=0
  \end{align*}
  $$

+ We get the same optimal weights as before:
  $$
  \begin{align*}
  w^*=(X^TX)^{-1}X^Tt
  \end{align*}
  $$

+ Linear regression is one of only a handful of models in this course that permit direct solution.

#### Feature Mapping (Basis Expansion)

The relation between the input and output may not be linear.

<img src="images\image-20210913203552639.png" alt="image-20210913203552639"  />

+ We can still use linear regression by mapping the input features to another space using <span style="color:blue"> feature mapping</span> (or <span style="color:blue">basis expansion</span>).

  $\mathcal \psi(x): \Bbb R^D\to \Bbb R^d$ and treat the mapped feature (in $\Bbb R^d$) as the input of a linear regression procedure.

+ Let us see how it works when $x\in \Bbb R$ and we use a polynomial feature mapping.

##### Polynomial Feature Mapping

If the relationship doesn’t look linear, we can fit a polynomial.

<img src="images\image-20210913173949436.png" alt="image-20210913173949436"  />

Fit the data using a degree-$M$ polynomial function of the form:
$$
\begin{align*}
y=w_0+w_1x+w_2x^2+...+w_Mx^M=\sum_{i=0}^Mx_ix^i
\end{align*}
$$

+ Here the feature mapping is $\mathcal \psi(x)=[1,x,x^2,...,x^M]^T$.

+ We can still use linear regression to find $w$ since $y = \mathcal \psi(x)^Tw$ is linear in $w_0, w_1,....$

+ In  general, $\mathcal \psi$  can be any  function.  Another example:

  $\mathcal \psi(x) = [1, sin(2\pi x), cos(2\pi x), sin(4\pi x), ...]T$.

**Polynomial Feature Mapping with M = 0**

<img src="images\image-20210913174053628.png" alt="image-20210913174053628" style="zoom:80%;" />

**Polynomial Feature Mapping with M = 1**

<img src="images\image-20210913174120312.png" alt="image-20210913174120312" style="zoom:80%;" />

**Polynomial Feature Mapping with M = 3**

<img src="images\image-20210913174144045.png" alt="image-20210913174144045" style="zoom:80%;" />

**Polynomial Feature Mapping with M = 9**

<img src="images\image-20210913174205511.png" alt="image-20210913174205511" style="zoom:80%;" />

#### Model Complexity and Generalization

<span style="color:blue">Underfitting</span> (M=0): model is too simple $——$ does not fit the data. 

<span style="color:blue">Underfitting</span> (M=9): model is too complex $——$ fits perfectly.

<img src="images\image-20210913204657666.png" alt="image-20210913204657666"  />

<span style="color:blue">Good model</span> (M=3): Achieves small test error (generalizes well).

<img src="images\image-20210913204715844.png" alt="image-20210913204715844"  />

<img src="images\image-20210913204748302.png" alt="image-20210913204748302" style="zoom:80%;" />

+ As $M$ increases, the magnitude of coefficients gets larger.
+ For $M  = 9$，the coefficients have become finely tuned to the data.
+ Between data points, the function exhibits large oscillations.

#### Regularization

+ The degree $M$ of the polynomial controls the model’s complexity. 
+ The value of $M$ is a hyperparameter for polynomial expansion, 
  just like $k$ in KNN. We can tune it using a validation set. 
+ Restricting the number of parameters $/$ basis functions $(M)$ is a crude approach to controlling the model complexity. 
+ Another approach: keep the model large, but <span style="color:blue">regularize</span> it 
  + <span style="color:blue">Regularizer</span>: a function that quantifies how much we prefer one hypothesis vs. another

##### L2 Regularization

+ We can encourage the weights to be small by choosing as our regularizer the <span style="color:blue">$L^2$ penalty</span>.
  $$
  \begin{align*}
  \mathcal R(w) = \frac{1}{2}||w||^2_2=\frac{1}{2}\sum_jw_j^2.
  \end{align*}
  $$

  + Note: To be precise, the $L^2$ norm is Euclidean distance, so we’re 
    regularizing the squared $L^2$ norm.

+ The regularized cost function makes a tradeoff between fit to the data and the norm of the weights.
  $$
  \begin{align*}
  \mathcal J_{reg}(w)=\mathcal J(w)+\lambda\mathcal R(w)=\mathcal J(w)+\frac{\lambda}{2}\sum_jw_j^2
  \end{align*}
  $$

+ If you fit training data poorly，$\mathcal J$ is large. If your optimal weights have high values, $\mathcal R$ is large. 

+ Large $\lambda$ penalizes weight values more. 

+ Like $M, \lambda$ is a hyperparameter we can tune with a validation set

+ The geometric picture:

  <img src="images\image-20210913174500531.png" alt="image-20210913174500531" style="zoom:80%;" />

##### L2 Regularized Least Squares: Ridge regression

For the least squares problem, we have $\mathcal J(w) = \frac{1}{2N} ||Xw — t||^2$.

+ When $\lambda > 0$ (with regularization), regularized cost gives
  $$
  \begin{align*}
  w_\lambda^{Ridge}=\underset {w}{\operatorname {argmin} }\, \mathcal J_{reg}(w)&=\underset {w}{\operatorname {argmin}}\frac{1}{2N}||Xw-t||_2^2+\frac{\lambda}{2}||w||_2^2\\
  &=(X^TX+\lambda I)^{-1}X^Tt
  \end{align*}
  $$

+ The case $\lambda=0$ (no regularization) reduces to least squares solution!

+ Note that it is also common to formulate this problem as $\underset {w}{\operatorname {argmin}}\frac{1}{2}||Xw-t||_2^2+\frac{\lambda}{2}||w||_2^2$ in which case the solution is 
  $w_\lambda^{Ridge}=(X^TX+\lambda I)^{-1}X^Tt$

#### Conclusion so far

+ In gradient descent, the learning rate a is a hyperparameter we need to tune. Here are some things that can go wrong:

  <img src="images\image-20210913210100346.png" alt="image-20210913210100346" style="zoom:80%;" />

+ Good values are typically between $0.001$ and $0.1$. You should do a grid search if you want good performance (i.e. try $0.1,0.03,0.01,...)$.

#### Gradient Descent

+ Now let’s see a second way to minimize the cost function which is more broadly applicable: <span style="color:blue">gradient descent</span>. 

+ Many times，we do not have a direct solution: Taking derivatives of $\mathcal J$ w.r.t $w$ and setting them to 0 doesn’t have an explicit solution. 

  <Span style="color:rgb(130,50,150)">***PN: w.r.t $\to$ with respect to***</span>

+ Gradient descent is an <span style="color:blue">iterative algorithm</span>, which means we apply 
  an update repeatedly until some criterion is met. 

+ We <span style="color:blue">initialize</span> the weights to something reasonable (e.g. all zeros) and repeatedly adjust them in the <span style="color:blue">direction of steepest descent</span>.

  <img src="images\image-20210913174751099.png" alt="image-20210913174751099" style="zoom:80%;" />

+ Observe: 

  + if $\partial \mathcal J/\partial w_j > 0$, then increasing $W_j$ increases $\mathcal J$. 
  + if $\partial \mathcal J/\partial w_j < 0$, then increasing $W_j$ decreases $\mathcal J$. 

+ The following update always decreases the cost function for small enough $\alpha$ (unless $\partial J/\partial w_j = 0)$:
  $$
  \begin{align*}
  w_j \leftarrow w_j -\alpha\frac{\partial \mathcal J}{\partial w_j}
  \end{align*}
  $$

+ $\alpha > 0$ is a <span style="color:blue">learning rate</span> (or step size). The larger it is, the faster $w$ changes. 

  + We'll see later how to tune the learning rate, but values are typically small, e.g. $0.01$ or $0.0001$. 
  + If cost is the sum of $N$ individual losses rather than their average, smaller learning rate will be needed $(\alpha' = \alpha/N)$.

+ This gets its name from the <span style="color:blue">gradient</span>:
  $$
  \begin{align*}
  \nabla_w\mathcal J = \frac{\partial\mathcal J}{\partial w}=
  \begin{pmatrix}
  \frac{\partial \mathcal J}{\partial w_1}\\
  \vdots\\
  \frac{\partial \mathcal J}{\partial w_D}
  \end{pmatrix}
  \end{align*}
  $$

  + This is the direction of fastest increase in $\mathcal J$.

+ Update rule in vector form:
  $$
  \begin{align*}
  w\leftarrow w-\alpha\frac{\partial \mathcal J}{\partial w}
  \end{align*}
  $$
  And for linear regression we have:
  $$
  \begin{align*}
  w\leftarrow w-\frac{\alpha}{N}\sum_{i=1}^N(y^{(i)}-t^{(i)})x^{(i)}
  \end{align*}
  $$

+ So gradient descent updates w in the direction of fastest decrease. 

+ Observe that once it converges, we get a critical point，i.e. $\frac{\partial \mathcal J}{\partial w} = 0$.

##### Gradient Descent for Linear Regression

+ The squared error loss of linear regression is a convex function.
+ Even for linear regression, where there is a direct solution, we 
  sometimes need to use GD. 
+ Why gradient descent, if we can find the optimum directly? 
  + GD can be applied to a much broader set of models 
  + GD can be easier to implement than direct solutions 
  + For regression in high-dimensional space, GD is more efficient than direct solution 
    + Linear regression solution: $(X^TX)^{-1}X^Tt $
    + Matrix inversion is an $\mathcal O(D^3)$ algorithm 
    + Each GD update costs $\mathcal O(ND)$
    + Or less with stochastic GD(SGD, in a few slides) 
    + Huge difference if $D\gg1$

##### Gradient Descent under the L2 Regularization

+ Gradient descent update to minimize $\mathcal J$
  $$
  \begin{align*}
  w\leftarrow w-\alpha\frac{\partial }{\partial w}\mathcal J
  \end{align*}
  $$

+ The gradient descent update to minimize the $L^2$ regularized cost $\mathcal J + \lambda \mathcal R$ results in <span style="color:blue">weight decay</span>:
  $$
  \begin{align*}
  w&\leftarrow w-\alpha\frac{\partial }{\partial w}(\mathcal J+\lambda \mathcal R)\\
  &=w-\alpha(\frac{\partial \mathcal J}{\partial w}+\lambda \frac{\partial \mathcal R}{\partial w})\\
  &=w-\alpha(\frac{\partial \mathcal J}{\partial w}+\lambda w)\\
  &=(1-\alpha\lambda)w-\alpha\frac{\partial \mathcal J}{\partial w}\\
  \end{align*}
  $$

##### Learning Rate (Step Size)

+ In gradient descent, the learning rate $\alpha$ is a hyperparameter we need to tune. Here are some things that can go wrong:

  <img src="images\image-20210913213443454.png" alt="image-20210913213443454" style="zoom:80%;" />

+ Good values are typically between $0.001$ and $0.1$. You should do a grid search if you want good performance (i.e. try $0.1,0.03,0.01,....$)

##### Training Curves

+ To diagnose optimization problems, it’s useful to look at <span style="color:blue">training curves</span>: plot the training cost as a function of iteration.

  <img src="images\image-20210913213555487.png" alt="image-20210913213555487" style="zoom:80%;" />

+ Warning: in general, it’s very hard to tell from the training curves whether an optimizer has converged. They can reveal major problems, but they can’t guarantee convergence.

#### Stochastic Gradient Descent

+ So far, the cost function $\mathcal J$ has been the average loss over the training examples:
  $$
  \mathcal J(\theta) = \frac{1}{N}\sum_{t=1}^N\mathcal L^{(i)}=\frac{1}{N}\sum_{i=1}^N\mathcal L(y(x^{(i)},\theta),t^{(i)}).
  $$
  ($\theta$ denotes the parameters; e.g.，in linear regression, $\theta = (w，b))$

+ By linearity,
  $$
  \frac{\partial\mathcal J}{\partial \theta}=\frac{1}{N}\sum_{i=1}^N\frac{\partial\mathcal L^{(i)}}{\partial \theta}.
  $$

+ Computing the gradient requires summing over all of the training examples. This is known as <span style="color:blue">batch training</span>. 

+ Batch training is impractical if you have a large dataset $N\gg1$ (e.g. millions of training examples)!

+ <span style="color:blue">Stochastic gradient descent (SGD)</span>: update the parameters based on the gradient for a single training example.

  + 1—     Choose $i$ uniformly at random.
  + 2—     $\theta\leftarrow\theta-\alpha\frac{\partial\mathcal L^{(i)}}{\partial\theta}$

+ Cost of each SGD update is independent of $N$! 

+ SGD can make significant progress before even seeing all the data! 

+ Mathematical justification: if you sample a training example uniformly at random, the stochastic gradient is an <span style="color:blue">unbiased estimate</span> of the batch gradient:
  $$
  \begin{align*}
  \Bbb E[\frac{\partial\mathcal L^{(i)}}{\partial\theta}]=\frac{1}{N}\sum_{i=1}^N\frac{\partial\mathcal L^{(i)}}{\partial\theta}=\frac{\partial\mathcal J}{\partial\theta}
  \end{align*}
  $$

+ Problems with using single training example to estimate gradient:

  + Variance in the estimate may be high 
  + We can’t exploit efficient vectorized operations

+ Compromise approach:

  + compute the gradients on a randomly chosen medium-sized set of training examples $\mathcal M\sub\{1,...,N\}, $ called a <span style="color:blue">min-batch</span>.

+ Stochastic gradients computed on larger mini-batches have smaller variance.

+ The mini-batch size $|\mathcal M|$ is a hyperparameter that needs to be set. 

  + Too large: requires more compute; e.g., it takes more memory to 
    store the activations, and longer to compute each gradient update 
  + Too small: can’t exploit vectorization, has high variance 
  + A reasonable value might be $|\mathcal M|=100$

+ Batch gradient descent moves directly downhill (locally speaking). 

+ SGD takes steps in a noisy direction, but moves downhill on 
  average.

  <img src="images\image-20210913175904577.png" alt="image-20210913175904577" style="zoom: 67%;" />

##### SGD Learning Rate

+ In stochastic training, the learning rate also influences the <span style="color:blue">fluctuations</span> clue to the stochasticity of the gradients.

  <img src="images\image-20210913175941265.png" alt="image-20210913175941265" style="zoom:80%;" />

+ Typical strategy:

  + Use a large learning rate early in training so you can get close to the optimum 
  + Gradually decay the learning rate to reduce the fluctuations

#### Conclusion

+ In this lecture, we looked at linear regression, which exemplifies a modular approach that will be used throughout this course: 
  + choose a <span style="color:blue">model</span> describing the relationships between variables of interest (**linear**) 
  + define a <span style="color:blue">loss function</span> quantifying how bad the fit to the data is (**squared error**) 
  + choose a <span style="color:blue">regularizer</span> to control the model complexity/ over fit ting ($L^2，L^p$ **regularization**) 
  + fit$/$optimize the model (**gradient descent, stochastic gradient descent, convexity**) 
+ By mixing and matching these modular components, we can obtain new ML methods. 
+ Next lecture: apply this framework to classification

## Lecture 3

### Content

>TBD

### Overview

+ <span style="color:blue">Classification</span>: predicting a discrete-valued target 
  + <span style="color:blue">Binary classification</span>: predicting a binary-valued target 
  + <span style="color:blue">Multiclass classification</span>: predicting a discrete($>2$)-valued target

+ Examples of binary classification 
  + predict whether a patient has a disease? given the presence or 
    absence of various symptoms 
  + classify e-mails as spam or non-spam
  + predict whether a financial transaction is fraudulent

**Binary linear classification **

+ **classification**: given a $D$-dimensional input $x \in \Bbb R^D$ predict $a$ discrete-valued target 
+ **binary**: predict a binary target $t \in {0,1} $
  + Training examples with $t = 1$ are called positive examples, and training examples with $t = 0$ are called negative examples. Sorry. 
  + $t \in \{0, 1\}$ or $t \in \{-1, +1\}$ is for coinput at ioiial convenience.  

+ **linear**: model prediction $y$ is a linear function of $x$, followed by a threshold $r$:

  $$
  \begin{array}{l}
  z=\mathbf{w}^{\top} \mathbf{x}+b \\
  y=\left\{\begin{array}{ll}
  1 & \text { if } z \geq r \\
  0 & \text { if } z<r
  \end{array}\right.
  \end{array}
  $$

### Simplifications

**Eliminating the threshold **

+ We can assume without loss of generality (WLOG) that the threshold $r = 0$:
  $$
  \mathbf{w}^{\top} \mathbf{x}+b \geq r \quad \Longleftrightarrow \quad \mathbf{w}^{\top} \mathbf{x}+\underbrace{b-r}_{\triangleq w_{0}} \geq 0
  $$

**Eliminating the bias **

+ Add a dummy feature $x_0$ which always takes the value 1. The weight $w_o = b$ is equivalent to a bias (same as linear regression) 

**Simplified model**

+ Receive input $x \in \Bbb R^{D+1}$ with $x_0=1$:
  $$
  \begin{array}{l}
  z=\mathbf{w}^{\top} \mathbf{x} \\
  y=\left\{\begin{array}{ll}
  1 & \text { if } z \geq 0 \\
  0 & \text { if } z<0
  \end{array}\right.
  \end{array}
  $$

#### **Examples**

+ Let’s consider some simple examples to examine the properties of our model 

+ Let’s focus on minimizing the training set error, and forget about whether our model will generalize to a test set.
  $$
  \begin{array}{l}
  \quad\textbf {NOT }\\
  \begin{array}{cc|c}
  x_{0} & x_{1} & \mathrm{t} \\
  \hline 1 & 0 & 1 \\
  1 & 1 & 0
  \end{array}
  \end{array}
  $$

+ Suppose this is our training set, with the dummy feature $x_0$ included. 

+ Which conditions on $w_0, w_1$ guarantee perfect classification?

  + When $x_1 =0$, need: $z = w_0x_0 + w_1x_1\geq0\Longleftrightarrow w_0\geq 0$ 

  + When $x_1 =1$, need: $z = w_0x_0 + w_1x_1\geq0\Longleftrightarrow w_0+w_1< 0$ 

+  Example solution: $w_0=1$, $w_1 = -2$

+ Is this the only solution?
  $$
  \textbf { AND }\\
  \begin{array}{ccc|lc}
  x_{0} & x_{1} & x_{2} & \mathrm{t} & z=w_{0} x_{0}+w_{1} x_{1}+w_{2} x_{2} \\
  \hline 1 & 0 & 0 & 0 & \text { need: } w_{0}<0 \\
  1 & 0 & 1 & 0 & \text { need: } w_{0}+w_{2}<0 \\
  1 & 1 & 0 & 0 & \text { need: } w_{0}+w_{1}<0 \\
  1 & 1 & 1 & 1 & & \\
  & & & & \text { need: } w_{0}+w_{1}+w_{2} \geq 0
  \end{array}
  $$
  Example solution: $w_0 = -1.5, w_1 =1, w_2 = 1$

### The Geometric Picture

**<span style="color:blue">Input Space</span>, or <span style="color:blue">Data Space</span> for NOT example**

<img src="images\image-20210918195635241.png" alt="image-20210918195635241"  />

+ Weights (hypothesis) **w** are points

+ Weights (hypotheses) w can be represented by <span style="color:blue">half-spaces</span>

  $H_+ = \{x:w^\top x\geq0\}, H_-=\{x:w^\top x< 0\}$

  + The boundaries of these half-spaces pass through the origin (why?)

+ The boundary is the <span style="color:blue">decision boundary</span>: $\{x:w^\top x=0\}$
  + In 2-D, it's a line, but in high dimensions it is a hyperplane

+ If the triaining examples can be perfectly sparated by a linear decision rule, we say <span style="color:blue">data is linearly separable</span>.

**<span style="color:blue">Weight Space</span>**

<img src="images\image-20210918200557690.png" alt="image-20210918200557690"  />

+ Weights (hypotheses) **w** are points

+ Each training example x specifies a half-space **w** must lie in to be correctly classified: $w^\top x\geq 0$ if $t=1$.

+ For NOT example:
  + $x_{0}=1, x_{1}=0, t=1 \Longrightarrow\left(w_{0}, w_{1}\right) \in\left\{\mathbf{w}: w_{0} \geq 0\right\}$
  + $x_{0}=1, x_{1}=0, t=1 \Longrightarrow\left(w_{0}, w_{1}\right) \in\left\{\mathbf{w}: w_{0}+w_1 < 0\right\}$

+ The region satisfying all the constraints is the <span style="color:blue">feasible region</span>; if this region is nonempty, the problem is <span style="color:blue">feasible</span>, otw it is <span style="color:blue">infeasible</span>.

  <Span style="color:rgb(130,50,150)">***PN: otw $\rightarrow$ otherwise***</span>

+ The AND example requires three dimensions, including the dummy one. 

+ To visualize data space and weight space for a 3-D example, we can look at a 2-D slice. 

+ The visualizations are similar. 
  + Feasible set will always have a corner at the origin.

Visualizations of the **AND** example

<img src="images\image-20210918201435274.png" alt="image-20210918201435274"  />

1. Graph 1

   \- Slice for $x_0=1$ and 

   \- example sol: $w_0=-1.5, w_1=1,w_2=1$

   \- decision boundary:$w_0x_0+w_1x_1+w_2x_2=0 \Longrightarrow -1.5+x_1+x_2=0$

2. Graph 2 

   \- Slice for $w_0=-1.5$ for the constraints

   \- $w_0<0$

   \- $w_0+w_2<0$

   \- $w_0+w_1<0$

   \- $w_0+w_1+w_2\geq0$

#### Summary | Binary Linear Classifiers

+ Summary: Targets $t\in \{0,1\}$, inputs $x\in \Bbb R^{D+1}$ with $x_0=1$, and model is defines by weights **w** and
  $$
  \begin{align*}
  \begin{array}{l}
  z=\mathbf{w}^{\top} \mathbf{x} \\
  y=\left\{\begin{array}{ll}
  1 & \text { if } z \geq 0 \\
  0 & \text { if } z<0
  \end{array}\right.
  \end{array}
  \end{align*}
  $$

+ How can we find good values for **w**? 
+ If training set is linearly separable, we could solve for **w** using linear programming 
  + We could also apply an iterative procedure known as the perceptron algorithm (but this is primarily of historical interest). 

+ If it's not linearly separable, the problem is harder 
  + Data is almost never linearly separable in real life.

### Logistic Regression

#### Loss Functions

+ Instead: define loss function then try to minimize the resulting cost function 
  + Recall: cost is loss averaged (or summed) over the training set 
  
+ Seemingly obvious loss function: <span style="color:blue">0-1 loss</span>
  $$
  \begin{aligned}
  \mathcal{L}_{0-1}(y, t) &=\left\{\begin{array}{cc}
  0 & \text { if } y=t \\
  1 & \text { if } y \neq t
  \end{array}\right.\\
  &=\mathbb{I}[y \neq t]
  \end{aligned}
  $$

##### Attempt 1: 0-1 loss

+ Usually, the cost $\mathcal J$ is the averaged loss over training examples; for 0-1 loss, this is the <span style="color:blue">misclassification rate</span>: 
  $$
  \mathcal{J}=\frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left[y^{(i)} \neq t^{(i)}\right]
  $$

+ Problem: liow to optimize? In general, a hard problem (can be NP-hard) 
+ This is clue to the step function (0-1 loss) not being nice (continuous / smooth /convex etc)

+ Minimum of a function will be at its critical points. 

+ Let’s try to find the critical point of 0-1 loss 

+ Chain rule:
  $$
  \frac{\partial \mathcal{L}_{0-1}}{\partial w_{j}}=\frac{\partial \mathcal{L}_{0-1}}{\partial z} \frac{\partial z}{\partial w_{j}}
  $$
  
+ But $\partial \mathcal{L}_{0-1}/\partial z$ is zero everywhere it’s defined!

  <img src="images\image-20210918231718595.png" alt="image-20210918231718595"  />

  + $\partial \mathcal{L}_{0-1}/\partial w_{j}=0$ means that changing the weights by a very small amount probably has no effect on the loss. 
  + Almost any point has 0 gradient!

##### Attempt 2: Linear Regression

+ Sometimes we can replace the loss function we care about with one which is easier to optimize. This is known as <span style="color:blue">relaxation</span> with a smooth <span style="color:blue">surrogate loss function</span>.

+ One problem with $\mathcal{L}_{0-1}$: defined in terms of final prediction, which inherently involves a discontinuity
+ Instead, define loss in terms of $w^Tx$ directly 
  + Redo notation for convenience: $z=w^Tx$

+ We already know liow to fit a linear regression model. Can we use this instead?
  $$
  \begin{aligned}
  z &=\mathbf{w}^{\top} \mathbf{x} \\
  \mathcal{L}_{\mathrm{SE}}(z, t) &=\frac{1}{2}(z-t)^{2}
  \end{aligned}
  $$

+ Doesn’t matter that the targets are actually binary. Treat them as continuous values. 
+ For this loss function, it makes sense to make final predictions by thresholding $z$ at $\frac{1}{2}$(why?)

**The problem:**

<img src="images\image-20210919000446361.png" alt="image-20210919000446361" style="zoom:80%;" />

+ The loss function hates when you make correct predictions with high confidence! 
+ If $t = 1$, it’s more unhappy about $z = 10$ than $z = 0$.

##### Attempt 3: Logistic Activation Function

+ There’s obviously no reason to predict values outside $[0,1]$. Let’s squash $y$ into this interval. 

+ The <span style="color:blue">logistic  function</span>  is a  kind  of  <span style="color:blue">sigmoid</span>,  or S-shaped function:
  
  <img src="C:\Users\ADMIN\AppData\Roaming\Typora\typora-user-images\image-20210919000941719.png" alt="image-20210919000941719"  />
  $$
  \begin{align*}
  \sigma(z) = \frac{1}{1+e^{-z}}
  \end{align*}
  $$
  
+ $\sigma^{-1}(y)=\log(y/(1-y))$ is called the <span style="color:blue">logit</span>. 
  $$
  \begin{align*}
  \text{let } x &= \text{logit}(y) = \log\frac{y}{1-y}\\
  e^x &= \frac{y}{1-y}\\
  1+e^x &= 1+\frac{y}{1-y} = \frac{1}{1-y}\\
  \frac{1}{1+e^x} &= 1-y\\
  y &= 1-\frac{1}{1+e^x} = \frac{e^x}{1+e^x}=\frac{1}{1+e^{-x}} = \sigma(z)
  \end{align*}
  $$
  <Span style="color:rgb(130,50,150)">***PN: $\sigma'(z) = \sigma(z)(1-\sigma(z))$***</span>

+ A linear model with a logistic nonlinearity is known as <span style="color:blue">log-linear</span>:
  $$
  \begin{aligned}
  z &=\mathbf{w}^{\top} \mathbf{x} \\
  y &= \sigma(z)\\
  \mathcal{L}_{\mathrm{SE}}(y, t) &=\frac{1}{2}(y-t)^{2}
  \end{aligned}
  $$
  
+ Used in this way, $\sigma$ is called an <span style="color:blue">activation function</span>.

**The problem:** 
(plot of $\mathcal L_{SE}$ as a function of $z$, assuming $t = 1$)

<img src="images\image-20210919001225654.png" alt="image-20210919001225654"  />

+ For z $\ll0$ , we have $\sigma(z) \approx 0.$ 
+ $\frac{\partial \mathcal L}{\partial z} \approx 0 (\text{check!}) \Longrightarrow \frac{\partial \mathcal L}{\partial w_j} \approx0 => \text{derivative w.r.t. } w_j$ is small $\Longrightarrow w_j$ is like a critical point 
+ If the prediction is really wrong, you should be far from a critical point (which is your candidate solution).

#### Logistic Regression

+ Because $y \in [0,1]$, we can interpret it as the estimated probability that $t = 1$. If $t = 0$, then we want to heavily penalize $y\approx 1$. 

+ The pundits who were $99\%$ confident Clinton would win were much more wrong than the ones who were only $90\%$ confident. 

+ <span style="color:blue">Cross-entropy</span> loss (aka log loss) captures this intuition:

  <Span style="color:rgb(130,50,150)">***PN: aka $\rightarrow$ as known as***</span>
  
  
  
  <img src="images\image-20210919002108339.png" alt="image-20210919002108339"  />
  $$
  \begin{aligned}
  \mathcal{L}_{\mathrm{CE}}(y, t) &=\left\{\begin{array}{ll}
  -\log y & \text { if } t=1 \\
  -\log (1-y) & \text { if } t=0
  \end{array}\right.\\
  &=-t \log y-(1-t) \log (1-y)
  \end{aligned}
  $$

**<span style="color:blue">Logistic Regression:</span>**

**Plot is for target $t=1$**
$$
\begin{aligned}
\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial z}=\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial y} \cdot \frac{\partial y}{\partial z}  &=\left(-\frac{t}{y}+\frac{1-t}{1-y}\right) \cdot y(1-y) \\
&=y-t
\end{aligned}
$$
<Span style="color:rgb(130,50,150)">***PN: When $z\rightarrow-3,\, y\rightarrow 0,\, \mathcal L_{CE}\rightarrow 3,\, y-t\rightarrow -1 (t=1).$***</span>

<Span style="color:rgb(130,50,150)">***Hence, the dashed line is tangent to the curve at the point $z = -3$:*** </span>
$$
\begin{align*}
d= (y-t)z+3 \Longrightarrow d = -z+3
\end{align*}
$$
<img src="images\image-20210919173531778.png" alt="image-20210919173531778" style="zoom:80%;" />
$$
\begin{align*}
z &= w^\top x\\
y &= \sigma(z)\\
&=\frac{1}{1+e^{-z}}\\
\mathcal L_{CE} &= -t\log y-(1-t)\log(1-y)\\

\end{align*}
$$

##### Gradient Descent for Logistic Regression

+ How do we minimize the cost $\mathcal J$ for logistic regression? No direct solution.
  + Taking derivatives of $\mathcal J$ w.r.t. **w** and setting them to 0 doesn't have an explicit solution.
  
    **Related link:** 
  
    1. [概率视角下的线性模型：逻辑回归有解析解吗？](https://kexue.fm/archives/8578)
    2. [Easy Logistic Regression with an Analytical Solution](https://towardsdatascience.com/easy-logistic-regression-with-an-analytical-solution-eb4589c2cd2d)
+ However, the logistic loss is a <span style="color:blue">convex function</span> in **w**, so let’s consider the <span style="color:blue">gradient descent</span> method from last lecture. 
  + Recall: we <span style="color:blue">initialize</span> the weights to something reasonable and repeatedly adjust them in the <span style="color:blue">direction of steepest descent</span>. 
  
  + A standard initialization is $w = 0$ (why?)
  
    <Span style="color:rgb(130,50,150)">***Andrew Ng***: *Logistic Regression doesn’t have a hidden layer. If you initialize the weights to zeros, the first example x fed in the logistic regression will output zero but the derivatives of the Logistic Regression depend on the input x (because there’s no hidden layer) which is not zero. So at the second iteration, the weights values follow x’s distribution and are different from each other if x is not a constant vector.*</span>

##### Gradient of Logistic Loss

$$
\begin{aligned}
\mathcal{L}_{\mathrm{CE}}(y, t) &=-t \log (y)-(1-t) \log (1-y) \\
y &=1 /\left(1+e^{-z}\right) \text { and } z=\mathbf{w}^{\top} \mathbf{x}
\end{aligned}
$$
Therefore
$$
\begin{aligned}
\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial w_{j}}=\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w_{j}} &=\left(-\frac{t}{y}+\frac{1-t}{1-y}\right) \cdot y(1-y) \cdot x_{j} \\
&=(y-t) x_{j}
\end{aligned}
$$
Gradient descent (coordinate-wise) update to find the weights of logistic regression:
$$
\begin{aligned}
w_{j} & \leftarrow w_{j}-\alpha \frac{\partial \mathcal{J}}{\partial w_{j}} \\
&=w_{j}-\frac{\alpha}{N} \sum_{i=1}^{N}\left(y^{(i)}-t^{(i)}\right) x_{j}^{(i)}
\end{aligned}
$$

##### Gradient Descent for Logistic Regression

Comparison of gradient descent updates:
- Linear regression:
$$
\mathbf{w} \leftarrow \mathbf{w}-\frac{\alpha}{N} \sum_{i=1}^{N}\left(y^{(i)}-t^{(i)}\right) \mathbf{x}^{(i)}
$$
- Logistic regression:
$$
\mathbf{w} \leftarrow \mathbf{w}-\frac{\alpha}{N} \sum_{i=1}^{N}\left(y^{(i)}-t^{(i)}\right) \mathbf{x}^{(i)}
$$
- Not a coincidence! These are both examples of <span style="color:blue">generalized linear models</span>. But we won't go in further detail.
- Notice $\frac{1}{N}$ in front of sums due to averaged losses. This is why you need smaller learning rate when cost is summed losses $\left(\alpha^{\prime}=\alpha / N\right)$.

### Multiclass Classification and Softmax Regression 

#### Overview

+ <span style="color:blue">Classification</span>: predicting a discrete-valued target
  + <span style="color:blue"> Binary classification</span> predicting a binary-valued target 
  + <span style="color:blue">Multiclass classification</span>: predicting a discrete($> 2$)-valued target
+ Examples of multi-class classification 
  + predict the value of a handwritten digit
  + classify e-mails as spam, travel, work, personal 

#### Multiclass Classification

+ Classification tasks with more than two categories

  <img src="images\image-20210919003612256.png" alt="image-20210919003612256"  />

+ Targets form a discrete set $\{1,...,K\}$

+ It’s often more convenient to represent them as <span style="color:blue">one-hot vectors</span>, or a <span style="color:blue">one-of-K encoding</span>:
  $$
  \begin{align*}
  \mathbf{t}=\underbrace{(0, \ldots, 0,1,0, \ldots, 0)}_{\text {entry } k \text { is } 1} \in \mathbb{R}^{K}
  \end{align*}
  $$

##### Multiclass Linear Classification

- We can start with a linear function of the inputs.
- Now there are $D$ input dimensions and $K$ output dimensions, so we need $K \times D$ weights, which we arrange as a <span style="color:blue">weight matrix</span> $\mathbf{W}$.
- Also, we have a $K$-dimensional vector $\mathbf{b}$ of biases.
- A linear function of the inputs:
$$
z_{k}=\sum_{j=1}^{D} w_{k j} x_{j}+b_{k} \text { for } k=1,2, \ldots, K
$$
- We can eliminate the bias b by taking $\mathbf{W} \in \mathbb{R}^{K \times(D+1)}$ and adding a dummy variable $x_{0}=1 .$ So, vectorized:
  $$
  \begin{align*}
  \mathbf{z}=\mathbf{W} \mathbf{x}+\mathbf{b} \quad \text{or with dummy }x_{0}=1 \quad \mathbf{z}=\mathbf{W} \mathbf{x}
  \end{align*}
  $$

- How can we turn this linear prediction into a <span style="color:blue">one-hot prediction</span>?
- We can interpret the magnitude of $z_{k}$ as an measure of how much the model prefers $k$ as its prediction.
- If we do this, we should set

$$
y_{i}=\left\{\begin{array}{ll}
1 & i=\arg \max _{k} z_{k} \\
0 & \text { otherwise }
\end{array}\right.
$$

##### Softmax Regression

- We need to soften our predictions for the sake of optimization.

- We want soft predictions that are like probabilities, i.e., $0 \leq y_{k} \leq 1$ and $\sum_{k} y_{k}=1$

- A natural activation function to use is the <span style="color:blue">softmax function</span>, a multivariable generalization of the logistic function:
  $$
  \begin{align*}
  y_{k}=\operatorname{softmax}\left(z_{1}, \ldots, z_{K}\right)_{k}=\frac{e^{z_{k}}}{\sum_{k^{\prime}} e^{z_{k^{\prime}}}}
  \end{align*}
  $$

  + Outputs can be interpreted as probabilities (positive and sum to 1 )

  + If $z_{k}$ is much larger than the others, then $\operatorname{softmax}(\mathbf{z})_{k} \approx 1$ and it behaves like argmax.

- If a model outputs a vector of class probabilities, we can use cross-entropy as the loss function:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{CE}}(\mathbf{y}, \mathbf{t}) &=-\sum_{k=1}^{K} t_{k} \log y_{k} \\
&=-\mathbf{t}^{\top}(\log \mathbf{y})
\end{aligned}
$$
​		where the $\log$ is applied elementwise.
- Just like with logistic regression, we typically combine the softmax and cross-entropy into a <span style="color:blue">softmax-cross-entropy</span> function.

- <span style="color:blue">Softmax regression</span> (with dummy $x_{0}=1$ ):
$$
\begin{aligned}
\mathbf{z} &=\mathbf{W} \mathbf{x} \\
\mathbf{y} &=\operatorname{softmax}(\mathbf{z}) \\
\mathcal{L}_{\mathrm{CE}} &=-\mathbf{t}^{\top}(\log \mathbf{y})
\end{aligned}
$$
- Gradient descent updates can be derived for each row of $\mathbf{W}$ :
$$
\begin{aligned}
\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial \mathbf{w}_{k}} &=\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial z_{k}} \cdot \frac{\partial z_{k}}{\partial \mathbf{w}_{k}}=\left(y_{k}-t_{k}\right) \cdot \mathbf{x} \\
\mathbf{w}_{k} & \leftarrow \mathbf{w}_{k}-\alpha \frac{1}{N} \sum_{i=1}^{N}\left(y_{k}^{(i)}-t_{k}^{(i)}\right) \mathbf{x}^{(i)}
\end{aligned}
$$
- Similar to linear/logistic regression (no coincidence) 

##### Prove the gradient ? 

$$
\begin{flalign*}
&\text{To prove }\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial \mathbf{w}_{k}} =\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial z_{k}} \cdot \frac{\partial z_{k}}{\partial \mathbf{w}_{k}}=\left(y_{k}-t_{k}\right) \cdot \mathbf{x} \\\\

\enclose{circle}{1}&\frac{\partial \mathcal L_{CE}}{\partial y_T} = -\frac{1}{y_T}, (t_T=1,t_{m\neq T}=0)\\
&(PN: \mathcal{L}_{\mathrm{CE}} =-\mathbf{t}^{\top}(\log \mathbf{y}))\\

\enclose{circle}{2}&\mathit{If }\,\, K = T:\\
&\frac{\partial y_T}{z_k} = \frac{\partial y_k}{\partial z_k} =\frac{\partial}{\partial z_k}(\frac{e^{z_k}}{\sum_{k^{'}}e^{z_{k^{'}}}})=e^{z_k}(\sum_{k^{'}}e^{z_{k^{'}}})^{-1}+e^{z_k}(-1)(\sum_{k^{'}}e^{z_{k^{'}}})^{-2}e^{z_k}\\
& =e^{z_k}(\sum_{k^{'}}e^{z_{k^{'}}})^{-1}(1-e^{z_k}(\sum_{k^{'}}e^{z_{k^{'}}})^{-1}) \\
&=y_k(1-y_k)\\
&\mathit{If }\,\, K \neq T:\\
&\frac{\partial y_T}{z_k} =\frac{\partial}{\partial z_k}(\frac{e^{z_T}}{\sum_{k^{'}}e^{z_{k^{'}}}})=\frac{\partial}{\partial z_k}(e^{z_T}(\sum_{k^{'}}e^{z_{k^{'}}})^{-1})\\
&=e^{z_T}(-1)(\sum_{k^{'}}e^{z_{k^{'}}})^{-2}e^{z_{k^{'}}} =(\frac{e^{z_T}}{\sum_{k^{'}}e^{z_{k^{'}}}})\vdot (\frac{e^{z_k}}{\sum_{k^{'}}e^{z_{k^{'}}}})\\
&=-y_Ty_k\\


\enclose{circle}{3}&(a)\mathit{If }\,\, t_k = 1,(or:T=k):\\
&\frac{\partial \mathcal L_{CE}}{\partial z_k} = \frac{\partial \mathcal L_{CE}}{\partial y_T}\vdot \frac{\partial y_T}{\partial z_k} = -\frac{1}{y_k}\vdot y_k\vdot (1-y_k)=y_k-1=y_k-t_k(t_k=1)\\
&(b)\mathit{If }\,\, t_k \neq 1,(or:T\neq k) \rightarrow t_k=0:\\
&\frac{\partial \mathcal L_{CE}}{\partial z_k} = \frac{\partial \mathcal L_{CE}}{\partial y_T}\vdot \frac{\partial y_T}{\partial z_k}=-\frac{1}{y_k}\vdot (-y_Ty_k)=y_k=y_k-0=y_k-t_k(t_k=0)\\


\enclose{circle}{4}& \frac{\partial z_k}{\partial w_k}=x\\



\enclose{circle}{5}& \frac{\partial \mathcal L_{CE}}{\partial w_k}=\frac{\partial \mathcal L_{CE}}{\partial z_k}\frac{\partial z_k}{\partial w_k} = (y_k-t_k)\vdot x\\
\end{flalign*}
$$

<Span style="color:rgb(130,50,150)">***PN:***</span>

1. <Span style="color:rgb(130,50,150)">***$\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial \mathbf{w}_{k}} =\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial z_{k}} \cdot \frac{\partial z_{k}}{\partial \mathbf{w}_{k}} = \frac{\partial \mathcal L_{CE}}{\partial y_T}\vdot \frac{\partial y_T}{\partial z_k}\vdot \frac{\partial z_{k}}{\partial \mathbf{w}_{k}}$***</span>
2. <Span style="color:rgb(130,50,150)">***$\frac{\partial \mathcal L_{CE}}{\partial y_T} = -\frac{1}{y_T}$***</span>
3. <Span style="color:rgb(130,50,150)">***$\frac{\partial y_T}{z_k} = y_k(1-y_k)\text{ when K=T or }\frac{\partial y_T}{z_k}=-y_Ty_k \text{ when K}\neq T $***</span>
4. <Span style="color:rgb(130,50,150)">***$\frac{\partial \mathcal L_{CE}}{\partial z_k} = \frac{\partial \mathcal L_{CE}}{\partial y_T}\vdot \frac{\partial y_T}{\partial z_k}=y_k-t_k$***</span>
5. <Span style="color:rgb(130,50,150)">***$\frac{\partial \mathcal L_{CE}}{\partial w}=\frac{\partial \mathcal L_{CE}}{\partial z_k}\frac{\partial z_k}{\partial w} = (y_k-t_k)\vdot x$***</span>

##### Limits of Linear Classification

Some datasets are not linearly separable, e.g. **XOR** 

<img src="images\image-20210919004705482.png" alt="image-20210919004705482"  />

Visually obvious, but how to show this? 

**Showing that XOR is not linearly separable (proof by contradiction)** 

+ If two points lie in a half-space, line segment connecting them also lie in the same halfspace. 

+ Suppose there were some feasible weights (hypothesis). If the positive examples are in the positive half-space, then the green line segment must be as well. 

+ Similarly, the red line segment must line within the negative half-space.

  <img src="images\image-20210919004838505.png" alt="image-20210919004838505"  />

+ But the intersection can't lie in both half-spaces. Contradiction! 

- Sometimes we can overcome this limitation using feature maps, just like for linear regression. E.g., for XOR:
$$
\boldsymbol{\psi}(\mathbf{x})=\left(\begin{array}{c}
x_{1} \\
x_{2} \\
x_{1} x_{2}
\end{array}\right)
$$
$$
\begin{array}{cc|ccc|c}
x_{1} & x_{2} & \psi_{1}(\mathrm{x}) & \psi_{2}(\mathrm{x}) & \psi_{3}(\mathrm{x}) & t \\
\hline 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 1 & 0 & 1 \\
1 & 0 & 1 & 0 & 0 & 1 \\
1 & 1 & 1 & 1 & 1 & 0
\end{array}
$$

- This is linearly separable. (Try it!)

### Next time...

Feature maps are hard to design well, so next time we’ll see how to *learn* nonlinear feature maps directly using neural networks...

<img src="images\image-20210919005138751.png" alt="image-20210919005138751" style="zoom:80%;" />

## Lecture 4

### Content

>TBD

### Binary Classification with a Linear Model

- Classification: Predict a discrete-valued target
- Binary classification: Targets $t \in\{-1,+1\}$
- Linear model:
$$
\begin{array}{l}
z=\mathbf{w}^{\top} \mathbf{x}+b \\
y=\operatorname{sign}(z)
\end{array}
$$
- Question: How should we choose $\mathbf{w}$ and $b$ ?

#### Zero-One Loss

- We can use the $0-1$ loss function, and find the weights that minimize it over data points
$$
\begin{aligned}
\mathcal{L}_{0-1}(y, t) &=\left\{\begin{array}{ll}
0 & \text { if } y=t \\
1 & \text { if } y \neq t
\end{array}\right.\\
&=\mathbb{I}\{y \neq t\}
\end{aligned}
$$
- But minimizing this loss is computationally difficult, and it can't distinguish different hypotheses that achieve the same accuracy.
- We investigated some other loss functions that are easier to minimize, e.g., logistic regression with the cross-entropy loss $\mathcal{L}_{\mathrm{CE}}$.
- Let's consider a different approach, starting from the geometry of binary classifiers.

#### Separating Hyperplanes

<img src="images\image-20210927170426624.png" alt="image-20210927170426624" style="zoom:67%;" />

Suppose we are given these data points from two different classes and want to find a linear classifier that separates them.

<img src="images\image-20210927170502486.png" alt="image-20210927170502486" style="zoom:67%;" />

- The decision boundary looks like a line because $\mathbf{x} \in \mathbb{R}^{2}$, but think about it as a $D-1$ dimensional hyperplane.
- Recall that a hyperplane is described by points $\mathbf{x} \in \mathbb{R}^{D}$ such that $f(\mathbf{x})=\mathbf{w}^{\top} x+b=0$

<img src="images\image-20210927170533866.png" alt="image-20210927170533866" style="zoom:67%;" />

- There are multiple separating hyperplanes, described by different parameters $(\mathbf{w}, b)$

  <img src="images\image-20210927170620034.png" alt="image-20210927170620034" style="zoom:67%;" />

### Optimal Separating Hyperplane

**Optimal Separating Hyperplane:** A hyperplane that separates two classes and maximizes the distance to the closest point from either class, i.e., maximize the **margin** of the classifier.

<img src="images\image-20210927170718842.png" alt="image-20210927170718842"  />

Intuitively, ensuring that a classifier is not too close to any data points leads to better generalization on the test data.

#### Geometry of Points and Planes

<img src="images\image-20210927170825622.png" alt="image-20210927170825622" style="zoom: 80%;" />

- Recall that the decision hyperplane is orthogonal (perpendicular) to $\mathbf{w}$.
- The vector $\mathbf{w}^{*}=\frac{\mathbf{w}}{\|\mathbf{w}\|_{2}}$ is a unit vector pointing in the same direction as $\mathbf{w}$.
- The same hyperplane could equivalently be defined in terms of $\mathrm{w}^{*}$.

The (signed) distance of a point $\mathbf{x}^{\prime}$ to the hyperplane is
$$
\frac{\mathbf{w}^{\top} \mathbf{x}^{\prime}+b}{\|\mathbf{w}\|_{2}}
$$

#### Maximizing Margin as an Optimization Problem

- Recall: the classification for the $i$-th data point is correct when
$$
\operatorname{sign}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right)=t^{(i)}
$$
- This can be rewritten as
$$
t^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right)>0
$$
- Enforcing a margin of $C$ :

$$
t^{(i)} \cdot \underbrace{\frac{\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right)}{\|\mathbf{w}\|_{2}}}_{\text {signed distance }} \geq C
$$

Max-margin objective:
$$
\begin{array}{l}
\max _{\mathbf{w}, b} C \\
\text { s.t. } \frac{t^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right)}{\|\mathbf{w}\|_{2}} \geq C \quad i=1, \ldots, N
\end{array}
$$
Plug in $C=1 /\|\mathbf{w}\|_{2}$ and simplify:
$$
\underbrace{\frac{t^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right)}{\|\mathbf{w}\|_{2}} \geq \frac{1}{\|\mathbf{w}\|_{2}}}_{\text {geometric margin constraint }} \quad \Longleftrightarrow \underbrace{t^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right) \geq 1}_{\text {algebraic margin constraint }}
$$
Equivalent optimization objective:
$$
\begin{array}{l}
\min \|\mathbf{w}\|_{2}^{2} \\
\text { s.t. } t^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right) \geq 1 \quad i=1, \ldots, N
\end{array}
$$
<img src="images\image-20210927171019995.png" alt="image-20210927171019995" style="zoom:67%;" />

Algebraic max-margin objective:
$$
\begin{array}{l}
\min _{\mathbf{w}, b}\|\mathbf{w}\|_{2}^{2} \\
\text { s.t. } t^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right) \geq 1 \quad i=1, \ldots, N
\end{array}
$$

- Observe: if the margin constraint is not tight for $\mathbf{x}^{(i)}$, we could remove it from the training set and the optimal $\mathbf{w}$ would be the same.
- The important training examples are the ones with algebraic margin 1 , and are called **support vectors**.
- Hence, this algorithm is called the (hard) **Support Vector Machine (SVM) (or Support Vector Classifier)**.
- SVM-like algorithms are often called **max-margin** or **large-margin**.

#### Non-Separable Data Points

How can we apply the max-margin principle if the data are **not** linearly separable? 

<img src="images\image-20210927171306088.png" alt="image-20210927171306088" style="zoom:67%;" />

#### Maximizing Margin for Non-Separable Data Points

<img src="images\image-20210927171334909.png" alt="image-20210927171334909" style="zoom:67%;" />

Main Idea:
- Allow some points to be within the margin or even be misclassified; we represent this with **slack variables** $\xi_{i}$.
- But constrain or penalize the total amount of slack.

- Soft margin constraint:
$$
\frac{t^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right)}{\|\mathbf{w}\|_{2}} \geq C\left(1-\xi_{i}\right)
$$
for $\xi_{i} \geq 0$
- Penalize $\sum_{i} \xi_{i}$

**Soft-margin SVM** objective:
$$
\begin{array}{ll}
\min _{\mathbf{w}, b, \xi} &\frac{1}{2}\|\mathbf{w}\|_{2}^{2}+\gamma \sum_{i=1}^{N} \xi_{i}  \\
\text { s.t. } &t^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right) \geq 1-\xi_{i} & i=1, \ldots, N \\
& \xi_{i} \geq 0 & i=1, \ldots, N
\end{array}
$$
- $\gamma$ is a hyperparameter that trades off the margin with the amount of slack.

- For $\gamma=0$, we'll get $\mathbf{w}=0$. (Why?)

  <Span style="color:rgb(130,50,150)">***PN: $\xi$ can be large enough to contain every points, so w will be 0 to make this minimum.***</span>

- As $\gamma \rightarrow \infty$ we get the hard-margin objective.

- Note: it is also possible to constrain $\sum_{i} \xi_{i}$ instead of penalizing it.

#### From Margin Violation to Hinge Loss

Let's simplify the soft margin constraint by eliminating $\xi_{i}$. Recall:
$$
\begin{array}{ll}
t^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right) \geq 1-\xi_{i} & i=1, \ldots, N \\
\xi_{i} \geq 0 & i=1, \ldots, N
\end{array}
$$
- Rewrite as $\xi_{i} \geq 1-t^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right)$.
- **Case 1:** $1-t^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right) \leq 0$
  - The smallest non-negative $\xi_{i}$ that satisfies the constraint is $\xi_{i}=0$.
- **Case 2:** $1-t^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right)>0$
  - The smallest $\xi_{i}$ that satisfies the constraint is $\xi_{i}=1-t^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right)$.
- Hence, $\xi_{i}=\max \left\{0,1-t^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right)\right\}$.
- Therefore, the slack penalty can be written as

$$
\sum_{i=1}^{N} \xi_{i}=\sum_{i=1}^{N} \max \left\{0,1-t^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right)\right\}
$$

If we write $y^{(i)}(\mathbf{w}, b)=\mathbf{w}^{\top} \mathbf{x}+b$, then the optimization problem can be written as
$$
\min _{\mathbf{w}, b, \xi} \sum_{i=1}^{N} \max \left\{0,1-t^{(i)} y^{(i)}(\mathbf{w}, b)\right\}+\frac{1}{2 \gamma}\|\mathbf{w}\|_{2}^{2}
$$
- The loss function $\mathcal{L}_{\mathrm{H}}(y, t)=\max \{0,1-t y\}$ is called the hinge loss.
- The second term is the $L_{2}$-norm of the weights.
- Hence, the soft-margin SVM can be seen as a linear classifier with hinge loss and an $L_{2}$ regularizer.

If we write $y^{(i)}(\mathbf{w}, b)=\mathbf{w}^{\top} \mathbf{x}+b$, then the optimization problem can be written as
$$
\min _{\mathbf{w}, b, \xi} \sum_{i=1}^{N} \max \left\{0,1-t^{(i)} y^{(i)}(\mathbf{w}, b)\right\}+\frac{1}{2 \gamma}\|\mathbf{w}\|_{2}^{2}
$$
- The loss function $\mathcal{L}_{\mathrm{H}}(y, t)=\max \{0,1-t y\}$ is called the hinge loss.
- The second term is the $L_{2}$-norm of the weights.
- Hence, the soft-margin SVM can be seen as a linear classifier with hinge loss and an $L_{2}$ regularizer.

#### SVM loss

<img src="images\image-20210927171950079.png" alt="image-20210927171950079" style="zoom:80%;" />

<img src="images\image-20210927172034918.png" alt="image-20210927172034918" style="zoom:80%;" />

<img src="images\image-20210927172049285.png" alt="image-20210927172049285" style="zoom:80%;" />

<img src="images\image-20210927172102951.png" alt="image-20210927172102951" style="zoom: 80%;" />

<img src="images\image-20210927172156980.png" alt="image-20210927172156980" style="zoom: 80%;" />

<img src="images\image-20210927172208528.png" alt="image-20210927172208528" style="zoom: 80%;" />

<img src="images\image-20210927172222391.png" alt="image-20210927172222391" style="zoom: 80%;" />

<img src="images\image-20210927172236650.png" alt="image-20210927172236650" style="zoom: 80%;" />

<img src="images\image-20210927172341117.png" alt="image-20210927172341117" style="zoom: 80%;" />

[0,+inf]

<img src="images\image-20210927172357259.png" alt="image-20210927172357259" style="zoom: 80%;" />

### Softmax

<img src="images\image-20210927172504468.png" alt="image-20210927172504468" style="zoom:80%;" />

<img src="images\image-20210927172523099.png" alt="image-20210927172523099" style="zoom:80%;" />

<img src="images\image-20210927172536587.png" alt="image-20210927172536587" style="zoom:80%;" />

<img src="images\image-20210927172550303.png" alt="image-20210927172550303" style="zoom:80%;" />

<img src="images\image-20210927172602967.png" alt="image-20210927172602967" style="zoom:80%;" />

<img src="images\image-20210927172613418.png" alt="image-20210927172613418" style="zoom:80%;" />

<img src="images\image-20210927172625224.png" alt="image-20210927172625224" style="zoom:80%;" />

<img src="images\image-20210927172639940.png" alt="image-20210927172639940" style="zoom:80%;" />

<img src="images\image-20210927172658754.png" alt="image-20210927172658754" style="zoom:80%;" />

<img src="images\image-20210927172713602.png" alt="image-20210927172713602" style="zoom:80%;" />

<img src="images\image-20210927172726888.png" alt="image-20210927172726888" style="zoom:80%;" />

### Softmax & SVM

<img src="images\image-20210927172750275.png" alt="image-20210927172750275" style="zoom:80%;" />
$$
\begin{align*}
\begin{array}{l}
\begin{array}{ll}
L_{i}=-\log \left(\frac{e^{s_{y_{i}}}}{\sum_{j} e^{s_{j}}}\right) & L_{i}=\sum_{j \neq y_{i}} \max \left(0, s_{j}-s_{y_{i}}+1\right) \\
\hline \text { assume scores: } & \text { Q: Suppose I take a datapoint } \\
{[10,-2,3]} & \text { and I jiggle a bit (changing its } \\
{[10,9,9]} & \text { score slightly). What happens to } \\
{[10,-100,-100]} & \text { the loss in both cases? } \\
\text { and } y_{i}=0 &
\end{array}
\end{array}
\end{align*}
$$
