# Machine Learning 

## Lecture 1 

### **Contents**

> [Nearest Neighbor](#Nearest Neighbor) (all)

1. *Coursework 1 : 15%* 

   *○ The coursework requires no lab practice.* 

2. *Coursework 2 : 15%*

    *○ This coursework requires lab practice.*

3. *Final Exam : 70%* 

   *○ Final Exam is the most important part for assessment. It will be a open  book exam.* 

#### **What is machine learning**

<img src="images\image-20210906165229750.png" alt="image-20210906165229750"  />

#### Relations to AI

#### Relations to Human Learning

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

**ML workflow sketch** 

1. Should I use ML on this problem? 
   + *Is there a pattern to detect?* 
   + *Can I solve it analytically?* 
   + *Do I have data?* 
2. Gather and organize data. 
+ *Preprocessing, cleaning, visualizing.*
3. Establishing a baseline. 
4.  Choosing a model, loss, regularization,… 
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

**Preliminaries and Nearest Neighbor Methods**

> supervised learning: This means we are given a training set consisting of inputs and corresponding labels. eg.
>
> |          Task           |     Inputs     |      Labels       |
> | :---------------------: | :------------: | :---------------: |
> |   object recognition    |     image      |  object category  |
> |    image captioning     |     image      |      caption      |
> | document classification |      text      | document category |
> |     speech-to-text      | audio waveform |       text        |

#### **Input Vector**

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

#### Decision Boundaries

We can visualize the behavior in the classification setting using a **Voronoi diagram**.

<img src="images\\image-20210906194744348.png" alt="image-20210906194744348" />

<img src="images\image-20210906194912474.png" alt="image-20210906194912474" />

+ Nearest neighbors **sensitive to noise or mis-labeled data** (“class noise”)
+ Smooth by having k nearest neighbors vote

#### k-Nearest Neighbors

<img src="images\image-20210906195111261.png" alt="image-20210906195111261" />

>**Algorithm:**
>
>1. Find $k$ example $(x^{(i)}, t^{(i)}) $ (from the stored training set) closest to the test instance $x$. 
>
>2. Classification output is majority class
> $$
>   y = \underset {t^{(z)}}{\operatorname {arg\,max} }\,\sum_{i=1}^k Ⅱ(t^{(z)}=t^{(i)})
> $$
>
>*$Ⅱ${statement} is the identity function and is equal to one whenever the statement is true. We could also write this as $\delta(t^{(z)},t^{(i)})$, with $\delta(a,b)=1$ if $a=6,0$ otherwise.* 

**$k=1$**

<img src="images\image-20210906200944380.png" alt="image-20210906200944380" style="zoom: 50%;" />

**$k=15$**

<img src="images\image-20210906201006702.png" alt="image-20210906201006702" style="zoom: 50%;" />

**Tradeoffs in choosing $k$?** 

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
+ Number of computations at **test time**，per query (naïve algorithm) 
  + Calculate D-dimensional Euclidean distances with $N$ data points: $O(ND)$ 
  + Sort the distances: $O(N logN)$ 
+ This must be done for each query, which is very expensive by the standards of a learning algorithm! 
+ Need to store the entire dataset in niemorv! 
+ Tons of work lias gone into algorithms and data structures for efficient nearest neighbors with high dimensions and/or large datasets.

#### Example: Digit Classification

+ KNN can perform a lot better with a good similarity measure. 
+  Example: shape contexts for object recognition. In order to achieve invariance to image transformations, they tried to warp one image to match the other image. 
  + Distance measure: average distance between corresponding points on warped images 
+ Achieved $0.63\%$ error on MNIST, compared with $3\%$ for Euclidean KNN. 
+ Competitive with conv nets at the time, but required careful engineering.

<img src="images\image-20210906204157608.png" alt="image-20210906204157608"  />

### Conclusion

+ Simple algorithm that does all its work at test time $--$ in a sense, no learning! 
+ Can control the complexity by varying $k $
+ Suffers from the Curse of Dimensionality 
+ Next time: parametric models, which learn a compact summary of the data rather than referring back to it at test time.

