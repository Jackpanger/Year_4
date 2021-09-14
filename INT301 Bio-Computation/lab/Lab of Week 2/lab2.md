## Lab2

| Matrix | Comment                                         |
| ------ | ----------------------------------------------- |
| p1     | Input vectors for the first problem             |
| t1     | Target outputs (classes) for the first problem  |
| p2     | Input vectors for the second problem            |
| t2     | Target outputs (classes) for the second problem |

>p1, t1: train data for the first problem
>
>p2, t2: training data the second problem
>
>Total number of sample is 100, first 50 samples are labeled as 0, the others are labeled as 1.

### Plot the graph

```matlab
plot (p1(1,1:50), p1(2,1:50), 'b+', p1(1,51:100), p1(2,51:100), 'ro')
```

>**plot(X,Y) plots vector Y versus vector X.**
>
>For this example, p1(1,1:50) is the data of x-axis, p1(2,1:50) is the data of y-axis
>
>samples of p1$--$ each column is a sample x:
>
><img src="..\..\images\image-20210914153344772.png" alt="image-20210914153344772"  />
>
>1. "b+" means how the graph look like, in this case, blue and plus label.
>
>```matrix
>b     blue          .     point              -     solid
>g     green         o     circle             :     dotted
>r     red           x     x-mark             -.    dashdot 
>c     cyan          +     plus               --    dashed   
>m     magenta       *     star             (none)  no line
>y     yellow        s     square
>k     black         d     diamond
>w     white         v     triangle (down)
>                    ^     triangle (up)
>                    <     triangle (left)
>                    >     triangle (right)
>                    p     pentagram
>                    h     hexagram
>```
>
>2. The last three parameters are same as the first three ones, hence, two classes are given:
>
><img src="..\..\images\image-20210914152917594.png" alt="image-20210914152917594" style="zoom:80%;" />



### Exercise 1

#### Create net

```matlab
mm = minmax(p1);
net = newp(mm,1);
```

1. **minmax - Ranges of matrix rows**

   <img src="..\..\images\image-20210914155057367.png" alt="image-20210914155057367" style="zoom: 150%;" />

   >Explanation:
   >
   >lowest x, 	largest x
   >
   >lowest y, 	largest y

2. **newp Create a perceptron.**

   *Syntax*

   ```matlab
    net = newp(p,t,tf,lf)
   ```

   NET = newp(P,T,TF,LF) takes these inputs,

   >P  - RxQ matrix of Q1 representative input vectors.
   >
   >T  - SxQ matrix of Q2 representative target vectors.
   >
   >TF - Transfer function, default = 'hardlim'.
   >
   >LF - Learning function, default = 'learnp'.
   >
   >​	Returns a new perceptron.

   + To be frank, P is a vector representing the dimension and range of the input, like the example before, giving a result of function minmax. T is a matrix representing the nodes of the output.

   + In this case, newp(mm, 1) means the perceptron is in range of mm, and with 1-node output.

#### Train net

1. **trainp_sh**

   *Syntax*

   ```matlab
   net = trainp_sh(net, p1, t1, neps)
   ```

   net = trainp_sh(net, p, t, neps)

   > Trains a perceptron with two inputs and shows the decision boundary after each epoch.
   > Input
   >
   > + net - perceptron
   >
   > + p - Two-dimensional input vectors
   >
   > + t - One dimensional target vectors (0 or 1)
   >
   > + neps - number of epochs to train, optional DEFAULT = 100
   >
   > + Return
   >
   >   net - the trained perceptron

   **Notes:**

   1. This function will also update the graphical display, showing the decision line as well as the training examples.

   **Code analysis**

   ```matlab
   function net = trainp_sh(net, p, t, neps)
   % net = trainp_sh(net, p, t, neps)
   % Trains a perceptron with two inputs and shows the decision
   % boundary after each epoch.
   % Input
   %  net  - perceptron
   %  p    - Two-dimensional input vectors
   %  t    - One dimensional target vectors (0 or 1)
   %  neps - number of epochs to train, optional DEFAULT = 100
   % Return
   %  net - the trained perceptron
   
   % nargin returns the number of arguments input when the function is called.
   % which means: the number of arguments is 4, if "neps" is ignored, the n it will be assigned to 100 by default.
   if nargin<4 
     neps = 100;
   end
   
   % Plot perceptron input/target vectors
   plotpv(p, t);
   % Plot classification line on perceptron vector plot
   h = plotpc(net.IW{1}, net.b{1});
   e = 1; % error
   ep = 0; % epoch
   while sum(abs(e))>0 & ep<neps, % while error still exists and the number of epochs not reached
   
   % adapt: Adapt neural network to data as it is simulated
   % net：Updated network
   % y: Network outputs
   % e: Network errors
   % pf: Final input delay conditions
   % af: Final layer delay conditions
   % ar: Training record (epoch and perf)
     [net, y, e, pf, af, ar] = adapt(net, p, t); % update the weights of the net and output prediction, error
     h = plotpc(net.IW{1},net.b{1}, h);
     drawnow; % redraw the plot
     ep = ep + 1; % epoch+=1
   end
   ```

#### Codes for the problems:

##### Problem 1

```matlab
plot (p1(1,1:50), p1(2,1:50), 'b+', p1(1,51:100), p1(2,51:100), 'ro')
mm = minmax(p1);
net = newp(mm,1);
net = trainp_sh(net, p1, t1, 50) % easy to stop, linear data
```

##### Problem 2

```matlab
plot (p2(1,1:50), p2(2,1:50), 'b+', p2(1,51:100), p2(2,51:100), 'ro')
mm = minmax(p1);
net = newp(mm,1);
net = trainp_sh(net, p1, t1, 50) % 500+ is ok for epochs
```

#### Consider the following question (for both data sets):  

1. Is the perceptron able to classify all training vectors correctly after training? If not, why?

   Reference answer from website

   No. Cons. below:

   + Cannot perfectly handle linearly indistinguishable training data
   + The final number of iterations is strongly influenced by the resulting hyperplane and the data in the training set
   + The goal of the loss function is only to reduce the distance between all misclassified points and hyperplanes, which will likely result in some sample points being very close to the hyperplane, which in a way is not particularly good for classification, a problem that will be well solved in the support vector machine.

### Exercises 2

***Data have been stored in the file "exercise.mat": train, label and test.***

"exercise_sh" is modified from "trainp_sh" :

+ Plot functions are commented out, because high-dimensional data cannot be visualized. Only support 1-3 dimension.

#### **Reference code**

```matlab
mm = minmax(train);
net = newp(mm,1);
# mod: acquire reminder
net = exercise_sh(net, train, mod(label+1,2), 10);
net(test)

% ans =
     0     1
```



