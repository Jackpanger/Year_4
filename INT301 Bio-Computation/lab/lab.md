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
>                     ^     triangle (up)
>                     <     triangle (left)
>                     >     triangle (right)
>                     p     pentagram
>                     h     hexagram
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

## Lab 4

#### Demo

##### PerceptronExample()

```matlab
function PerceptronExample 
% Rosenblatt's Perecptron
% References :
% Neural Networks for Pattern Recognition By C. Bishop

% clear the screen
clc
%============================================
% Generate 2 dimensions linear separable data 
%===========================================

% You may change the size of the data from here or input your own data
%   note that the drawing is for two dimensions only, hence you need to 
%   modify the code for different data.
mydata = rand(500,2);  % random generate (500,2) data
% Separate the data into two classes, acquire data with distance larger than 0.012 between the first column and the second column.
acceptindex = abs(mydata(:,1)-mydata(:,2))>0.012;
mydata = mydata(acceptindex,:); % data
myclasses = mydata(:,1)>mydata(:,2); % labels
[m n]=size(mydata);

% The next two lines divide the data into training and testing parts
%training data: first 400 rows
x=mydata(1:400,:);  y=myclasses(1:400);
% test data : the rest of the rows
xt=mydata(401:m,:); yt=myclasses(401:m);
%=====================================
% Train the perceptron
%=====================================
% PerecptronTrn function will be analyzed below
[w,b,pass] = PerecptronTrn(x,y); 
Iterations=pass

%=====================================
% Test
%=====================================
% PerecptronTst function will be analyzed below
e=PerecptronTst(xt,yt,w,b);
% num2str() transform number to string
disp(['Test_Errors=' num2str(e) '     Test Data Size= ' num2str(m-400)])

%=====================================
% Draw the result (sparating hyperplane)
%=====================================
 l=y;
 figure;
 hold on % keep the plot
 plot(x(l,1),x(l,2),'k.' );
 plot(x(~l,1),x(~l,2),'b.');
 plot([0,1],[0,1],'r-')
 axis([0 1 0 1]), axis square, grid on
 drawnow
```

##### PerecptronTrn()

```matlab
function [w,b,pass]=PerecptronTrn(x,y)
% %Rosenblatt's Perecptron
tic % combine with toc, measure the time of the process 
[l,p]=size(x);
w=zeros(p,1); % initialize weights with all zero.
b=0;          % initialize bias
ier=1;        % initialize a misclassification indicator
pass=0;       % number of iterations
n=0.5;        % learning rate
r=max(sqrt(sum(x))); % max norm
iter = 0;     % iteration index
while ier==1 %repeat until no error
       ier=0; iter = iter + 1;
       e=0; % number of training errors
       for i=1:l  % a pass through x           
           xx=x(i,:); % current data xx
           ey=xx*w+b; % estimated y
           if ey>=0.5 % threshold (ey=1 if ey>=0.5 else 0)
              ey=1;
           else
              ey=0;
           end
           if y(i)~=ey % if y not equals ey
              er=y(i)-ey;      % error difference
              w=w'+(er*n)*x(i,:);  % can be written as w = w'+(er*n)*xx;    
              e=e+1 ;         % number of training errors
              w=w';   % 'means transpose
           end
       end
       e_list(iter)=e; % a list to record each error
       ee=e;    % number of training errors
       if ee>0  % continue if there is still errors
          ier=1;           
       end
       pass=pass+1; % stop after 10000 iterations
       if pass==10000
          ier=0;
          pass=0;
       end
end

figure; % create a new window to draw someting;
plot([0:length(e_list)-1], e_list, '-ko' , 'LineWidth', 0.1); % plot the error during the training process
xlabel('iteration')
ylabel('e-training')

disp(['Training_Errors=' num2str(e) '     Training data Size=' num2str(l)])
toc % measure the time
```

##### PerecptronTst()

```matlab
function e=PerecptronTst(x,y,w,b)
%==========================================
% Testing phase
%==========================================
tic
[l,p]=size(x);
e=0; % number of test errors
for i=1:l          
    xx=x(i,:); % take one row
    ey=xx*w+b; % apply the perceptron classification rule
    if ey>=0.5 
       ey=1;
    else
       ey=0;
    end
    if y(i)~=ey
       e=e+1;
    end
end
toc
```

### Exercise 

#### Exercise 1

>Related files: PerceptronExercise1.m, PerecptronTrnExercise1.m, PerecptronTstExercise1.m
>
>Usage:
>
>```matlab
>PerceptronExercise1();
>```

#### Exercise 2

> Related files: PerceptronExercise2.m, PerecptronTrnExercise2.m, PerecptronTstExercise2.m
>
> Usage:
>
> ```matlab
> PerceptronExercise2();
> ```
