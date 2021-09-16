# Big Data Analytics

> Project 1: Data Scraping
>
> + 15%
> + Personal project
> + Release on week 4
>
> Project 2: Big Data Competition
>
> + Titanic dataset
> + 15%
> + Personal project
> + Kaggle
> + Release on week 10
>
> Written Exam
>
> + 70%
> + TBD

## Lecture 1

### Nothing important

*No need for watching recording or PPT for this week*

## Lecture 2

### THE DATA SCIENCE PROCESS

**Recall the data science process.**

+ Ask questions 
+ Data Collection 
+ Data Exploration 
+ Data Modeling 
+ Data Analysis 
+ Visualization and Presentation of Results

<img src="images\image-20210916110810202.png" alt="image-20210916110810202" style="zoom:80%;" />

### Data Collection 

#### WHAT ARE DATA?

"A datum is a single measurement of something on a scale that is  understandable to both the recorder and the reader. Data are multiple  such measurements."

Claim: everything is (can be) data!

#### WHERE DO DATA COME FROM?

+ Internal sources: already collected by or is part of the overall data  collection of you organization. For example: business-centric data that is available in the organization data base to record day to day operations; scientific or experimental  data 
+ Existing External Sources: available in ready to read format from an  outside source for free or for a fee. For example: public government databases, stock market data, Yelp  reviews, [your favorite sport]-reference 
+ External Sources Requiring Collection Efforts: available from external  source but acquisition requires special processing. For example: data appearing only in print form, or data on websites

#### WAYS TO GATHER ONLINE DATA

How to get data generated, published or hosted online: 

+ **API (Application Programming Interface):** using a prebuilt set of  functions developed by a company to access their services. Often pay to use. For example: Google Map API, Facebook API, Twitter  API 
+ **RSS (Rich Site Summary):** summarizes frequently updated online  content in standard format. Free to read if the site has one. For  example: news-related sites, blogs 
+ **Web scraping:** using software, scripts or by-hand extracting data  from what is displayed on a page or what is contained in the HTML  file.

#### WEB SCRAPING

+ Why do it? Older government or smaller news sites might not have APIs  for accessing data, or publish RSS feeds or have databases for  download. Or, you don’t want to pay to use the API or the database.
+ How do you do it? See HW1 
+ Should you do it? 
  + You just want to explore: Are you violating their terms of service?  Privacy concerns for website and their clients? 
  + You want to publish your analysis or product: Do they have an API or  fee that you are bypassing? Are they willing to share this data? Are  you violating their terms of service? Are there privacy concerns?

#### TYPES OF DATA

What kind of values are in your data (data types)? 

Simple or atomic: 

+ **Numeric:** integers, floats 
+ **Boolean:** binary or true false values 
+ **Strings:** sequence of symbols

Compound, composed of a bunch of atomic types: 

+ **Date and time:** compound value with a specific structure 
+ **Lists:** a list is a sequence of values 
+ **Dictionaries:** A dictionary is a collection of key-value pairs, a  pair of values x : y where x is usually a string called the key  representing the “name” of the entry, and y is a value of any type. 

Example: Student record: what are x and y? 

+ First: Kevin 
+ Last: Rader 
+ Classes: [CS-109A, STAT139]

#### DATA FORMAT

How is your data represented and stored (data format)? 

+ Textual Data 
+ Temporal Data
+ Geolocation Data

#### DATA STORAGE

How is your data represented and stored (data format)? 

+ **Tabular Data:** a dataset that is a two-dimensional table, where  each row typically represents a single data record, and each  column represents one type of measurement (csv, dat, xlsx, etc.). 
+ **Structured Data:** each data record is presented in a form of a  [possibly complex and multi-tiered] dictionary (json, xml, etc.) 
+ **Semistructured Data:** not all records are represented by the same  set of keys or some data records are not represented using the  key-value pair structure.

##### TABULAR DATA

In tabular data, we expect each record or observation to represent a set of  measurements of a single object or event. We’ve seen this already in Lecture 0:

<img src="images\image-20210916113311742.png" alt="image-20210916113311742"  />

Each type of measurement is called a **variable** or an **attribute** of the data (e.g. *seq_id*, *status* and *duration* are variables or attributes). The number of attributes is called the **dimension**. These are often called **features**.    

<span style="color:rgb(130,50,150)">***PN: title somestimes !=  features, it depends on the clean or dirty data.***</span>

We expect each table to contain a set of **records** or **observations** of the same  kind of object or event (e.g. our table above contains observations of  rides/checkouts).

#### TYPES OF DATA

We’ll see later that it’s important to distinguish between classes of  variables or attributes based on the type of values they can take on. 

+ **Quantitative variable:** is numerical and can be either: 
  + **discrete** - a finite number of values are possible in any  bounded interval. For example: "Number of siblings" is a  discrete variable 
  + **continuous** - an infinite number of values are possible in any  bounded interval. For example: "Height" is a continuous variable 
+ **Categorical variable:** no inherent order among the values For example:  "What kind of pet you have" is a categorical variable

##### QUANTITATIVE VARIABLE (1)

<img src="images\image-20210916180810533.png" alt="image-20210916180810533" style="zoom: 67%;" />

##### QUANTITATIVE VARIABLE (2)

<img src="images\image-20210916180911535.png" alt="image-20210916180911535" style="zoom:67%;" />

##### QUANTITATIVE VARIABLE (3)

<img src="images\image-20210916180941915.png" alt="image-20210916180941915" style="zoom:67%;" />

##### CATEGORICAL VARIABLE

<img src="images\image-20210916181009312.png" alt="image-20210916181009312" style="zoom:67%;" />

### COMMON ISSUES

Common issues with data: 

+ Missing values: how do we fill in? 
+ Wrong values: how can we detect and correct? 
+ Messy format 
+ Not usable: the data cannot answer the question posed

#### MESSY DATA

The following is a table accounting for the number of produce  deliveries over a weekend. 

What are the variables in this dataset? What object or event are we  measuring?

<img src="images\image-20210916114424798.png" alt="image-20210916114424798" style="zoom:80%;" />

What’s the issue? How do we fix it?

We’re measuring individual deliveries; the variables are Time, Day,  Number of Produce.

Problem: each column header represents a single value rather than a  variable. Row headers are "hiding" the Day variable. The values of the  variable, "Number of Produce", is not recorded in a single column.

#### FIXING MESSY DATA

We need to reorganize the information to make explicit the event  we’re observing and the variables associated to this event.

<img src="images\image-20210916114724997.png" alt="image-20210916114724997"  />

#### MORE MESSINESS

What object or event are we measuring? 

What are the variables in this dataset? 

How do we fix?

<img src="images\image-20210916114752288.png" alt="image-20210916114752288"  />

We’re measuring individual deliveries; the variables are Time, Day,  Number of Produce:

<img src="C:\Users\ADMIN\AppData\Roaming\Typora\typora-user-images\image-20210916114849631.png" alt="image-20210916114849631"  />

#### FIXING MESSY DATA

We need to reorganize the information to make explicit the event  we’re observing and the variables associated to this event.

<img src="images\image-20210916114947230.png" alt="image-20210916114947230"  />

#### TABULAR = HAPPY PAVLOS

Common causes of messiness are: 

+ Column headers are values, not variable names 

+ Variables are stored in both rows and columns 

+ Multiple variables are stored in one column/entry 

+ Multiple types of experimental units stored in same table In general, we want each file to correspond to a dataset, each  column to represent a single variable and each row to represent a  single observa9on. 

  We want to tabularize the data. This makes Python happy.

##### EXAMPLE

<img src="images\image-20210916115133861.png" alt="image-20210916115133861" style="zoom:80%;" />

### DATA EXPLORATION: DESCRIPTIVE STATISTICS

#### BASICS OF SAMPLING

Population versus sample: 

+ A **population** is the entire set of objects or events under study.  Population can be hypothetical “all students” or all students in this  class. 
+ A **sample** is a "representative" subset of the objects or events under  study. Needed because it’s impossible or intractable to obtain or  compute with population data. Biases in samples: 
+ **Selection bias**: some subjects or records are more likely to be selected 
+ Volunteer/**nonresponse bias**: subjects or records who are not easily  available are not represented Examples?

#### SAMPLE MEAN

The **mean** of a set of n observations of a variable is denoted and  is defined as:
$$
\begin{align*}
\overline{x} = \frac{x_1+x_2+...+x_n}{n} = \frac{1}{n}\sum_{i=1}^nx_i
\end{align*}
$$
<img src="images\image-20210916115806494.png" alt="image-20210916115806494" style="zoom:80%;" />

The mean describes what a "typical" sample value looks like, or where is  the "center" of the distribution of the data.

Key theme: there is always uncertainty involved when calculating a sample mean to estimate a population mean.

#### SAMPLE MEDIAN

The **median** of a set of n number of observations in a sample, ordered by  value, of a variable is is defined by
$$
\begin{align*}
Median = 
\begin{cases}
x_{(n+1)/2}\quad &\text{if n is odd}\\
\frac{x_{n/2}+x_{(n+1)/2}}{2}\quad &\text{if n is even}
\end{cases}
\end{align*}
$$
Example (already in order): 

> Ages: 17, 19, 21, 22, 23, 23, 23, 38 
>
> Median = (22+23)/2 = 22.5 

The median also describes what a typical observation looks like, or where is  the center of the distribution of the sample of observations.

#### MEAN VS. MEDIAN

The mean is sensitive to extreme values (**outliers**)

<img src="images\image-20210916120455514.png" alt="image-20210916120455514" style="zoom:80%;" />

#### MEAN, MEDIAN, AND SKEWNESS

The mean is sensitive to outliers.

<img src="images\image-20210916120840697.png" alt="image-20210916120840697"  />

The above distribution is called **right-skewed** since the mean is greater  than the median. Note: **skewness** often "follows the longer tail".

#### REGARDING CATEGORICAL VARIABLES...

For categorical variables, neither mean or median make sense. Why?

<img src="images\image-20210916121005716.png" alt="image-20210916121005716" style="zoom:80%;" />

The mode might be a better way to find the most "representative" value

#### MEASURES OF SPREAD

##### MEASURES OF SPREAD: RANGE

The spread of a sample of observations measures how well the mean  or median describes the sample. One way to measure spread of a sample of observations is via the **range**. 

*Range = Maximum Value - Minimum Value*

##### MEASURES OF SPREAD: VARIANCE

The (sample) **variance**, denoted s2 , measures how much on average  the sample values deviate from the mean:
$$
\begin{align*}
s^2 = \frac{1}{n-1}\sum_{i=1}^n|x_i-\overline{x}|^2
\end{align*}
$$
Note: the term $|x_i-\overline{x}|$ measures the amount by which each $x_i$ deviates from the mean  $\overline{x}$. Squaring these deviations means that $s_2$ is  sensitive to extreme values (outliers). 

Note: $s_2$ doesn’t have the same units as the $x_i$ :(  

What does a variance of 1,008 mean? Or 0.0001?

##### MEASURES OF SPREAD: STANDARD DEVIATION

The (sample) **standard deviation**, denoted s, is the square root of  the variance
$$
\begin{align*}
s = \sqrt{s^2} = \sqrt{\frac{1}{n-1}\sum_{i=1}^n|x_i-\overline{x}|^2}
\end{align*}
$$
