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

## Lecture 3

### Content

>TBD
>
>

### The Basic EDA Workflow

1. Build a DataFrame from the data (ideally, put all data in this object) 

2. Clean the DataFrame. It should have the following properties: 

   – Each row describes a single object

   – Each column describes a property of that object 

   – Columns are numeric whenever appropriate 

   – Columns contain atomic properties that cannot be further decomposed 

3. Explore global properties. Use histograms, scatter plots, and aggregation functions to summarize the data. 

4. Explore group properties. Use groupby, queries, and small multiples to compare subsets of the data.

#### BUILDING A DATAFRAME

+ The easiest way to build a dataframe is simply to read in a CSV file. 
+ We WILL see an example of this here, and we shall see more examples in labs.
+ We'll also see how we may combine multiple data sources into a larger dataframe.

#### CLEANING DATA

+ Dealing with missing values 
+ Transforming types appropriately 
+ Taking care of data integrity

##### WHY DATA CLEANING IS ESSENTIAL?

1. Error-Free Data 
2. Data Quality 
3. Accurate and Efficient Data 
4. Complete Data 
5. Maintains Data Consistency

<img src="images\image-20210923114046415.png" alt="image-20210923114046415" style="zoom: 50%;" />

##### DATA CLEANING CYCLE5

<img src="images\image-20210923114117396.png" alt="image-20210923114117396" style="zoom:50%;" />

###### IMPORT DATASET

<img src="images\image-20210923114222432.png" alt="image-20210923114222432" style="zoom: 67%;" />

**DISPLAY FIRST FIVE ROWS OF DATASET**

<img src="images\image-20210923114243221.png" alt="image-20210923114243221" style="zoom: 67%;" />

###### MERGE DATASET

+ Merging the dataset is the process of combining two datasets in one.

  <img src="images\image-20210923114324243.png" alt="image-20210923114324243" style="zoom: 67%;" />

  link: [https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html)

###### REBUILD MISSING DATA

+ To find and fill the missing data in the dataset we will use another function. 

+ **Using isnull() /isna() function:**

  <img src="images\image-20210923114441888.png" alt="image-20210923114441888" style="zoom:67%;" />

+ **Using isna(). sum()**

  <img src="images\image-20210923114508838.png" alt="image-20210923114508838" style="zoom:67%;" />

+ **De-Duplicate** 

+ De-Duplicate means remove all duplicate values data.duplicated()

  <img src="images\image-20210923114819036.png" alt="image-20210923114819036" style="zoom:67%;" />

+ **DataFrame.fillna()** 

+ Fill NA/NaN values using the specified method.

  <img src="images\image-20210923114901555.png" alt="image-20210923114901555" style="zoom:67%;" />

  link: [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html)

+ If a dataset contains duplicate values it can be removed using the drop_duplicates() function.

  <img src="images\image-20210923114957870.png" alt="image-20210923114957870" style="zoom:67%;" />

  link: [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html)

###### STANDARDIZATION AND NORMALIZATION

<img src="images\image-20210923115306712.png" alt="image-20210923115306712" style="zoom:67%;" />

Case A (correct) : 

1. Import Data 
2. Split Data into training and test sets 
3. Scale the training and test sets together using MinMaxScaler. 
4. Visualization

Case B: 

1. Import Data 
2. Split Data into training and test sets 
3. Scale training set using MinMaxScaler. 
4. Rescale the test set separately using MinMaxScaler. 
5. Visualization

<img src="images\image-20210923122948520.png" alt="image-20210923122948520"  />

##### VERIFY AND ENRICH

+ We should verify the dataset and validate its accuracy. 
+ We have to check that the data cleaned so far is making any sense. 
+ If the data is incomplete we have to enrich the data. 
+ approaching the clients again, re-interviewing people, etc.

**DATA TRANSFORMATION**

+ Query 
+ Sort 
+ Select Columns
+ Select Distinct 
+ Assign
+ Group by 
+ Joint

### Grammar of Data

#### PANDAS

+ Pandas is well suited for many different kinds of data: 
+ Tabular data with heterogeneously-typed columns, as in an SQL table or Excel spreadsheet 
+ Ordered and unordered (not necessarily fixed-frequency) time series data. 
+ Arbitrary matrix data with row and column labels 
+ Any other form of observational / statistical data sets.

<img src="images\image-20210923124307637.png" alt="image-20210923124307637" style="zoom:67%;" />

<img src="images\image-20210923124326974.png" alt="image-20210923124326974" style="zoom:67%;" />

#### GRAMMAR OF DATA

+ [If you need to find a Data Related work!!](https://www.w3resource.com/python-exercises/pandas/index.php)

+ Why bother? 
+ learn how to do core data manipulations, no matter what the system is. 
+ one off questions: google, stack-overflow, http://chrisalbon.com

##### BEST PRACTICE

+ Go to notebook: grammarofdata.ipynb

##### HOW TO CREATE A SERIES FROM A LIST, NUMPY ARRAY AND DICT?

<img src="images\image-20210923124555398.png" alt="image-20210923124555398" style="zoom:67%;" />

##### HOW TO COMBINE MANY SERIES TO FORM A DATAFRAME?

<img src="images\image-20210923124721982.png" alt="image-20210923124721982" style="zoom:67%;" />

##### HOW TO GET USEFUL INFOS

<img src="images\image-20210923124737635.png" alt="image-20210923124737635" style="zoom: 80%;" />

##### GROUPBY

+ Example: Candidates

  <img src="images\image-20210923124806727.png" alt="image-20210923124806727" style="zoom:67%;" />

+ Contributors

  <img src="images\image-20210923124856354.png" alt="image-20210923124856354" style="zoom:80%;" />

+ Groupby: 

  + Splitting the data into groups based on some criteria 
  + Applying a function to each group independently 
  + Combining the results into a data structure

  <img src="images\image-20210923124929496.png" alt="image-20210923124929496"  />

##### MERGE

+ Merge: 
+ Combine tables on a common key-value

<img src="images\image-20210923125017260.png" alt="image-20210923125017260"  />

### QUESTION 

#### 1. MAPPING

***Answer: B C A***

<img src="images\image-20210923125039313.png" alt="image-20210923125039313" style="zoom:80%;" />

#### 2. How many of the given statements are the correct reason why data cleansing is critical： 

Error-Free Data 

Data Quality 

Accurate and Efficient Data 

Complete Data 

Maintains Data Consistency

#### 3. FILL IN A BLANK

+ How to get useful infos of a Dataframe SER 
+ SER. describe()

<img src="images\image-20210923125355914.png" alt="image-20210923125355914" style="zoom:80%;" />

## Lecture 4

### Content

>TBD

### OUTLINE

- What is Web Service?
- Data Scraping
- Gathering data from APIs

<img src="images\image-20210930113018261.png" alt="image-20210930113018261" style="zoom: 50%;" />

<img src="images\image-20210930113046601.png" alt="image-20210930113046601" style="zoom: 50%;" />

### Web Servers

- A server is a long running process (also called daemon) which listens on a prespecified port
- and responds to a request, which is sent using a protocol called HTTP
- A browser parses the url.

#### HOW IT WORKS 

<img src="images\image-20210930113136912.png" alt="image-20210930113136912" style="zoom: 67%;" />

<img src="images\image-20210930113154225.png" alt="image-20210930113154225" style="zoom: 67%;" />

<img src="images\image-20210930113219885.png" alt="image-20210930113219885" style="zoom: 67%;" />

##### Example:

- Our notebooks also talk to a local web server on our machines:
http://localhost:8888/Documents/cs109/BLA.ipynb#something
- protocol is http, hostname is localhost, port is 8888
- url is / Documents/cs109/BLA.ipynb
- url fragment is #something
- Request is sent to localhost on port 8888 . It says:
- Request: GET /request-URI HTTP/version

<img src="images\image-20210930113309432.png" alt="image-20210930113309432" style="zoom:80%;" />

#### HTTP STATUS CODES

<img src="images\image-20210930113335917.png" alt="image-20210930113335917" style="zoom:80%;" />

#### WEB SERVERS

- Requests:

  - great module built into python for http requests

  - ```python
    req=requests.get(“https://en.wikipedia.org/wiki/Harvard_University”)
    ```

  - <Response [200]>

  - page = req.text

  - ```python
    '<!DOCTYPE html>\n<html class="client-nojs" lang="en" dir="ltr">\n<head>\n<meta
    charset="UTF-8"/>\n<title>Harvard University -
    Wikipedia</title>\n<script>document.documentElement.className=document.documentElement.cl
    assName.replace( /(^|\\s)client-nojs(\\s|$)/,"$1client-js$2");</script>\n<script>(window.RLQ=window.RLQ||[]).push(function(){mw.config.set({"wgCanonicalNamespace":"","wgCanonicalSpecialPageName":false,"wgNamespaceNumber":0,"wgPageName":"Harvard_University","wgTitle":"Harva...'
    ```

<img src="images\image-20210930113544429.png" alt="image-20210930113544429" style="zoom:80%;" />

### Python data scraping

#### Why scrape the web?

- vast source of information, combine with other data sets
- companies have not provided APIs
- automate tasks
- keep up with sites
- fun!

#### CHALLENGES IN WEB SCRAPING

- Which data?
  - It is not always easy to know which site to scrape
  - Which data is relevant, up to date, reliable?
- The internet is dynamic
  - Each web site has a particular structure, which may be changed anytime
- Data is volatile
  - Be aware of changing data patterns over time

#### LEGAL

- Privacy:
  - Legislation on protection of personal information
  - At this moment we only scrape public sources
- Netiquette (practical):
  - respect the **Robots Exclusion Protocol** also known as the robots.txt (example)
  - identify yourself (user-agent)
  - do not overload servers, use some idle time between requests, run crawlers at night / morning
  - Inform website owners if feasible

#### NOTICE

- copyrights and permission:
  - be careful and polite
  - give credit
  - care about media law
  - don't be evil (no spam, overloading sites, etc.)

#### ROBOTS.TXT

- specified by web site owner
- gives instructions to web robots (aka your script)
- is located at the top-level directory of the web server
- e.g.: http://google.com/robots.txt

#### HTML

- angle brackets
- should be in pairs, eg \<p> Hello\</p>
- maybe in implicit bears, such as \<br/>

```html
<!DOCTYPE html>
<html>
	<head>
		<title>Title</title>
	</head>
	<body>
		<h1>Body Title</h1>
		<p >Body Content</p>
	</body>
</html>
```

#### DEVELOPER TOOLS

- ctrl/cmd shift- i in chrome
- cmd-option-i in safari
- look for "inspect element"
- locate details of tags

#### BEAUTIFUL SOUP

+ will normalize dirty html 
+ basic usage

```python
import bs4
# get bs4 object
soup = bs4.BeautifulSoup(source)
## all a tags
soup.findAll('a')
## first a 
soup.find('a')
# get all links in the page
link_list = [l.get('href') for l in soup.findAll('a')]
```

##### HTML IS A TREE

```python
tree = bs4.BeautifulSoup(source)
## get html root node
root_node = tree.html
## get head from root using contents
head = root_node.contents[0]
## get body from root
body = root_node.contents[1]
## could directly access body
tree.body
```

##### DEMOGRAPHICS TABLE WE WANT

<img src="images\image-20210930114616912.png" alt="image-20210930114616912" style="zoom:80%;" />

##### TABLE WITH SOLE CLASS WIKITABLE

<img src="images\image-20210930114641648.png" alt="image-20210930114641648" style="zoom:80%;" />

##### BEAUTIFUL SOUP CODE

```python
dfinder = lambda tag:tag.name=='table' and tag.get('class')==['wikitable']
table_demographics = soup.find_all(dfinder)
rows = [row for row in table_demographics[0].find_all("tr")]
header_row = rows[0]
columns = [col.get_text() for col in header_row.find_all("th") if col.get_text()]
columns = [rem_nl(c) for c in columns]
indexes = [row.find("th").get_text() for row in rows[1:]]
values = []
for row in rows[1:]:
    for value in row.find_all("td"):
        values.append(to_num(value.get_text()))
stacked_values_lists = [values[i::3] for i in range(len(columns))]
stacked_values_iterator = zip(*stacked_values_lists)
df = pd.DataFrame(list(stacked_values_iterator),columns=columns,index=indexes)
```

#### PROJECT EXAMPLE

- https://github.com/alirezamika/autoscraper
- https://github.com/scrapy/scrapy
- https://yasoob.me/posts/github-actions-web-scraperschedule-tutorial/

### Gathering data from APIs

### API

- API = **A**pplication **P**rogram **I**nterface
- Many data sources have API's - largely for talking to other web interfaces
- Consists of a set of methods to search, retrieve, or submit data to, a data source
- We can write R code to interface with an API (lot's require authentication though)
- Many packages already connect to well-known API's (we'll look at a couple today)

#### PUBLIC API

https://any-api.com

<img src="images\image-20210930115757513.png" alt="image-20210930115757513" style="zoom:80%;" />
