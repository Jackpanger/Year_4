# Big Data Analytics 5-8

## Lecture 5

### Content

>TBD

### Outline

- Visualization motivation
- Principle of Visualization
- Types of Visualization
- Example

### Anscombe’s Data

The following four data sets comprise the Anscombe’s Quartet; all four sets of data have identical simple summary statistics.

<img src="images\image-20211025190319259.png" alt="image-20211025190319259" style="zoom:80%;" />

Summary statistics clearly don’t tell the story of how they differ. But a picture can be worth a thousand words:

<img src="images\image-20211025190343502.png" alt="image-20211025190343502" style="zoom:80%;" />

### VISUALIZATION MOTIVATION

If I tell you that the average score for Homework 0 was: 7.64/15 = 50.9% last year, what does that suggest?

<img src="images\image-20211025190430669.png" alt="image-20211025190430669" style="zoom:80%;" />

**And what does the graph suggest?**

Visualizations help us to analyze and explore the data. They help to:
- Identify hidden patterns and trends
- Formulate/test hypotheses
- Communicate any modeling results
- Present information and ideas succinctly
- Provide evidence and support
- Influence and persuade
- Determine the next step in analysis/modeling



### PRINCIPLES OF VISUALIZATIONS

Some basic data visualization guidelines from Edward Tufte:

1. Maximize data to ink ratio: show the data.

   Which is better?

   <span style="color:rgb(130,50,150)">***PN: The right one is better, because the underline of the left figure will distract the attention of audience.***</span>

   <img src="images\image-20211025190638605.png" alt="image-20211025190638605" style="zoom:80%;" />

2. Don’t lie with scale (Lie Factor)

   <img src="images\image-20211025191002843.png" alt="image-20211025191002843" style="zoom:80%;" />

3. Minimize chart-junk: show data variation, not design variation

   <img src="images\image-20211025214242738.png" alt="image-20211025214242738" style="zoom:80%;" />

4. Clear, detailed and thorough labeling

### TYPES OF VISUALIZATIONS

What do you want your visualization to show about your data?
- **Distribution**: how a variable or variables in the dataset distribute over a range of possible values.
- **Relationship**: how the values of multiple variables in the dataset relate
- **Composition**: how a part of your data compares to the whole.
- **Comparison**: how trends in multiple variable or datasets compare

<img src="images\image-20211025214423593.png" alt="image-20211025214423593"  />

#### DISTRIBUTION

- When studying how quantitative values are located along an axis, distribution charts are the way to go.
- By looking at the shape of the data, the user can identify features such as value range, central tendency and outliers.

##### HISTOGRAMS TO VISUALIZE DISTRIBUTION

A **histogram** is a way to visualize how 1-dimensional data is distributed across certain values.

<img src="images\image-20211025214545300.png" alt="image-20211025214545300" style="zoom:80%;" />

Note: Trends in histograms are sensitive to number of bins.

##### SCATTER PLOTS TO VISUALIZE RELATIONSHIPS

A **scatter plot** is a way to visualize how multi-dimensional data are distributed across certain values. A scatter plot is also a way to visualize the relationship between two different attributes of multi-dimensional data.

<img src="images\image-20211025214718419.png" alt="image-20211025214718419" style="zoom:80%;" />

#### RELATIONSHIP

- They are used to find correlations, outliers, and clusters in your data.
- While the human eye can only appreciate three dimensions together, you can visualize additional variables by mapping them to the size, color or shape of your data points.

For 3D data, color coding a categorical attribute can be “effective”

<img src="images\image-20211025220421354.png" alt="image-20211025220421354" style="zoom:80%;" />

**Except when it’s not effective. What could be a better choice?**

Relationships may be easier to spot by producing multiple plots of lower dimensionality.

<img src="images\image-20211025220728686.png" alt="image-20211025220728686" style="zoom:80%;" />

##### 3D CAN WORK

For 3D data, a quantitative attribute can be encoded by size in a bubble chart.

<img src="images\image-20211025220754109.png" alt="image-20211025220754109" style="zoom:80%;" />

The above visualizes a set of consumer products. The variables are: revenue, consumer rating, product type and product cost.

#### COMPARISON

- These are used to compare the magnitude of values to each other and to easily identify the lowest or highest values in the data.
- If you want to compare values over time, line or bar charts are often the best option.
  - Bar or column charts $\rightarrow$ Comparisons among items,,
  - Line charts $\rightarrow$ A sense of continuity.
  - Pie charts for comparison as well

##### MULTIPLE HISTOGRAMS

Plotting **multiple histograms** (and **kernel density estimates** of the distribution, here) on the same axes is a way to visualize how different variables compare (or how a variable differs over specific groups).

<img src="images\image-20211025221001693.png" alt="image-20211025221001693"  />

##### BOXPLOTS

A **boxplot** is a simplified visualization to compare a quantitative variable across groups. It highlights the range, quartiles, median and any outliers present in a data set.

<img src="images\image-20211025221033935.png" alt="image-20211025221033935"  />

#### COMPOSITION

- Composition charts are used to see how a part of your data compares to the whole.
- Show relative and absolute values.
- They can be used to accurately represent both static and time-series data.

##### PIE CHART FOR A CATEGORICAL VARIABLE?

A **pie chart** is a way to visualize the static composition (aka, distribution) of a variable (or single group).

<img src="images\image-20211025221120826.png" alt="image-20211025221120826" style="zoom:80%;" />

Pie charts are often frowned upon (and bar charts are used instead). Why?

##### STACKED AREA GRAPH TO SHOW TREND OVER TIME

A **stacked area graph** is a way to visualize the composition of a group as it changes over time (or some other quantitative variable). This shows the relationship of a categorical variable (AgeGroup) to a quantitative variable (year).

<img src="images\image-20211025221150441.png" alt="image-20211025221150441"  />

### [NOT] ANYTHING IS POSSIBLE!

Often your dataset seem too complex to visualize:
- Data is too high dimensional (how do you plot 100 variables on the same set of axes?)
- Some variables are categorical (how do you plot values (like Cat or No?)

<img src="images\image-20211025221217665.png" alt="image-20211025221217665"  />

### More dimensions not always better

When the data is high dimensional, a scatter plot of all data attributes can be impossible or unhelpful

<img src="images\image-20211025221242500.png" alt="image-20211025221242500"  />

### An Example

Use some simple visualizations to explore the following dataset:

<img src="images\image-20211025221310321.png" alt="image-20211025221310321"  />

**How should we begin?**

A bar graph showing resistance of each bacteria to each drug:

<img src="images\image-20211025221333853.png" alt="image-20211025221333853"  />

What do you notice?

Bar graph showing resistance of each bacteria to each drug (grouped by Group Number):

<img src="images\image-20211025221410047.png" alt="image-20211025221410047"  />

Now what do you notice?

Scatter plot of Drug #1 vs Drug #3 resistance:

<img src="images\image-20211025221544892.png" alt="image-20211025221544892"  />

Key: the process of data exploration is iterative (visualize for trends, re- visualize to confirm)!

##### Q1: WHAT ARE SOME IMPORTANT FEATURES OF A GOOD DATA VISUALIZATION?

The data visualization should be light and must highlight essential aspects of the data; looking at important variables, what is relatively important, what are the trends and changes. Besides, data visualization must be visually appealing but should not have unnecessary information in it.

One can answer this question in multiple ways: from technical points to mentioning key aspects, but be sure to remember saying these points:
1. Maximize data to ink ratio: show the data
2. Don't lie with scale: minimize
3. Minimize chart-junk: show data variation, not design variation
4. Clear, detailed and thorough labeling

##### QUESTION2: WHAT IS A SCATTER PLOT? FOR WHAT TYPE OF DATA IS SCATTER PLOT USUALLY USED FOR?

- A scatter plot is a chart used to plot a correlation between two or more variables at the same time. It's usually used for numeric data.

##### QUESTION3: WHAT TYPE OF PLOT WOULD YOU USE IF YOU NEED TO DEMONSTRATE “RELATIONSHIP” BETWEEN VARIABLES/PARAMETERS?

- When we are trying to show the relationship between 2 variables, scatter plots or charts are used. When we are trying to show "relationship" between three variables, bubble charts are used.

