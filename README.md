# Rossman Prediction Sales

---
## Contents <p id="contents"></p>
- <a href="#bquestions">Business Questions</a>
- <a href="#methodology">Methodology</a>
  - <a href="#bunderstanding">Business Understanding</a>
  - <a href="#data_collect">Data Collect</a>
  - <a href="#data_cleaning">Data Cleaning</a>
    - <a href="#data_description">Data Description</a>
    - <a href="#feat_engineering">Feature Engineering</a>
    - <a href="#filtering_var">Filtering Variables</a>
  - <a href="#data_exploration">Data Exploration</a>
    - <a href="#univariate_analysis">Univariate Analysis</a>
    - <a href="#bivariate_analysis">Bivariate Analysis</a>
    - <a href="#multivariate_analysis">Multivariate Analysis</a>
  - <a href="#data_modelling">Data Modelling</a>
    - <a href="#data_preparation">Data Preparation</a>
    - <a href="#feat_selection">Feature Selection</a>
  - <a href="#machine_learning">Machine Learning Algorithms</a>
    - <a href="#machine_learning_modelling">Machine Learning Modelling</a>
    - <a href="#hyperparameter_fine_tuning">Hyperparameter Fine Tuning</a>
  - <a href="#evaluate">Evaluate Algorithm</a>
    - <a href="#error_interpretation">Error Interpretation and Translation</a>
  - <a href="#deploy">Deploy Model</a>
    - <a href="#error_interpretation">Deploy Model on Heroku</a>
  - <a href="#telegram_bot">Telegram Bot</a>
--- 

## Business Questions <p id="bquestions"></p>

In this project, the CFO summoned all store managers to deliver the sales forecast for up to six weeks advance.

So, many of these managers contacted me for consult the data to make the prediction for their stores. So, I went looking for the real stakeholder to understand the main reason for this sales forecast. When talking to managers, I was told that was a request from the CFO at a meeting with all managers.

When discussing with the CFO, he informs me that wants to know the store sales forecast because wants to renovate them, and he would like to know how much money will have in cash to carry out the construction. So, i understand the root cause and start looking for the data that will help me make this prediction.

As this is a problem originating from a Kaggle competition, the data was made available through the platform, and it is not necessary to use extraction from databases and other sources.

**Motivation:** The CFO requested this solution during a monthly results meeting.

**Main problem:** Investment in store renovation

**Stakeholder:** CFO

**Solution Format:** Daily sales for the next 6 weeks, prediction problem, time series, mobile delivery (Telegram).

## Methodology <p id="methodology"></p>

In this project I will use the CRISP-DS (*Cross Industry Standard Process for Data Science*) development methodology, which consists of cycles of interaction and continuous improvement. The image below shows the steps taken to solve the problem.

![CRISP CYCLE](https://user-images.githubusercontent.com/40616142/104822560-189ce580-5822-11eb-8491-7d4ee6698ba3.png)


Main advantages of CRISP-DS:
- With each completed cycle, it is already possible to deliver value to the business.
- Value delivery speed.
- Mapping of all possible project problems.

## Business Understanding <p id="bunderstanding"></p>

This step consists in the identification and understanding of the company's business demand, seeking to understand the true Stakeholder and whether their request can really be carried out. As previously presented, we know that the stakeholder is the CFO, and the solution format will consist of a stores forecast sales using a time series.

## Data Collect <p id="data_collect"></p>

The data used in this project are available through the Kaggle platform, and can be find [here](https://www.kaggle.com/c/rossmann-store-sales/data). But, if it were a real company environment, this data would be collected through database queries, and other sources of information.

## Data Cleaning <p id="data_cleaning"></p>

Consists of cleaning the dataset downloaded from the previous step, performing operations such as:

### Data Description <p id="data_description"></p>

  - Rename column names.
  - Check the size and type of the data.
  - Check for the existence of missing values, and if so, use an approach to fill in these values.
    - *In this step i found some columns with missing values, and filled them thinking about the business.*
  - Standardize the types of variables.
  - Perform a descriptive analysis of the data to gain business knowledge and be able to identify any inconsistency in the information.
    - *This descriptive analysis consisted of creating a table (for numerical variables) containing the basic statistical metrics of the data set, and building a boxplot (for categorical variables) to observe its behavior.*
    
    Statistical metrics for **numerical variables**. Looking at the table below, we realize that the data are not normalized, the range variable shows the discrepancy in the variation of some features
    ![metrics](https://user-images.githubusercontent.com/40616142/104855382-3bdf9780-58eb-11eb-8cb7-dc22c27c2c0e.png)
    
    Analyzing **categorical variables** behavior with the boxplot.
    ![categorical_boxplot](https://user-images.githubusercontent.com/40616142/104855511-038c8900-58ec-11eb-9fee-766111559c15.png)

### Feature Engeneering <p id="feat_engineering"></p>

Important to increase the amount of information needed to better understand the phenomenon we are trying to model. Feature Engineering is also instrumental in obtaining more variables available for study during Data Analysis, which is the next step in this project.

- **Hypothesis Mind Map.**

Roadmap that shows what analyzes we need to do to validate some hypotheses, and what variables we should derive. These steps help to create faster EDA (Exploratory Data Analysis) and bring valuable insights. The figure below shows the hypothesis mental map built in this first stage of the cycle.

![mind_map_hypothesis](https://user-images.githubusercontent.com/40616142/104855234-56fdd780-58ea-11eb-8658-a5e3807dd4f6.png)

Analyzing the mind map, a list is constructed with the hypotheses to be tested, it is noticed that this has only those that are possible to be answered based on the available data set. In a real environment, the missing information would be requested from the data engineering team to provide some important variables to answer the other hypotheses.

| Hypothesis      | Validate |
| ----------- | ----------- | 
| Stores with a larger assortment of products should sell more. | -
| Stores with closer competitors should sell less. | -
| Stores with longer competitors should sell more. | -
| Stores with active promotions for longer should sell more. | -
| Stores with more consecutive promotions should sell more. | -
| Stores that open during the Christmas holiday should sell more. | -
| Stores should sell less on weekends. | -
| Stores should sell more over the years. | -
| Stores should sell more in the second half of the year. | -
| Stores should sell more after the 10th day of each month. | -
| Stores should sell less during school holidays. | -

To increase the features available for analysis, we will derive the following columns:

- `date`: originates `new year`, `month`, `day`, `week_of_year`, and `year_week columns` columns.

- `competition_open_since_year` and `competition_open_since_month` originates `competition_since` column.

- `promo2_since_year` and `promo2_since_week` originates `promo_since` column.

Categorical features like `state_holiday` and `assortment` had their attributes renamed from character to their real label.

### Filtering Variables <p id="filtering_var"></p>

Identify which are the variables that have some business restriction, and which will be available to put the model into production. Our goal is to predict the value of sales up to six weeks in advance, therefore, we will not have some information that our dataset provide at the time of forecast.

In this step we will excluding variables such as: 

- `customers`: we don’t know the number of customers in stores in future.
- `open`: as we only filter when stores are open, this column is irrelevant in our analysis.
- `promo_interval` and` month_map` were used to create new columns and not will be used anymore.

## Exploratory Data Analysis <p id="data_exploration"></p>

The exploratory data analysis proposes to analyze how the variables map the phenomenon we want to model, and what is the strength of this impact. Proposed basically for 3 objectives:

- Gain business experience.
- Validate business hypotheses and generate insights.
- Identify variables that are important to the model.

This analysis was divided into 3 stages:

### Univariate Analysis <p id="univariate_analysis"></p>

Studies the behavior of the resource, divided between numeric and categorical variables. Analyzing the behavior of the response variable `sales`, we noticed that it does not have a normal distribution, which can influence the modeling of the machine learning algorithm which are usually more effective in variables that exhibit Gaussian behavior.

![response_variable](https://user-images.githubusercontent.com/40616142/104969704-f93fbd00-59c7-11eb-8cbe-56ffc2c3d84b.png)

Analyzing the graph below the numerical variables, it can be seen that the feature `day_of_week` does not have variations, the stores basically sold the same quantity during each day of the week, therefore, this feature will not be relevant to the model. Other features presented important information for the business, such as `is_promo`, it is noticed that there were more sales of products that were not on sale, which can be an insight for the business team.

![numerical_variables](https://user-images.githubusercontent.com/40616142/104971274-06ab7600-59cd-11eb-9c45-ce4cbf8d4296.png)

Analyzing the graph of categorical variables, it can be seen that the variable `state_holiday` has 3 holidays in this data set and among them the one with the greatest number of holidays is *public_holiday*. The variable `store_type` has 4 different types of stores, and the` assortment` has 3 types of category. The graphs on the right show an illustration of the amount of sales for each of them.

![categorical_variables](https://user-images.githubusercontent.com/40616142/104971392-6ace3a00-59cd-11eb-98e4-710451e7c746.png)

### Bivariate Analysis <p id="bivariate_analysis"></p>

Studies the impact of each feature on the response variable, this stage of the project basically consists on testing the hypotheses elucidated previously by the mind map, gain business knowledge, and seeking to understand how relevant some features will be for the machine learning model by analyzing their correlation with the response variable. I will not write all the hypothesis tests in this documentation, and will leave only one example showed in the image below, the others can be found in the notebook.

**Hypothesis: Stores should sell more after the 10th day of each month**
![hp20](https://user-images.githubusercontent.com/40616142/104972846-b71b7900-59d1-11eb-9df7-b49ec28a808e.png)

This hypothesis was validated as **true**. Analyzing the graphs above, it can be seen that stores make more sales after the 10th day of each month. To infer the correlation of the feature with the response variable, a heatmap was used. In which, it is noticed that this variable alone has a reasonable negative correlation with sales.

Upon completion of the hypothesis tests, a summary of the result and relevance to machine learning model is shown in the table below.

| Hypothesis      | Validate | Relevance |
| ----------- | ----------- |  ----------- |
| Stores with a larger assortment of products should sell more. | False | Low 
| Stores with closer competitors should sell less. | False | Medium
| Stores with longer competitors should sell more. | False | Medium
| Stores with active promotions for longer should sell more. | False | Low
| Stores with more consecutive promotions should sell more. | False | Low
| Stores that open during the Christmas holiday should sell more. | False | Medium
| Stores should sell less on weekends. | True | High
| Stores should sell more over the years. | False | High
| Stores should sell more in the second half of the year. | False | High
| Stores should sell more after the 10th day of each month. | True | High
| Stores should sell less during school holidays. | True | Low

### Multivariate Analysis <p id="multivariate_analysis"></p>

Study how the variables are related, analyzing the general correlation between the features of the dataset. First, the inspection of numerical variables is performed using the Pearson correlation, shown in the image below. Analyzing the image below, it can be seen that the variable `customers` has a high correlation with the response variable, however, this variable cannot be used when forecasting sales, as we will not have the number of customers in the store in the future. Some other features also showed a considerable correlation, however, we will not exclude them from our analysis because they have relevant information.

![numerical_corr](https://user-images.githubusercontent.com/40616142/104974222-f64bc900-59d5-11eb-8c72-756bebdf901e.png)

To analyze the correlation between categorical variables, the Cramer's V method was used, which is very popular to evaluate this type of variable. Analyzing the correlation of categorical variables, it is noticed that the features `store_type` and` assortment` have a considerable correlation, but it is not enough to choose one for exclusion, as they present information that will be important for the model.

![categorical_corr](https://user-images.githubusercontent.com/40616142/104974439-ca7d1300-59d6-11eb-8dc2-b28185125615.png)





