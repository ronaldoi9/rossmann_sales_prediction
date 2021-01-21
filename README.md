# Forecast Rossmann Store Sales

![rossmann](https://user-images.githubusercontent.com/40616142/104977231-ca344600-59dd-11eb-94c4-26f431bc2802.png)

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

## BUSINESS QUESTIONS <p id="bquestions"></p>

In this project, the CFO summoned all store managers to deliver the sales forecast for up to six weeks advance.

Many of these managers contacted me for consult the data to make the prediction for their stores. So, I went looking for the real stakeholder to understand the main reason for this sales forecast. Talking to managers, they told me that was a request from the CFO at a meeting with all managers.

When discussing with the CFO, he informs me that wants to know the store sales forecast because wants to renovate them, and he would like to know how much money will have in cash to carry out the construction. I understod the main problem and start looking for the data that will help me make this prediction.

As this is a problem originating from a Kaggle competition, the data are available through their platform, and it is not necessary to use extraction from databases and other sources.

**Motivation:** The CFO requested this solution during a monthly results meeting.

**Main problem:** Investment in store renovation

**Stakeholder:** CFO

**Solution Format:** Daily sales for the next 6 weeks, prediction problem, time series, mobile delivery (Telegram).

## METHODOLOGY <p id="methodology"></p>

In this project i used the CRISP-DS (*Cross Industry Standard Process for Data Science*) development methodology, which consists of cycles of interaction and continuous improvement. The image below shows the steps taken to solve the problem.

![CRISP CYCLE](https://user-images.githubusercontent.com/40616142/104822560-189ce580-5822-11eb-8491-7d4ee6698ba3.png)


Main advantages of CRISP-DS:
- With each completed cycle, it is already possible to deliver value to the business.
- Value delivery speed.
- Mapping of all possible project problems.

## BUSINESS UNDERSTANDING <p id="bunderstanding"></p>

This step consists in the identification and understanding of the company's business demand, seeking to understand the true Stakeholder and whether their request can really be carried out. As previously presented, we know that the stakeholder is the CFO, and the solution format will consist of a stores forecast sales using a time series.

## DATA COLLECT <p id="data_collect"></p>

The data used in this project are available through the Kaggle platform, and can be find [here](https://www.kaggle.com/c/rossmann-store-sales/data). But, if it were a real company environment, this data would be collected through database queries, and other sources of information.

## DATA CLEANING <p id="data_cleaning"></p>

Cleaning the dataset downloaded from the previous step, performing operations such as:

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

## EXPLORATORY DATA ANALYSIS <p id="data_exploration"></p>

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

## DATA MODELLING <p id="data_modelling"></p>

A modelagem dos dados é divida em duas etapas que são essenciais para a aplicação dos algoritmos de machine learning. A primeira etapa consiste na Preparação dos Dados e a segunda é a Seleção das Features mais relevantes.

### Data Preparation <p id="data_preparation"></p>

Analyzing the table with basic statistics of the numerical variables, it was notice that the metric *range* are very wide, indicating that the data need a normalization or a rescale. Therefore, the data preparation aims to analyze which metric will be applied to carry out this standardization. This method was divided into 3 stages:

- **1. Normalization**

Checking the numerical features graphs in **Univariate analysis** section, it was concluded that the data we worked on does not have a normal distribution. Therefore, normalization of these will not be applied. This step would be used if any characteristic has a normal distribution pattern, which is not the case.

- **2. Rescaling**

The standardization method defined in this project will be the Rescaling. To identify which rescaling transformation method to use, i  check the boxplot variables to verify which will be the best. Analyzing the graphs below, it was noticed that for features `competition_distance` and `competition_time_month` the most suitable method will be Robust Scaler, due to its high number of outliers, since the features `year` and `promo_time_week` the transformation method will be Min Max Scaler, as they do not have as many outliers.

![rescaling](https://user-images.githubusercontent.com/40616142/105102831-39697300-5a8e-11eb-9adc-7bd522db7820.png)


- **3. Transformation**

- **3.1 Encoding**

In this step I'll use some popular methods used to perform categorical variable encoding, such as *One Hot Encoding, Label Encoding and Ordinal Encoding*. As we are in the first cycle of CRISP-DS, i can encoding according to my guess for each feature. Remembering that nothing prevents you from using the same type of encoding for all variables, which can be revised in the second cycle.

- **3.2 Response Variable Transformation**

The response variable does not have a normal distribution, so a logarithmic transformation will be applied to `sales` feature, making it closer to a normal one, as shown in the image below.

![sales_tranformation](https://user-images.githubusercontent.com/40616142/105103375-476bc380-5a8f-11eb-8159-12c15cdcb1ac.png)

- **3.3 Nature Transformation**

Finally, the last transformation carried out on this dataset will be the  nature transformation. That will be applied to time variables that are repeated, since we want to preserve the cyclical effect of these characteristics, such as `weekday`,` month`, `day` and` week_year`.

### Feature Selection <p id="feat_selection"></p>

The first step before selecting the features, will be to identify which dates will be part of the training and test dataset. The last 6 weeks will be the test data and the training data will be the entire dataset up to the discovered deadline. Check the notebook for more information.

The selection of features is not a simple task, so in this project I'll use a very effective tool for this purpose, the *Boruta Algorithm*. This algorithm seeks to identify which features will be most relevant to the Machine Learning model, but its identification process demands a long processing time.

## MACHINE LEARNING ALGORITHMS <p id="machine_learning"></p>

It is this project stage that involves the part that most data scientists like to work on, the use of machine learning algorithms. All the previous steps were designed to maximize the algorithms efficiency.

### Machine Learning Modelling <p id="machine_learning_modelling"></p>

**Average Model**

Well, to choose which model to use, we must start from a simple model that solves our problem, this should always be the goal for any type of resolution, so as not to "kill an ant using a war tank". My initial modeling will be an algorithm that calculates the average sales for each store, and then evaluate the algorithm performance.

| Model Name	| MAE	| MAPE	| RMSE |
| ----------- | ----------- |  ----------- |  ----------- |
| Average Model |	1354.800353 |	0.455051 |	1835.135542

It is noticed that using this approach the ** RMSE ** (Root Mean Squared Error) was very high. But, this performance will serve as a baseline to compare with the other algorithms.

**Linear Regression Model**

I intend to use an algorithm very popular in the data science world, the Linear Regression. Using this algorithm I intend to identify if the problem we are trying to model, has a linear behavior. The table below shows the performance obtained using this algorithm.

| Model Name	| MAE	| MAPE	| RMSE |
| ----------- | ----------- |  ----------- |  ----------- |
| Linear Regression	| 1867.089774 |	0.292694	| 2671.049215

For our surprise, the RMSE value surpassed our simple average model, indicating that this problem has non-linear characteristics, requiring use non-linear algorithms. But for comparative purposes I'll use another linear aspect algorithm to compare with a simple linear regression.

**Linear Regression Regularized Model**

The Regularized Linear Regression algorithm has a linear characteristic, but unlike the previous one, it uses an alpha parameter that allows to regulate a line inclination using weights. The table below shows a performance of this algorithm.

| Model Name	| MAE	| MAPE	| RMSE |
| ----------- | ----------- |  ----------- |  ----------- |
| Linear Regression Regularized - Lasso |	1891.704881	| 0.289106	| 2744.451737

**Random Forest Regressor**

Then I start using non-linear algorithms to perform a better  problem modeling, starting with the Random Forest Regressor, which is a tree-based algorithm that is very popular for classification, but is also very powerful for making value predictions. The table below shows the results obtained with this algorithm.

| Model Name	| MAE	| MAPE	| RMSE |
| ----------- | ----------- |  ----------- |  ----------- |
| Random Forest Regressor |	686.894635 |	0.101039 |	1024.295227

**XGBoost Regressor**

Another non-linear algorithm that was built using the concept of trees, this algorithm became very popular to be implemented by competitors who won Kaggle competitions, from then on it was highly disseminated in the data science community. The table below shows the results obtained with this algorithm.

| Model Name	| MAE	| MAPE	| RMSE |
| ----------- | ----------- |  ----------- |  ----------- |
| XGBoost Regressor | 843.112293 |	0.122609	| 1250.952637

But, the performance tables generated above were formed using the dataset without cross-validation, which brings the real algorithm performance. This approach makes the model use all partitions of the training dataset, exposing the algorithm to greater variations in data, consequently increasing reliability on results. The table below shows the algorithms performances using cross-validation.

| Model Name	| MAE CV	| MAPE CV	| RMSE CV |
| ----------- | ----------- |  ----------- |  ----------- |
| Random Forest Regressor	| 837.6 +/- 216.27	| 0.12 +/- 0.02	| 1256.95 +/- 317.35
|	XGBoost Regressor	| 1030.28 +/- 167.19	| 0.14 +/- 0.02	| 1478.26 +/- 229.79
|	Linear Regression	| 2081.73 +/- 295.63	| 0.3 +/- 0.02	| 2952.52 +/- 468.37
|	Lasso	| 2116.38 +/- 341.5	| 0.29 +/- 0.01	| 3057.75 +/- 504.26

Analyzing non-linear algorithms performance, was notice the 2 algorithms had the best results were **Random Forest Regressor** and **XGBoost Regressor**, both got a very similar RMSE and much smaller than linear algorithms. Therefore, i need to use my intuition as a data scientist choose which model will be used in production. Analyzing the the time it took the Random Forest Regressor to be trained, and the possible financial cost this algorithm can bring to be implemented, considering the difference between it and the **XGBoost Regressor** is not to high compared to the others models, I decided to choose **XGBoost** as the ideal algorithm to put the model into production and adjust the parameter to improve its performance.
