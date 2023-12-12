# Churn Prediction using Pycaret on Python

Hello everyone, in this project I will teach you about Customer Churn Prediction and build a Churn Prediction model using practical libraries in the field of machine learning such as Pycaret, Pandas Profiling.

![1_47xx1oXuebvYwZeB0OutuA](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/2720deef-03e6-4eb1-9773-18689639054b) [1]

## STEP 1: Introduction

First of all, I would like to explain a concept known as **KPI** in the business world.

A **Key Performance Indicator (KPI)** is a specific metric used to measure and evaluate the performance of an organization. KPIs are often directly related to the organization's strategic goals and are used to understand, monitor and evaluate the company's performance. Using KPIs, businesses can assess progress towards achieving specific goals. Furthermore, these metrics must be quantitative so that they can be calculated and monitored.

For example, this KPI could be "Monthly Active Subscribers" for Netflix, "Monthly Total Sales Revenue" for Amazon, or "Overall Employee Happiness" for a company's human resources department.

Customer retention is also a KPI and is a metric that measures a business's ability to retain customers over a given period of time. It is calculated as a percentage and is usually expressed as the number of customers that remain with a business after a given period divided by the total number of customers at the beginning of the period.

Customer retention is important for all businesses, but it is particularly important for subscription-based businesses. This is because subscription-based businesses rely on recurring revenue from their customers. When a customer unsubscribes, the business loses this revenue.

Yes, now we have enough information about the KPI's.
## What is Customer Churn?

Customer churn refers to the number of customers a business loses from its customer base during a specific period. This metric is particularly crucial for subscription-based businesses, such as streaming services like Netflix, music platforms like YouTube Music, and telecom providers like Türk Telekom. Customer churn is typically expressed as a rate, calculated by dividing the number of customers lost by the total number of customers at the beginning of that period. For example, if you got 100 customers and lost 5 last month, then your monthly churn rate is 5 percent.

Customer churn is considered a significant business metric because the costs of acquiring new customers are often higher than retaining existing ones. Therefore, reducing or preventing customer churn is important for a business's long-term sustainability.

Customer churn can be influenced by various factors, including customer dissatisfaction, competition, service quality, pricing policies, customer support, and changing customer needs. To predict and prevent customer churn, businesses develop customer churn prediction models and strategies.

## Benefits of Customer Churn Analysis for the Business

Customer Churn Analysis or Churn Prediction is important because it provides several advantages to the business, here are some advantages:

Preventing Customer Loss: Churn prediction enables the ability to forecast the likelihood of customers leaving in advance. These predictions offer companies the opportunity to develop strategies to prevent or minimize customer loss.

Cost Savings: The costs of acquiring new customers are often higher than retaining existing ones. Predicting customer churn in advance allows companies to save costs, as they can focus on more effective strategies for customer loyalty and satisfaction.

Strengthening Customer Relationships: Preventing customer loss translates to strengthening customer relationships. Churn prediction can help companies better understand customer needs and provide tailored solutions.

Competitive Advantage: Predicting customer loss in advance provides companies with a competitive advantage over rivals. Swiftly responding and offering solutions to customer dissatisfaction can increase customer loyalty and position the company ahead in the market.

## Stages of Customer Churn Analysis

### How to Create Customer Churn Dataset for Machine Learning Model?

First, we start by collecting the company's historical data to use in the Machine Learning model to predict Churn.

Here are two very important terms used in almost every Customer Churn Analysis: Cut-off Point or Date and Performance Period or Window. I will explain them with a quote.

![Screenshot 2023-12-10 204025](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/3df67a8e-9b9d-436d-aff2-48d6e74e2837) [2]

Simply, the 12-month period you see in the graph is the input variables of our Machine Learning model, while the so-called Performance Window is the output variables of the model (yes=1 and no=0).

I am adding one more graph for better understanding.

![Screenshot 2023-12-10 204210](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/ea85bf23-cc96-44ad-89b7-cd00820a2f05) [3]

This is followed by steps such as data cleaning, EDA, feature engineering, and model creation.

Periodically or every month, depending on our objectives, Churn probabilities for each customer in the company's database are estimated through Machine Learning models and customers are categorized as Low Risk, Medium Risk and High Risk (there may be more categories).

These results are transferred to the necessary departments and customers are tried to be kept in the company with various promotions, coupons or other tools.

Now let's make an example using Python on Jupyter Lab.

## STEP 2: About the Dataset

In this project, we will use the Telco Customer Churn Public Dataset from Kaggle and IBM.

Telco Customer Churn Data contains information about a fictitious telco company that provided home phone and Internet services to 7043 customers in California in Q3. It shows which customers left or stayed with their service. [4]

Dataset Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---------------------------------------------------------------------------------------

### Dataset Features

**customerID** : Customer ID

**gender** : Whether the customer is a male or a female

**SeniorCitizen** : Whether the customer is a senior citizen or not (1, 0)

**Partner** : Whether the customer has a partner or not (Yes, No)

**Dependents** : Whether the customer has dependents or not (Yes, No)

**tenure** : Number of months the customer has stayed with the company

**PhoneService** : Whether the customer has a phone service or not (Yes, No)

**MultipleLines** : Whether the customer has multiple lines or not (Yes, No, No phone service)

**InternetService** : Customer’s internet service provider (DSL, Fiber optic, No)

**OnlineSecurity** : Whether the customer has online security or not (Yes, No, No internet service)

**OnlineBackup** : Whether the customer has online backup or not (Yes, No, No internet service)

**DeviceProtection** : Whether the customer has device protection or not (Yes, No, No internet service)

**TechSupport** : Whether the customer has tech support or not (Yes, No, No internet service)

**StreamingTV** : Whether the customer has streaming TV or not (Yes, No, No internet service)

**StreamingMovies** : Whether the customer has streaming movies or not (Yes, No, No internet service)

**Contract** : The contract term of the customer (Month-to-month, One year, Two year)

**PaperlessBilling** : Whether the customer has paperless billing or not (Yes, No)

**PaymentMethod** : The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))

**MonthlyCharges** : The amount charged to the customer monthly

**TotalCharges** : The total amount charged to the customer

**Churn** : Whether the customer churned or not (Yes or No)

---------------------------------------------------------------------------------------

## STEP 2: Data Understanding and EDA

In this step, we will take an overview of the dataset and explore the relationships between customer variables and churn. We will use the "Pandas Profiling" library to do this. This Python library allows us to perform EDA with just a few lines of code.

**Warning**: We will not do a detailed Exploratory Data Analysis as it is not the main topic of this article.

First we need to import the necessary libraries and read the dataset, then we can get an overview of the dataset with the info() function.

![Screenshot 2023-12-12 133916](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/d998ce9f-22aa-48c8-a3cb-ba9010352b6d)

After we have seen the variable types, let's see the first 5 lines.

![Screenshot 2023-12-12 134249](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/f35f97c8-170e-4c27-a2b4-c6c8084605f8)

Instead of visualizing each variable individually during the EDA phase and writing functions for missing data or outliers, Pandas Profiling makes it easy to see information about the entire data set.

![Screenshot 2023-12-12 134719](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/603cde81-ad46-41ac-b990-f36b445f7c54)

Save the results in an HTML file.

![Screenshot 2023-12-12 134815](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/34d61d6d-2c30-451c-ab6f-b885f8980ac2)

Let's see the results shortly.

![Screenshot 2023-12-12 134510](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/8215d1b3-f7bf-499a-8507-09e1bbe78aa5)

Churn is our target feature to predict which customers have churned.

![Screenshot 2023-12-12 134948](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/84b40fce-b596-47c5-a61a-a617a008b8c5)

Our target feature "Churn" seems to be imbalanced, let's continue.

Other numerical features:

![Screenshot 2023-12-12 174109](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/d185934b-035e-436f-930f-ca5c906dbb11)

![Screenshot 2023-12-12 174119](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/add2aaa2-7393-49f8-80b4-0069dfe402ff)

I will not analyze all the variables, I will be uploading the HTML file to the Repo so that you can analyze it.

Now let's see the effects of some variables on Churn.

I keep all categorical and numeric variables in different lists to make visualizations faster.

![Screenshot 2023-12-12 135509](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/935c70c0-dbd1-4ed0-8c8a-796eff943867)

And after that we change the target values for visualizations from Yes-No to 1-0

![Screenshot 2023-12-12 135518](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/ec629bb6-0ad0-4b4b-98c7-bf05d682f988)

Let's look at how all the categorical variables interact with "Churn" with a simple for loop and analyze some of them.

![Screenshot 2023-12-12 135721](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/c40c69fe-34be-460d-bd76-e3c9457161a5)

The results:

![Screenshot 2023-12-12 174826](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/10448627-290c-4745-83b0-016208463cd8)

As you can see, there is no correlation between "Gender" and "Churn".

Now let's look at other features that might be important.

![Screenshot 2023-12-12 174902](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/842e6dfa-fa46-4e5b-b4f6-001b074c0120)

We see that customers using "Fiber" as their Internet service have a much higher churn rate. This is actually unexpected, as these customers are using a higher quality service. However, many factors can cause this, such as price, quality of service, customer service, etc.

Let's look at the relationship between "Contract" and "Churn".

![Screenshot 2023-12-12 140656](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/14076fb7-13e6-45de-b4a5-54c73682eda6)

As expected, the shorter contract means higher churn rate.

Now let's look at the relationship between "Tenure" and Churn. Our expectation is that the two are inversely correlated. High Tenure means low Churn.

![Screenshot 2023-12-12 145522](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/b33201e5-1372-4233-8d80-7f19b53a5a50)

Exactly as we expected. A detailed analysis of all variables can be found in this article. Let's start building the model.

Detailed analysis of all variables: https://medium.com/@zulfikarirham02/telco-customer-churn-prediction-using-machine-learning-and-deep-learning-8d1905b04980

## STEP 3: Building a Classification Model to Predict Churn with Pycaret

Let's see if there is any missing data in our data set.

![Screenshot 2023-12-12 145959](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/093560a6-1cb4-4b10-b886-a89973a2f3f2)

The results:

![Screenshot 2023-12-12 150055](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/1b21eb5a-434f-4987-a557-f5c35a85418b)

We only have 11 missing data in our "TotalCharges" column, but it doesn't matter, in this project we won't do any data preprocessing (outliers and missing data or feature engineering etc.) steps like in other projects, Pycaret will do that for us.

But first, what is Pycaret?

They describe themselves as follows: PyCaret is an open source, low-code machine learning library and end-to-end model management tool built in Python for automating machine learning workflows. PyCaret is known for its ease of use, simplicity, and ability to quickly and efficiently build and deploy end-to-end machine learning pipelines. To learn more about PyCaret, please visit their GitHub.

GitHub/Pycaret: https://github.com/pycaret/pycaret

Let's build a classification model using Pycaret. As you can see, I only gave Pycaret the target variable and the dataset, I also told it that the target variable is unbalanced and that it should not include a variable like "customerID" in the model. Pycaret will handle all the variable engineering, missing and outlier data, etc.

![Screenshot 2023-12-12 181256](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/429ecc70-9638-44c1-99a6-a7c31da6c3d2)

The result:

![Screenshot 2023-12-12 181312](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/30c12af0-d2aa-4627-849b-682b4b0ea626)

![Screenshot 2023-12-12 181349](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/e8095d0c-3100-4c14-abd5-fcc8ead59be5)

As you can see, all data pre-processing steps were performed automatically by Pycaret.

Now let's train our model, the compare_models() function trains all the models in the Pycaret model repository and prints various evaluation metrics.

![Screenshot 2023-12-12 195105](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/4c1d4db2-531f-489f-915a-5a74e43cb672)

To see best parameters for the model:

![Screenshot 2023-12-12 195431](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/2e54ee39-fe78-4c0b-8347-e052234637e2)

At this point we can automatically tune the hyperparameters of the model if we want but for this project I won't. If you want to automatically tune the hyperparameters of your model you can use tune_model() function.

Let's take a look at the variables that our model is paying more attention to in the prediction process.

![Screenshot 2023-12-12 195706](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/64a480d8-3df1-4843-9ebc-22635452af8d)

Let's look at the confusion matrix and analyze the model.

![Screenshot 2023-12-12 213856](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/0066b081-c363-43c6-8b5e-fb1dde3a039c)

Now let's interpret the results, this part is very important, please read carefully.

As you can see in the confusion matrix, 1346 customers are not really churned and we estimated them correctly. 227 customers are churned (1 in reality, but we estimated them as 0). 206 customers are not really churning, but we have categorized them as churning. Since these customers will not really churn, we will incur unnecessary costs to bring them back to the company (such as coupons or promotions). 334 customers will actually churn and we categorized them as churn, so we guessed right.






Let's consider an example like this:

For every customer we keep from churning, we earn $5000 in customer lifetime value. And let's say we give a $1000 gift card to every customer we think will churn.

Our model says that a total of 540 customers (True Positive + False Positive) will churn, and we gave each customer a $1000 gift certificate. Suppose we prevented all of the customers who would have churned from churning. As a result, we gave a total of 1000 x 540 = $540,000 in gift certificates to all the customers we thought would churn, but we only generated 334 x 5000 = $1,670,000 in revenue. Our total profit is 1,670,000 - 540,000 = 1,130,000 dollars.

What I mean is that it is not right to choose the appropriate model for the business problem by looking only at metrics like AUC, Recall, Accuracy. For each business problem, we can develop a metric that is appropriate for that problem.

Now let's develop our own metric in Pycaret and find and use the model that brings the most profit/income for our business problem, not the model that gives the best AUC score as before.

![Screenshot 2023-12-12 221110](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/398f7bc4-9187-4249-9233-97325156c330)

Now let's look again at the performance of all models according to the "profit" metric we developed and select the appropriate model.

![Screenshot 2023-12-12 221211](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/c5bfe338-1558-462a-beeb-4aad202f9d59)

Logistic regression gives best results by profit metric. Let's interpret the model.

![Screenshot 2023-12-12 221226](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/7a7937b2-1c2b-4a8e-8586-5e6b083717a1)

(441 + 399) We predicted that 840 customers will Churn and we gave each of them a $1000 gift voucher. But as you can see, only 441 customers will actually Churn and assuming we win them all back, we will get 441 x 5000 = 2,205,000 in revenue from these customers.

2.205.000 - 840.000 = 1.365.000 profit.

As you can see, we have seen that we can earn higher revenues from models with lower "Accuracy" and "AUC". And now we can take action. We should contact the customers we think will churn and make them our active customers again.
If we want, we can segment these customers as "High Risk", "Medium Risk" and "Low Risk" according to their Churn probability and apply different strategies.

Thanks for reading.


























































## Sources:

* [1] [Classification Problem: Customer Churn in a Bank by gokcesimge](https://medium.com/i%CC%87stanbuldatascienceacademy/classification-problem-customer-churn-in-a-bank-aab878ef87f7)
* [2] [Customer Churn for any Timeline by Sai Teja Pasula](https://saitejapasula.medium.com/customer-churn-for-any-timeline-fbea57c146a7).
* [3] [Predict Customer Churn using Pycaret by Moez Ali](https://towardsdatascience.com/predict-customer-churn-the-right-way-using-pycaret-8ba6541608ac)
* [4] [About the Dataset](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)



