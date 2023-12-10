# Churn Prediction using Pycaret

Hello everyone, in this project I will give you some information about Customer Churn Prediction and we will build a Churn Prediction model together using Pycaret.

![1_47xx1oXuebvYwZeB0OutuA](https://github.com/enesbesinci/CRM-churn-prediction-using-Pycaret/assets/110482608/2720deef-03e6-4eb1-9773-18689639054b) [1]

## Introduction

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




## Sources:

* [1] [Classification Problem: Customer Churn in a Bank by gokcesimge](https://medium.com/i%CC%87stanbuldatascienceacademy/classification-problem-customer-churn-in-a-bank-aab878ef87f7)
* [2] [Customer Churn for any Timeline by Sai Teja Pasula](https://saitejapasula.medium.com/customer-churn-for-any-timeline-fbea57c146a7).
* [3] [Predict Customer Churn using Pycaret by Moez Ali](https://towardsdatascience.com/predict-customer-churn-the-right-way-using-pycaret-8ba6541608ac)



