
NOTE:      CAN ALSO REFERENCE README ipynb file above



# Bank Customer Churn Prediction
[Project Description/Goals] [Initial Questions] [Planning] [Data Dictionary] [Reproduction Requirements]

[Conclusion]

# Project Description/Goals:

There is concern by banks why customers move from one bank to another. In this regard we need to understand which aspects of service influence customer's decisions to leave. And with this case study, bank management can concentrate their efforts to improve their services to retain customers.

# Business understanding
The goal of this project is to identify key drivers of customer churn and build a neutral model that accurately determine whether they will leave.

# Data understanding
The bank customer churn data was obtained from Kaggle open-source dataset. It is stored in a csv file, named as "Bank Customer Churn Prediction.csv". It has 12 columns called features, including 10000 row numbers. The fetures are customer id, credit score, country, gender, age, tenure, balance, number of products purchased through the bank, whether customer has a credit card, whether customer is an active member, estimated salary, and whether customer left the bank.

# Prepare data
For categorical data features country and gender was handled by encoding. After checking the missing values in the dataset, it showed no missing values.

# EDA, modeling
Divided whole dataset into training data and test data. The overall churn rate is 20%. I explored the correlation between every features and churn outcome. I built three machine learning models (Decision Tree, Random Forest and Logistic Regression) to predict the customer churn.

# Evaluate the results
Evaluated the predictions with accuracy and confusion matrix.

[Back to top]

Initial Questions:
What are the key factors that affect customer churn?
How do these factors affect customer churn?
Which is the best method to predict customer churn?
[Back to top]

# Planning:
Create README.md with data dictionary, project and business goals, and come up with initial hypotheses.

Acquire data from The bank customer churn data was obtained from Kaggle open-source dataset. It is stored in a csv file, named as "Bank Customer Churn Prediction.csv".

Clean and prepare data for the first iteration through the pipeline, MVP preparation. Create a function to automate the process.

Store the acquisition and preparation functions in a wrangle.py module function, and prepare data in Final Report Notebook by importing and using the function.

Clearly define at least two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.

Establish a baseline accuracy and document well.

Train 3 different regression models.

Evaluate models on train and validate datasets.

Choose the model that performs the best and evaluate that single model on the test dataset.

Document conclusions, recommendations, and next steps in the Final Report Notebook.
[Back to top]

# Data Dictionary
|Target Attribute	|Definition	|Data Type|
|------------------|-----------|----------|
|Churn	|whether or not the customer left the bank	|int |
		
|Feature	|Definition|Data Type|
|----------|----------|---------|
| Customer_Id	|contains random values and has no effect on customer leaving the bank	| int |
| CreditScore	|can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank	|int |
| Country	|customer’s location can affect their decision to leave the bank	| object |
| Gender	|it’s interesting to explore whether gender plays a role in a customer leaving the bank	| object |
|Age	|this is certainly relevant, since older customers are less likely to leave their bank than younger ones	|int|
|Tenure	|refers to the number of years that the customer has been a client of the bank. Normally, older clients are more loyal and less likely to leave a bank	|int|
|Balance|	also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to those with lower balances	|float|
|Products Number	|refers to the number of products that a customer has purchased through the bank	|int|
|Credit Card|	denotes whether or not a customer has a credit card. This column is also relevant, since people with a credit card are less likely to leave the bank	|int|
|Active Member|	active customers are less likely to leave the bank	|int|
|Estimated Salary	|as with balance, people with lower salaries are more likely to leave the bank compared to those with higher salaries	|float|
[Back to top]

Conclusion and Next Steps:

Conclusion:

Customer churn prediction is important to a long-term financial stability of banks. In this project, I successfully created a machine learning model - Decision Tree Classifier that was able to predict customer churn with an accuracy of 81% performing 2 percent better than the baseline's 79%.

We can conclude that of all the features age, country, gender and balance had a impact on customer churn.

Age: Customers at age 30-40 considered as the young are more likely to churn than oder ones, it is also this age bracket that has less than 600 credit score.

Country: Germany has the highest churn rate compared to other countries Spain and France.

Gender: Female customers are easier to churn than male customers.

Balance: Customers with high balances are more likely to churn.

Recommendations:

Age: We can see that younger customers are more likely to leave a bank because they are uneducated on building good credit and wealth. Banks should come up with loyalty and retention programs aimed at customers who can still be saved especially customers with poor credit scores given their age, or anyone with a credit score below 600.

Gender: Since banks are losing more female customers, they need to allocate more resources into pursuing female-oriented promotions such as offering rewards cards and points. Our data evidently shows female are likely to leave because they are bigger spenders and will tend to spend a lot to keep their lifestyles.

Next Steps:

My next step would be to find out how much did our y variable change over the period of 2 years to acquire the right information to do another research.
I would also like to do a research on Country - Germany and find out why it has the highest churn compared to the rest. Is it because of policies? Do banks have heavy competition?

I would recommend that in the future we research on age and find out why middle age adult are leaving and find ways to retain them.

Finally, I would love to check if my recommendations had an impact on our future data.

Reproduction Requirements:

The bank customer churn data was obtained from Kaggle open-source dataset. It is stored in a csv file, named as "Bank Customer Churn Prediction.csv". From kaggle(https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset). Using this data to predict bank customer churn.

1) Download the following files

Wrangle.py

Run the final_report.ipynb notebook

2) After downloading files make sure all files are in the same folder or location

3) Once step two and step one are done you would be able to run finalnotebook without errors

[Back to top]
