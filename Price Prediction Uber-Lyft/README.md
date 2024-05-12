
# Uber Lyft Price Prediction Model

Ridesharing services such as Uber and Lyft have become revolutionary in the ever-changing urban transportation market, providing millions of users with flexible and convenient mobility options.


<p>
<img src="https://github.com/goeludit/Data-Science/blob/main/Price%20Prediction%20Uber-Lyft/Images/uber-logo.jpg?raw=true" alt="Girl in a jacket" height="150">
<img src="https://github.com/goeludit/Data-Science/blob/main/Price%20Prediction%20Uber-Lyft/Images/Emblem-Lyft.jpg?raw=true" alt="Girl in a jacket" height="150" style="vertical-align: top;">

</p>

# Objective
The objective of this project is to develop a robust and accurate price prediction model for ridesharing services, specifically targeting Uber and Lyft. The goal is to create a model that can estimate the fare for a ride based on various factors such as distance, time, location, and potential surge pricing for the Boston dataset. 


# Model Used
- Ridge and Lasso Regression
- Random Forest
- XG Boost



# Optimization and Hyper-Parameter Tuning
Bagging regressor was applied to XGboost to resample the data and run the simulations with 5-Fold Cross validation using Sci-kit's GridSearchCV. Tuning params were as below:

```
tuning_params = {
"n_estimators": [x for x in range(50, 400, 50)],
"max_depth": [3,6,9],
"gamma": [0.01, .1],
"learning_rate": [.001, .01, .1, 1]
}
```


The led to increase in accuracy by 2% resulting into total accuracy of 96.9%. 

# Author
Udit Goel, \
Masters in Data Science @ Rutgers University, New Brunswick \
[https://www.linkedin.com/feed/](https://www.linkedin.com/in/goeludit7rma/)
