#####################################################
##  Week 3 Assignment - Multiple Linear and        ##
##                      Logistic Regression        ##
##  Due: 11/17/24                                  ##
#####################################################

library(MASS)
library(caret)
library(Metrics)

##You will be using the attached data files to run a multiple regression and a logistic regression analysis analysis

##Multiple regression
## Read in the data set called GaltonHeightData.txt (note this is a text file so read in accordingly)
GHData <- read.table("GaltonHeightData.txt", header = TRUE, sep = "")

##The column called "Family" is just an ID column and will cause issues with the analysis, so remove that here
GHData$Family <- NULL

## Let's use the machine learning technique of splitting the data into a training and test dataset
## Use 80% of the data for training and 20% for testing
in_train <- createDataPartition(GHData$Height, p=0.80, list=FALSE)
GHData_train <- GHData[in_train, ]
GHData_test <- GHData[-in_train, ]


## Create a multiple regression model with all of the features using the training dataset created above. 
## The dependent variable (the Y variable) will be height
full_model = lm(Height ~ ., data = GHData_train)


## In your script using hashtags, please interpret the results of your multiple regression focusing on the 
## adjusted R squared value of the overall model and the statistical significance of the features

# Father, Mother and Gender are significant predictors of height according to the P values being zero.
# Males are predicted to be taller than females when all else are equal.
# The number of kids in the family is not statistically significant.
# The adjusted R squared value of 0.6435 is close to the R squared value of 0.6454.
# This tells us that the model is not overfitted.
# There's a slight penalty for the number of features used in the model. 
# Removing the "Kids" predictor will likely improve the adj R squared number.

## Using the technique shown in the lecture, run a model selection procedure
# we will use the stepwise selection
optimal_model <- stepAIC(full_model, scope = ~ ., trace = TRUE)
summary(optimal_model)

## Interpret your model selection procedure focusing on whether all of the features remained in the model or not. 
## If any features were dropped from the optimal model, give an explanation of why you think this feature was removed.
# The Kids feature was dropped from the optimal model because it was not helpful in lowering the AIC score. 


## Finally, use your optimal model from your selection procedure and make predictions on your test dataset created above. 
## Combine the predicted data with the original data, and make a qualitative statement about how well your model is predicting.
predictions <- predict(optimal_model, newdata = GHData_test)

mae <- mean(abs(GHData_test$Height - predictions))
cat("Mean Absolute Error:", mae, "\n")

mse <- mse(GHData_test$Height, predictions)
cat("Mean Squared Error:", mse, "\n")

rmse <- rmse(GHData_test$Height, predictions)
cat("Root Mean Squared Error:", rmse, "\n")

# a Mean Absolute Error of 1.702 indicates that, on average, the predicted heights differ from the actual heights by approximately 1.7 inches.
# The MAE of the full model is 1.693. I was not expecting a lower MAE for the full model.
# Perhaps removing the Kids feature might have resulted in losing some information that was helpful for predicting height
# even though it didn't seem significant in the stepwise selection process?


##Logistic regression
## Read in the data set called YERockfish.csv
fishData <- read.table("YERockfish.csv", header = TRUE, sep = ",")

## We won't need the "date" or "stage" features, so you can remove those here
fishData$date <- NULL
fishData$stage <- NULL

## Background: In fisheries biology, we often use logistic regression to determine the probability of maturity status of fish species 
## based on their length. From this information we can set things like minimum size to harvest so that we maintain a sustainable population.

## Let's use the machine learning technique of splitting the data into a training and test dataset
## Use 80% of the data for training and 20% for testing
in_train <- createDataPartition(fishData$maturity, p=0.80, list=FALSE)
fish_train <- fishData[in_train, ]
fish_test <- fishData[-in_train, ]

## Create a logistic regression model with the "age" and "length" features using the training dataset created above. 
## The dependent variable (the Y variable) will be the numeric "maturity" column. Since there are only two features, swap the order of the features 
## in the regression formula and rerun the model. 
fishModel1 <- glm(maturity ~ length + age, family=binomial, data = fish_train)
summary(fishModel1)

fishModel2 <- glm(maturity ~ age + length, family=binomial, data = fish_train)
summary(fishModel2)

## In your script using hashtags, please interpret the results of your logistic regression focusing on the statistical significance of the features, 
## and comment on whether there was a difference in output between the two models that you ran.

# Based on the outcome, I see that length is more statistically significant than age due to 
# length's p-value of 0.00094 and age's p-value is much higher at 0.02980.
# The p-value for age (0.02980) is still below 0.05, meaning it is statistically significant, but it is not as impactful compared to length.
# I saw no difference in the AIC between the two models with the feature swap.


## Finally, use your model to make predictions on your test dataset created above. If you found a difference between the models, 
## use the one with the lowest AIC value (AIC is produced by the summary function). Create a confusion table and make a statement about 
## the accuracy of your model based on this outcome. Note: logistic regression creates probabilities between 0 and 1, so you need to set 
# a threshold to make something a 1 or a 0. Check out this link for a slick way to do this using the caret package: https://stackoverflow.com/questions/46028360/confusionmatrix-for-logistic-regression-in-r

# Predict probabilities of maturity on the test dataset
predicted_maturity <- predict(fishModel1, newdata = fish_test, type = "response")

# Convert probabilities into binary classifications (maturity = 1 if probability > 0.5)
predicted_classes <- ifelse(predicted_maturity > 0.5, 1, 0)

conf_matrix <- confusionMatrix(factor(predicted_classes), factor(fish_test$maturity))
print(conf_matrix)
