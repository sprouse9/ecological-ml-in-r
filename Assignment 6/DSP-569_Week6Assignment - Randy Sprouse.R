#####################################################
##  Week 6 Assignment - Artificial Neural Networks ##
##  Due: 12/8/24                                   ##
#####################################################


## We will use a dataset on possum morphometrics to see if we can determine its age.
## Data from: https://www.kaggle.com/datasets/abrambeyer/openintro-possum 



## You will need a series of packages, you can put them below. 
## Exactly which packages you need will depend on how you run the analysis (follow what was in the lecture or do another way, as long as it answers the question correctly, its fine).
library(dplyr)
library(tidyr)
library(caret)
library(neuralnet)


## Read in the data set and summarize the data. 
possum.dat <- read.csv("possum.csv", header = T)


## We want to work with just the numeric data so get rid of the columns case, site, pop, and sex, and remove rows with NAs
possum.dat <- possum.dat %>%
  select(-case, -site, -Pop, -sex) %>% 
  drop_na()


## Re-scale the data for use in the ANN
normalize <- function(x) { return ( (x-min(x))/(max(x)-min(x))) }
possum_n <- as.data.frame( lapply(possum.dat, normalize) )       # lapply() returns a list therefore we convert it to a data frame


## Split into training and testing data, make it an 80/20 split. Reminder, if you are randomizing anything in the following steps, please set a seed so I can reproduce your answer.
set.seed(60)
in_train <- createDataPartition(possum_n$age, p=0.80, list=FALSE)
train_data <- possum_n[in_train, ]
test_data <- possum_n[-in_train, ]


## Train a simple multilayer feedforward network with the default settings using only a single hidden node.
# The neuralnet function will automatically exclude age from the predictors when using the formula age ~ ...
possom_model <- neuralnet(age ~ hdlngth + skullw + totlngth + taill + footlgth + earconch + eye + chest + belly, 
                          data=train_data)


## Create a Visualization of the network
plot(possom_model)


## Evaluate your model by looking at the correlation between the predicted (or computed) ages versus the ages in the test dataset.
model_results <- compute(possom_model, test_data[, -which(names(test_data) == "age")])
predicted_strength <- model_results$net.result
cor(predicted_strength, test_data$age)

## Provide an interpretation of the evaluation
# I get a correlation of 0.475 with the default hidden layer of 1.
# This is a somewhat moderate correlation between the actual and predicted ages.
# The error is 1.509
# I'm inclined to say that the neural network is struggling with our small dataset of 104 rows.


## Try improving your ANN by adding two hidden layers of 6, and also change the activation function to a softplus activation function
softplus <- function(x) {log(1+exp(x))}

possom_improved_model <- neuralnet(age ~ hdlngth + skullw + totlngth + taill + footlgth + earconch + eye + chest + belly, 
                          data=train_data, hidden = c(6,6), act.fct = softplus)

model_results <- compute(possom_improved_model, test_data[, -which(names(test_data) == "age")])
predicted_strength <- model_results$net.result
cor(predicted_strength, test_data$age)


## Create a Visualization of this new network
plot(possom_improved_model)


## Evaluate your model by looking at the correlation between the predicted (or computed) ages versus the ages in the test dataset.
# correlation = 0.1972494
# our new correlation decreased significantly.

## Provide an interpretation of the evaluation and comment on if the increased complexity of the new neural network improved performance.
# The added complexity of the new neural network decreased our model's performance.
# The best correlation I was able to achieve was 0.52 with 5 hidden layers.


## If your neural network did not perform well, explain why you think that might be, and offer if there is another technique that we learned this semester that you think may have worked better on this particular dataset. Explain why you think this other approach may have worked better. 
# It seems neural networks do not perform well with small data sets.  
# I thought logistic regression would provide a better model due to its simplicity but the correlation I got was worse than the neural network.
# I tried KNN with k=10 and improved the correlation to 0.61.



# Fit a linear model for predicting age
log_reg_model <- glm(age ~ hdlngth + skullw + totlngth + taill + footlgth + earconch + eye + chest + belly, 
                     data = train_data, 
                     family = gaussian())  # gaussian family for continuous response variable

log_reg_model <- glm(age ~ belly, 
                     data = train_data, 
                     family = gaussian())  # gaussian family for continuous response variable


# View the summary of the model
summary(log_reg_model)

# Predict on test data
predicted_age_log_reg <- predict(log_reg_model, test_data)

# Calculate SSE for test set
sse_log_reg <- sum((predicted_age_log_reg - test_data$age)^2)
sse_log_reg

# Predict age using the logistic regression model
predicted_age_log_reg <- predict(log_reg_model, newdata = test_data)

# Calculate the correlation between the predicted age and actual age
cor_log_reg <- cor(predicted_age_log_reg, test_data$age)
cor_log_reg


# Load necessary package
library(FNN)

# Perform KNN regression (use features excluding 'age')
knn_reg_model <- knn.reg(train = train_data[, -1], test = test_data[, -1], y = train_data$age, k = 12)

# Predicted ages from the KNN regression model
predicted_age_knn <- knn_reg_model$pred

# Calculate the correlation between predicted and actual values
cor_knn <- cor(predicted_age_knn, test_data$age)
cor_knn


