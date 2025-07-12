#####################################################
##  Week 4 Assignment - KNN, Decision Trees, and   ##
##                       Random Forest             ##
##  Due: 11/24/24                                  ##
#####################################################


##You will be using the attached data files to run a K-Nearest Neighbors analysis, a decision tree analsysis, and a random forest analysis

## K-NN
## Read in the data set called Pumpkin_Seeds_Dataset.csv
##Citation for the data:          https://www.muratkoklu.com/datasets/
                                  #KOKLU, M., SARIGIL, S., & OZBEK, O. (2021). The use of machine learning methods in classification of pumpkin seeds (Cucurbita pepo L.). Genetic Resources and Crop Evolution, 68(7), 2713-2726. Doi: https://doi.org/10.1007/s10722-021-01226-0
                                  #https://link.springer.com/article/10.1007/s10722-021-01226-0
                                  #https://link.springer.com/content/pdf/10.1007/s10722-021-01226-0.pdf
library(tidyverse)
library(class)
library(gmodels)
library(OneR)
library(C50)
library(randomForest)

pumpkin_seeds.dat <- read_csv("Pumpkin_Seeds_Dataset.csv")


## The target for this analysis is the column called class (don't forget to convert to a factor).
pumpkin_seeds.dat$Class <- as.factor(pumpkin_seeds.dat$Class)

## You must normalize the other columns as they are scaled very differently in many cases.
normalize <- function(x) { return ( (x-min(x))/(max(x)-min(x))) }

pSeeds_n <- as.data.frame(lapply(pumpkin_seeds.dat[, !names(pumpkin_seeds.dat) %in% "Class"], normalize))

# at this point pSeeds_n does not contain the "Class" column since it was dropped when we normalized


# pseeds_n <- as.data.frame(lapply(pumpkin_seeds.dat[1:12], normalize))  # remove "Class" by column numbers
# pSeeds_n$Class <- pumpkin_seeds.dat$Class

## Let's use the machine learning technique of splitting the data into a training and test dataset
## Use 75% of the data for training and 25% for testing.
## Important note: These data are not randomly arranged so you must randomly split the dataset up. Please set your seed so I can reproduce.
set.seed(60)
in_train <- createDataPartition(pumpkin_seeds.dat$Class, p=0.75, list=FALSE)
pSeeds_train <- pSeeds_n[in_train, ]
pSeeds_test  <- pSeeds_n[-in_train, ]

pSeeds_train_labels <- pumpkin_seeds.dat[in_train, "Class", drop=TRUE]
pSeeds_test_labels  <- pumpkin_seeds.dat[-in_train, "Class", drop=TRUE]


## Run a K-NN analysis using the training dataset created above.
k <- round(sqrt(nrow(pSeeds_n)))
pumpkinSeeds_pred <- knn(train=pSeeds_train, test=pSeeds_test, cl=pSeeds_train_labels, k=k)


##Evaluate your model using the CrossTable procedure shown in the lecture.
CrossTable(x = pSeeds_test_labels, y=pumpkinSeeds_pred, prop.chisq = FALSE)


## In your script using hashtags, please interpret the results of your K-NN analysis
# 293 out of 325 seeds were correctly labeled as Cercevelik
# 250 out of 300 seeds were correctly labeled as Urgup_Sivrisi
# Our initial model's overall accuracy is (293+250)/625 ≈ 87%
# The model performs well overall but has room for improvement. We need to reduce the 13% misclassification rate.

## Investigate some other values of k and see if any improve the model performance. Look at at least 5 different k values. 
## Please describe these results and you may plot them as well.
k_values <- c(1, 5, 11, 15, 21, 27, 35, 40, 45)

my_vector <- vector(mode = "numeric")

for(k_val in k_values) {
  pumpkinSeeds_pred <- knn(train=pSeeds_train, test=pSeeds_test, cl=pSeeds_train_labels, k=k_val)
  
  Table = CrossTable(x = pSeeds_test_labels, y=pumpkinSeeds_pred, prop.chisq = FALSE)
  
  values = (Table$t[1,2] + Table$t[2,1]) # add the misclassified observations
  my_vector <- append(my_vector, values)
  
}

# Plot k values vs. number of errors
plot(k_values, my_vector, type = "b", col = "blue", pch = 16,
     xlab = "k (Number of Neighbors)", ylab = "Number of Errors",
     main = "Number of Errors vs. k")

## There was a steep decline in errors when k went from 1 to 15.
## 15 seems to the lowest error I could achieve for my model.



## Decision Tree
## Read in the data set called mushrooms.csv. A description of this dataset can be found in your textbook.
mushrooms.dat <- read.csv("mushrooms.csv", stringsAsFactors = TRUE)


## There is a column veil_type that only has a single level in it. This is a recording error in the dataset, so please remove this column before proceeding with the analysis.
mushrooms.dat$veil_type <- NULL

## Let's use the machine learning technique of splitting the data into a training and test dataset
## Use 75% of the data for training and 25% for testing
set.seed(444)
in_train <- sample(1:nrow(mushrooms.dat), size = 0.75 * nrow(mushrooms.dat))
mushrooms_train <- mushrooms.dat[in_train, ]
mushrooms_test <- mushrooms.dat[-in_train, ]

## Run a decision tree analysis on this dataset, using the C5.0 technique shown in the lecture.
mushroom_model <- C5.0(type ~ ., data=mushrooms_train)
summary(mushroom_model)

## Evaluate your model using the CrossTable approach shown in the lecture, but also run the Kappa statistic so comparisons can be made later.
mushroom_pred <- predict(mushroom_model, mushrooms_test)

CrossTable(x = mushrooms_test$type, y = mushroom_pred, prop.chisq = FALSE)

confusion_matrix <- confusionMatrix(data = mushroom_pred, reference = mushrooms_test$type)
print(confusion_matrix)


## See if you can improve the performance of your tree by boosting.
mushroom_model_boost10 <- C5.0(type ~ ., data=mushrooms_train, trials=10)
summary(mushroom_model_boost10)

## Again, evaluate by using the CrossTable and Kappa statistic.
boosted_mushroom_pred <- predict(mushroom_model_boost10, mushrooms_test)
CrossTable(x = mushrooms_test$type, y = boosted_mushroom_pred, prop.chisq = FALSE)
confusion_matrix <- confusionMatrix(data = boosted_mushroom_pred, reference = mushrooms_test$type)
print(confusion_matrix)

## Seems the accuracy went up and now kappa = 1



## Random Forest
## Using the same dataset, run a random forest analysis using the random forest function shown in the lecture
rf <- randomForest(type ~ ., data = mushrooms.dat)


## Evaluate your RF model using the Kappa statistic
rf_predictions <- predict(rf, newdata = mushrooms.dat)
conf_matrix <- confusionMatrix(rf_predictions, mushrooms.dat$type)
conf_matrix$overall["Kappa"]

## Finally, describe in words the outcome of the decision tree, the boosted tree, and the random forest models by comparing the kappa statistics across the methods.
## The decision tree model built using the C5.0 technique produced a Kappa of 1. This indicates perfect agreement between the predicted and actual classifications.
## After boosting the decision tree the Kappa statistic remained at 1.
## The random forest model also produced a Kappa statistic of 1.

