#####################################################
##  Week 5 Assignment - Natural Language Processing##
##  Due: 12/1/24                                   ##
#####################################################


## You will be using the attached data file to run a classification task using NLP
## Read in the data set called NLP_data2.csv
## Citation for the data is in the lecture slides.

#install.packages("data.table")
library(data.table)

#install.packages("stringr")
library(stringr)

#install.packages("tm")
library(tm)

#install.packages("slam")
library(slam)

#install.packages("e1071")
library(e1071)

#install.packages("tidyverse")
library(tidyverse)
library(readr)

#install.packages("data.table")
library(data.table)

nlp_dat = read_csv("NLP_data2.csv")

## Please set your seed so I can reproduce.
set.seed(60)
## To resolve memory issues, subset the data to 1,000 random rows like I did in the lecture
sample_dat = sample(nrow(nlp_dat), 1000)
nlp_dat2 = nlp_dat[sample_dat, ]
## You will need to preprocess (remove: uppercase letters, punctuation, numbers, and stop words) and then vectorize the data (don't forget to convert your matrix into a data table and add your labels back to the dataset).
##The classifications are the diseases in the Labels column (don't forget to convert to a factor).
## Note, you will see a series of warnings as you run this code, these are warnings and can be ignored as we are not using document names in our analysis, so no documents will be dropped.
corpus = Corpus(VectorSource(nlp_dat2$Abstract))

corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removeWords, stopwords("SMART"))
corpus = tm_map(corpus, stripWhitespace)

labels = as.numeric(factor(nlp_dat2$Label))-1
#labels = labels - 1

# Vectorization
dtm <- DocumentTermMatrix(corpus)
matrix <- as.matrix(dtm)

dtm_df <-as.data.table(matrix)
dtm_df$label <- labels

##We will start with a bag-of-words model
## Let's use the machine learning technique of splitting the data into a training and test dataset
## Use 80% of the data for training and 20% for testing. 
split_index <- sample(1:nrow(dtm_df), 0.8 * nrow(dtm_df))
train_set <- dtm_df[split_index]
test_set <- dtm_df[-split_index]

## Use the support vector matrix approach I used in the lecture to create the model to classify your data.  
## Another note, you will get another warning here (quite extensive) and it is due to their being a bunch of 
## rarely occurring words, which show up as zeroes in the training data. It is just a warning, the model will still work for the rest of the assignment.
model <- svm(label ~., data=train_set, kernel="linear")

predictions <- predict(model, newdata=test_set[, -"label"])

## Evaluate your model by using it to predict on the test dataset, set a threshold level (0.5), covert these to binary 
## predictors, and use a confusion matrix to calculate the accuracy of our model on the test dataset.
threshold <- 0.5
binary_predictions <- ifelse(predictions > threshold, 1, 0)
confusion_matrix <- table(binary_predictions, test_set$label)
#print(confusion_matrix)

accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
#print(paste("Accuracy:", accuracy)). # we will print later

## Using the same dataset, covert from a bag-of-words process to a TF-IDF process, and comment on the comparison of the performance between the two approaches. 
tfidf <- weightTfIdf(dtm)
tfidf

matrix2 <- as.matrix(tfidf)
dtm_df2 <- as.data.table(matrix2)
dtm_df2$label <- labels

split_index <- sample(1:nrow(dtm_df2), 0.8 * nrow(dtm_df2))
train_set2 <- dtm_df2[split_index]
test_set2 <- dtm_df2[-split_index]

model2 <- svm(label ~., data=train_set2, kernel="linear")
predictions2 <- predict(model2, newdata=test_set2[, -"label"])

binary_predictions2 <- ifelse(predictions2 > threshold, 1, 0)
confusion_matrix2 <- table(binary_predictions2, test_set2$label)

accuracy2 <- sum(diag(confusion_matrix2)) / sum(confusion_matrix2)

print(paste("Accuracy:", accuracy))
print(confusion_matrix)

print(paste("Accuracy2:", accuracy2))
print(confusion_matrix2)

## The accuracy increased by 1% when using TF-IDF.
## The TP and TN both increased by 1.
## Although the FP decreased by 4, the FN had a slight increase of 2.
## Both techniques perform well, but TF-IDF is slightly better in this case. 
