#####################################################
##  Week 2 Assignment - More Data Wrangling        ##
##  Due: 11/10/24                                  ##
##  Randy Sprouse                                  ##
#####################################################


## We will load a dataset on wildlife strikes from github and work on some more data wrangling this week.
## Questions adapted from: https://p4a.jhelvy.com/data-wrangling 

## You will need a series of packages, you can put them below. 
## Exactly which packages you need will depend on how you run the analysis (follow what was
## in the lecture or do another way, as long as it answers the question correctly, its fine).

library(tidyverse)
library(caret)
library(mltools)
library(data.table)
library(pROC)
library(vcd)

## Read in the data set and summarize the data. I have left the code in for where the data is online, but you will need to make it into an object you can work with in your R environment, and then summarize it.
download.file(
  url = "https://github.com/jhelvy/p4a/raw/main/data/wildlife_impacts.csv",
  destfile = file.path('wildlife_impacts.csv')
)
wildlife.dat <- read_csv("wildlife_impacts.csv")


## There are 21 columns in the dataset. Please subset the data to create a new data object with 
## only the following columns: state, species, damage, time_of_day, and phase_of_flt 
wildlife_subset <- wildlife.dat %>%
  select(state, species, damage, time_of_day, phase_of_flt)


## From your new dataset, remove the phase_of_flt column
wildlife_subset$phase_of_flt <- NULL


## Going back to the full dataset, filter the rows for wildlife impacts that cost more than $0.5 million in damages. 
## These values can be found in the cost_repairs_infl_adj column.
wildlife.dat %>%
  filter(cost_repairs_infl_adj > 500000)


## Use a pipe operator on the original dataset and filter for the state of RI, then select the columns species, airport, and time_of_day
wildlife.dat %>%
  filter(state == RI) %>%
  select(species, airport, time_of_day)


## Sort the original dataset by the speed column in decending order
wildlife.dat %>%
  arrange( desc(speed) )


## Create a new column that turns the height column (currently recorded in feet) into miles by dividing the height in each cell by 5280, and filter out the NAs (use pipes).
wildlife.dat %>%
  filter(!is.na(height)) %>%
  mutate(height=height/5280)


##Finally, add a new column that computes the mean height of reported wildlife impacts for each state, and start by removing the NAs.
## This will add a new column "mean_height" to the existing table. The drawback is that the mean height will be a repeated for each state.
wildlife.dat %>% filter(!is.na(height), !is.na(state) & state != "N/A" & state != "") %>% 
  group_by(state) %>%
  mutate(mean_height = mean(height))

## This version of the code will create a table with 2 columns: state and mean_height
wildlife.dat %>% filter(!is.na(height), !is.na(state) & state != "N/A" & state != "") %>% 
  group_by(state) %>%
  summarize(mean_height = mean(height))


##Moving to the simMat_dat.csv dataset, lets calculate some evaluation metrics. Start by creating a confusion matrix. You can use whatever function/method you want as long as it produces a confusion table correctly. 
simMat_.dat <- read_csv("simMat_dat.csv")

## remove any rows with NA or those with a negative probability
simMat.dat[apply(simMat.dat, 1, function(x) all(!is.na(x) & x>=0)),  ]

# Ensure both columns are factors with the same levels
simMat.dat$Predicted <- factor(simMat.dat$Predicted, levels = c("Immature", "Mature"))
simMat.dat$Actual    <- factor(simMat.dat$Actual, levels    = c("Immature", "Mature"))


# Generate a confusion matrix
conf_matrix <- confusionMatrix(simMat.dat$Predicted, simMat.dat$Actual)

# Print the confusion matrix and additional metrics
print(conf_matrix)


##compute Kappa (reminder that the base package kappa() function is a different calculation).
Kappa(conf_matrix$table)
print(conf_matrix$overall['Kappa']) # another way to get the Kappa statistic


##Compute MCC
simMat_dt <- as.data.table(simMat.dat)
#simMat_dt$Predicted <- factor(simMat_dt$Predicted, levels = c("Immature", "Mature"))
#simMat_dt$Actual    <- factor(simMat_dt$Actual, levels    = c("Immature", "Mature"))
mcc_value <- mcc(simMat_dt$Actual, simMat_dt$Predicted)
print(mcc_value)

# find MCC another way
TN <- as.numeric(conf_matrix$table[1, 1])  # True Negatives
FP <- as.numeric(conf_matrix$table[1, 2])  # False Positives
FN <- as.numeric(conf_matrix$table[2, 1])  # False Negatives
TP <- as.numeric(conf_matrix$table[2, 2])  # True Positives

print((TP*TN - FP*FN) / sqrt((TP+FP)*(TP + FN)*(TN + FP)*(TN + FN)))


##Create a ROC curve for this dataset
simMat_roc <- roc(response = as.numeric(simMat.dat$Actual == "Mature"), 
                  predictor= as.numeric(simMat.dat$Predicted == "Mature"))

plot(simMat_roc, main="test", 
     col="blue", lwd=2, grid=TRUE, legacy.axes=TRUE)



