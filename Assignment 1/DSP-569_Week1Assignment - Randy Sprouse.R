#####################################################
##  Week 1 Assignment - Data Wrangling and PCA     ##
##  Due:  11/03/2024                               ##
##  Randy Sprouse                                  ##
#####################################################

# You will be using the attached csv files to run a PCA analysis

# You will need a series of packages, you can put them below. 
# Exactly which packages you need will depend on how you run the analysis (follow what was
# in the lecture or do another way, as long as its correct, its fine).

library(tidyverse)
library(flextable)
library(ggcorrplot)
library(factoextra)


# Read in the data sets
data1<-read_csv("penguins1.csv")
data2<-read_csv("penguins2.csv")

# When using read_csv() the data will imported into a tibble rather than a standard data frame. 
# Tibbles are part of the tidyverse package, which enhances data frames in R by providing additional 
# features like printing more user-friendly summaries. R has a built in function read.csv()

# The data both represent unique rows of the same dataset, so you can combine them by 
# their columns to run your analysis. Use a data wrangling technique to combine the two datasets
full_data<-rbind(data1, data2)

# Take a look at the data to make sure your wrangling worked. Put your code for how you review below. 

print(nrow(data1))
print(nrow(data2))
print(nrow(full_data))

if( nrow(data1)+nrow(data2) == nrow(full_data) )
    print("The datasets have been succesfully merged")

str(full_data)

# One of the columns won't work for a PCA, remove that column before proceeding.
quant.dat <- full_data
quant.dat$sex <- NULL     # specify the column to remove by name

# Let's remove any missing values or negative values since those wouldn't make sense for our data.
# For each row we check all columns. If a neg or NA is found, the all() function will return FALSE.
# We store only those row indexes that returned TRUE
quant.dat <- quant.dat[apply(quant.dat, 1, function(x) all(!is.na(x) & x>=0)),  ]     # Note: col entry is left blank


# Take a look at the data to see that it worked. Put your code for how you review below.
print(head(quant.dat))    # wouldn't display the data without the print()

# For a PCA to run without being biased by the magnitude of the data on different scales, 
# you need to scale your data. Please do that here.
scaled.dat = data.frame(scale(quant.dat))
head(scaled.dat)

# PCA works best when some of the data is strongly correlated. Run a correlation analysis 
# on your data and use a visualization to see if there are pairs of data with strong 
# correlations.
corr_matrix <- cor(scaled.dat)
ggcorrplot(corr_matrix)

# Run your PCA and then summarize the output
data.pca <- princomp(corr_matrix)
summary(data.pca)

data.pca$loadings[, 1:2]

# Create a scree plot of the PCA
fviz_eig(data.pca, addlabels=TRUE)


# Create a Biplot of your PCA
fviz_pca_var(data.pca, col.var="cos2", 
             gradient.cols=c("black", "orange", "green"),
             repel=TRUE)


# In your script using hashtags, please interpret the results of you PCA. You don't need 
# to write a dissertation, but a few sentences about the major outcome of your analysis
# will suffice.

# The importance of components summary tells us that the first 2 components explain
# 96% of the variance in the data. Components 3 has a low standard deviation 
# and doesn't have as much of an impact. Component 4 does not contribute to the variance at all.
# Base on this analysis, I can say that reducing the dimensionality to the first two
# principal components is justified without loosing significant information.
# The culmen_length_mm has a positive loading of 0.4952 indicating that it contributes
# positively to this component 1 while culmen_depth_mm has a negative loading of -0.6004
# indicating an inverse relationship with component 1.
# Component 2 seems to be driven primarily by flipper length in an opposite direction to
# culmen depth and length. This component might represent a factor contrasting flipper length with culmen dimensions.