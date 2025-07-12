#####################################################
##  Week 7 Assignment - Deep Learning              ##
##  Due: 12/15/24                                  ##
#####################################################


## We will use a dataset on mosquitos to see if we can teach our computers to identify two species. I thought this would be of interest because it cross cuts both ecology and public health. Mosquito's are known vectors of disease, and the way we monitor for potential exposure to West Nile Virus and Eastern Equine Encephalitis is through trapping mosquitos, identifying those mosquitos (currently done by humans), and then processing for the presence of the diseases in the sample. So, if we could automate the identification process, this could increase the throughput of the information, thus enhancing the benefits to public health.
## Data from: https://www.kaggle.com/datasets/pradeepisawasan/aedes-mosquitos 

## You will need a series of packages, you can put them below. 
##Call in needed packages (recall the challenges I mentioned in the lecture with the package set up)

library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)


#use_virtualenv("~/tf_keras_env", required = TRUE)

# Now install the TensorFlow R package
#install_tensorflow(extra_packages="pillow")


tf$version$VERSION     # version: 2.16.2
#keras::keras_version() # version: 3.7.0
reticulate::py_run_string("import keras; print(keras.__version__)") # version: 3.7.0
py_config()            # version: 3.9.9

#reticulate::install_python("3.10:latest")

#install.packages("keras")  # Install the R package
#install_keras()     # Install TensorFlow and Keras in Python

#tf$constant("Hello, TensorFlow!") # the tf variable is automatically created when you load the tensorflow package in R


##Set up your working directory and do so in a way that gets the species labels from the folder names. 
##I have set up the data like shown in the lecture so the pictures for each species are in a folder with the species name
setwd("~/Desktop/DSP 569/Assignment 7/Mosquitos")
label_list <- dir("train")  # this retrieves "aegypti" "albopictus"
output_n <- length(label_list)
save(label_list, file="label_list.R")


##Set up the picture size and the number of color channels. The specs set up in the lecture can work for this exercise as well.
width  <- 224
height <- 224

target_size <- c(width, height)
rgb <- 3  # color channels


##Define the preprocessing of the data. You can keep things simple by just rescaling the pixel values to values between 0 and 1. Remember to reserve 20% of the data for a validation dataset.
path_train <- "~/Desktop/DSP 569/Assignment 7/Mosquitos/train"
train_data_gen <- image_data_generator(rescale=1/255, validation_split = .2)



##Batch-process the images. Assign the folder names in your “train” folder as the class labels. Create two objects for the training and validation data, respectively. Remember we are classifying our data as categorical.
train_images <- flow_images_from_directory(path_train, train_data_gen, 
                                           subset = 'training', 
                                           target_size = target_size, 
                                           class_mode = "categorical", 
                                           shuffle = F, 
                                           classes = label_list, 
                                           seed = 2024)

validation_images <- flow_images_from_directory(path_train, train_data_gen, 
                                           subset = 'validation', 
                                           target_size = target_size, 
                                           class_mode = "categorical", 
                                           shuffle = F, 
                                           classes = label_list, 
                                           seed = 2024)

##Check to see that it is working correctly. You can create a table and/or pull out a sample picture like shown in the lecture to achieve this.
table(train_images$classes)

plot(as.raster(train_images[[1]][[1]][17,,,]))


##Use a pre-trained model, namely the xception-network, like done in the lecture. Load the xception-network with the weights pre-trained on the ImageNet dataset except for the final layer (which classifies the images), and train that final layer on the mosquito dataset.
mod_base <- application_xception(weights="imagenet",
                                 include_top = FALSE,
                                 input_shape = c(width, height, 3))

freeze_weights(mod_base)

##Now write a small function that builds a layer on top of the pre-trained network. When building the function, set the learning rate, drop out rate, and n_dense parameters to variables so you can test some different values to improve your model.
model_function <- function(mod_base, learning_rate = 0.001, dropoutrate = 0.2, n_dense = 1024) {
  k_clear_session()
  
  # Add custom layers on top of the pre-trained base
  outputs <- mod_base$output %>%
    layer_global_average_pooling_2d() %>%
    layer_dense(units = n_dense, activation = "relu") %>%
    layer_dropout(rate = dropoutrate) %>%
    layer_dense(units = output_n, activation = "softmax")
  
  # Create the complete model
  model <- keras_model(inputs = mod_base$input, outputs = outputs)
  
  class(model) <- c("keras.engine.training.Model", class(model))
  
  # Compile the model
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = c("accuracy")
  )
  
  return(model)
}


model <- model_function(mod_base)



##Train the model using the existing architecture. I recommend using the same number of epochs and same batch size as in the lecture. 
batch_size <- 32
epochs <- 6

hist <- model %>% 
  fit(train_images, 
                steps_per_epoch = train_images$n %/% batch_size,
                epochs = epochs,
                validation_data = validation_images,
                validation_steps = validation_images$n %/% batch_size,
                verbose = 2)



##Test the model you just built on the test dataset, remembering to pre-process the data in the same way that you did the training data.
path_test <- "~/Desktop/DSP 569/Assignment 7/Mosquitos/test"
test_data_gen <- image_data_generator(rescale = 1/255)

test_images <- flow_images_from_directory(path_test, 
                                          test_data_gen, 
                                          target_size = target_size,
                                          class_mode = "categorical",
                                          classes = label_list,
                                          shuffle = F, seed = 2021)

model %>% evaluate_generator(test_images,
                             steps = test_images$n)

## results: accuracy: 0.8556 - loss: 0.3365



##Create a plot that ranks how well the model did on the test categories.
predictions <- model %>% 
  predict_generator( generator = test_images, 
                     steps = test_images$n) %>% as.data.frame

names(predictions) <- paste0("Class",0:1)

predictions$predicted_class <- paste0("Class",
                                        apply(predictions,1,which.max)-1)

predictions$true_class <- paste0("Class",
                                 test_images$classes)

predictions %>% 
  group_by(true_class) %>% 
  summarise(percentage_true = 100*sum(predicted_class == true_class)/n()) %>%
  left_join(data.frame(mosquito_species= names(test_images$class_indices),
                       true_class=paste0("Class",0:1)),by="true_class") %>%
  select(mosquito_species, percentage_true) %>%
  mutate(mosquito_species = fct_reorder(mosquito_species, percentage_true)) %>%
  
  ggplot(aes(x=mosquito_species,y=percentage_true,fill=percentage_true, label=percentage_true)) +
  geom_col() + theme_minimal() + coord_flip() + geom_text(nudge_y = 3) +
  ggtitle("Percentage correct classifications by mosquito species")















##Use the tuning procedure shown in the lecture to see if you can improve the accuracy of the model. 
## Warning: This dataset is much smaller than that used for the lecture, so this shouldn't take nearly as long to run, but it still make take some time depending on the computer you are using.
tune_grid <- data.frame("learning_rate" = c(0.001, 0.0001), 
                        "dropoutrate" = c(0.3, 0.2), 
                        "n_dense" = c(1024, 256))

tuning_results <- NULL

set.seed(2024)


for (i in 1:length(tune_grid$learning_rate)){
  for (j in 1:length(tune_grid$dropoutrate)){
    for (k in 1:length(tune_grid$n_dense)){
      
      model <- model_function(mod_base = mod_base, learning_rate = tune_grid$learning_rate[i],
                              dropoutrate = tune_grid$dropoutrate[j], n_dense = tune_grid$n_dense[k])

      hist <- model %>% fit_generator(train_images,
                                      steps_per_epoch = train_images$n %/% batch_size,
                                      epochs = epochs, validation_data = validation_images,
                                      validation_steps = validation_images$n %/%batch_size, verbose = 2)

      #Save model configurations
      tuning_results <- rbind(tuning_results, c("learning_rate" = tune_grid$learning_rate[i],
                                                "dropoutrate" = tune_grid$dropoutrate[j], "n_dense" = tune_grid$n_dense[k],
                                                "val_accuracy" = hist$metrics$val_accuracy))
    }
  }
}

tuning_results

best_results <- tuning_results[which(tuning_results[,ncol(tuning_results)] ==
                                       max(tuning_results[,ncol(tuning_results)])),]

best_results


##Summarize the model configuration that gives you the best results.

# our initial model had parameters:               
#   learning_rate = 0.001, dropoutrate = 0.2, n_dense = 1024
# with a validation accuracy (epoch 6) of 0.8556 with a loss of 0.3365.


# the best tuned model yielded when attempting to tune: 
#   learning_rate = 0.001, dropoutrate = 0.2, n_dense = 256
# with a validation accuracy (epoch 6) of 0.772 

# Reducing the number of dense units from 1024 to 256 resulted in a simpler model with a higher validation accuracy.
# The learning rate and dropout rates seemed to be the optimal values.


