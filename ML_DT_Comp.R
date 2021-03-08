getwd()
setwd("~/Dev/PDev/Mashup_Vid_Processing/metadados")

mashup_v <- read.csv("Mashup_Geral_Data_24b.csv")
mashup_v <- na.omit(mashup_v)
teste <- read.csv("24_B.csv")


library(rpart)
library(randomForest)
library(rpart.plot)
library(caret)
library(e1071)
library(ipred)
library(Metrics)
library(MLmetrics)
library(gbm)

str(mashup_v)
names(mashup_v)
head(mashup_v)

#Train/test split

# Total number of rows
n <- nrow(mashup_v)
# Number of rows for the training set (80% of the dataset)
n_train <- round(0.80 * n)

#Create a vector of indices which is an 80% random sample

set.seed(123)
train_indices <- sample(1:n, n_train)

# Subset the credit data frame to training indices only
v_train <- mashup_v[train_indices, ]  

# Exclude the training indices to create the test set
v_test <- mashup_v[-train_indices, ]  

# Train the model 
v_model <- rpart(formula = Frame ~ Cut + VideoID_N+ Thematic_Content+ Music_Sec_N +Music_Sec + hue_stdev +  brightness_stdev + saturation_stdev,
                   data = v_train, 
                   method = "class")
print(v_model)

# Generate predicted classes using the model object
class_prediction <- predict(object = v_model,  
                            newdata = v_test,  
                            type = "class")       

# Calculate the confusion matrix for the test set
confusionMatrix(data = class_prediction,         
                reference = v_test$Frame)  

###Compare models with a different splitting criterion###

"shape_median"    

# Train a gini-based model +add after "=" values
v_model_gini <- rpart(formula = Frame ~ Cut + VideoID_N+ Thematic_Content+ Music_Sec_N +Music_Sec + hue_stdev +  brightness_stdev + saturation_stdev,
                        data = v_train, 
                        method = "class",
                        parms = list(split = "gini"))

# Train an information-based model
v_model_info <- rpart(formula = Frame ~ Cut + VideoID_N+ Thematic_Content+ Music_Sec_N +Music_Sec + hue_stdev +  brightness_stdev + saturation_stdev
                        data = v_train, 
                        method = "class",
                        parms = list(split = "information"))


# Generate predictions on the validation set using the gini model
pred1 <- predict(object = v_model_gini,
                 newdata = v_test,
                 type = "class")    

# Generate predictions on the validation set using the information model
pred2 <- predict(object = v_model_info, 
                 newdata = v_test,
                 type = "class")

# Compare classification error

mean(v_test$Frame == pred1)
mean(v_test$Frame == pred2)

plotcp(v_model_gini)


?bagging

v_model_gini2 <- bagging(formula = Frame ~ Cut + VideoID_N+ Thematic_Content+ Music_Sec_N +Music_Sec + hue_stdev +  brightness_stdev + saturation_stdev,
                           data = v_train, 
                           method = "class",
                           coob = TRUE)

print(v_model_gini2)

pred3 <- predict(object = v_model_gini2,
                 newdata = v_test,
                 type = "class")    

# Print the predicted classes
print(pred3)


confusionMatrix(data = pred3,         
                reference = v_test$Frame)

#Predict on a test set 
pred <- predict(object = v_model_gini2,
                newdata = v_test,
                type = "prob")  

# `pred` is a matrix
class(pred)

# Look at the pred format
head(pred) 

########################

#Train a Random Forest model#

set.seed(1)  # for reproducibility
v_forest <- randomForest(Frame ~ Cut + VideoID_N+ Thematic_Content+ Music_Sec_N +Music_Sec + hue_stdev +  brightness_stdev + saturation_stdev,
                           data = v_train)

print(v_forest)

#Evaluate out-of-bag error
# Grab OOB error matrix & take a look
err <- v_forest$err.rate
head(err)

# Look at final OOB error rate (last row in err matrix)
oob_err <- err[nrow(err), "OOB"]
print(oob_err)

# Plot the model trained in the previous exercise
plot(v_forest)

# Add a legend since it doesn't have one by default
legend(x = "right", 
       legend = colnames(err),
       fill = 1:ncol(err))

# Generate predicted classes using the model object
class_prediction <- predict(object = v_forest,  # model object 
                            newdata = v_test,  # test dataset
                            type = "class")         # return classification labels

# Calculate the confusion matrix for the test set
cm <- confusionMatrix(data = class_prediction,          # predicted classes
                      reference = v_test$Frame)  # actual classes
print(cm)

# Compare test set accuracy to OOB accuracy
paste0("Test Accuracy: ", cm$overall[1])
paste0("OOB Accuracy: ", 1 - oob_err)


#########

#?gbm

set.seed(1)
v_model3 <- gbm(Frame ~ Cut + VideoID_N+ Thematic_Content+ Music_Sec_N +Music_Sec + hue_stdev +  brightness_stdev + saturation_stdev,
                  distribution = "multinomial", 
                  data = v_train,
                  n.trees = 10000)

# Print the model object                    
print(v_model3) 

# summary() prints variable importance
summary(v_model3)  

# Generate predictions on the test set
preds4 <- predict(object = v_model3, 
                  newdata = v_test,
                  n.trees = 10000)

# Generate predictions on the test set (scale to response)
preds5 <- predict(object = v_model3, 
                  newdata = v_test,
                  n.trees = 10000,
                  type = "response")

# Compare the range of the two sets of predictions
range(preds4)
range(preds5)


#
# Optimal ntree estimate based on OOB
ntree_opt_oob <- gbm.perf(object = v_model3, 
                          method = "OOB", 
                          oobag.curve = TRUE)

preds6 <- predict(object = v_model3, 
                  newdata = v_test,
                  n.trees = ntree_opt_oob,
                  type = "response")