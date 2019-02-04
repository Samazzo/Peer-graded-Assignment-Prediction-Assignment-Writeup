library(caret)
library(corrplot)
library(randomForest)
library(rattle)
library(rpart)
# I split the trainingset into a training and testing part using the proportion 0.7/0.3 as learned this course
# This I can use to predict "pml-testing".
training_data <- read.csv("c:/Users/auke.beeksma/Desktop/pml-training.csv")


intrain <- createDataPartition(training_data$classe, p = .7, list= FALSE)
training <-training_data[intrain,] 
testing <- training_data[-intrain,]
prediction_data <-  read.csv("c:/Users/auke.beeksma/Desktop/pml-testing.csv")

# Remove variables which don't add variance
nsv <- nearZeroVar(training)
training <- (training[, - nsv])
testing <- (testing[, - nsv])
# Remove variables with no prediction value. i.e identificationcodes (X, user_name, timestamps)
training <- (training[, - (1:5)])
testing <- (testing[, - (1:5)])
# There are many columns with a lot of NA's
# Only work with columns with more then 50% percent is not NA.
training <- training[, -which(colMeans(is.na(training)) > 0.5)]
testing <- testing[, -which(colMeans(is.na(testing)) > 0.5)]



# Making three different models with three different methods. Just using with different settings for each method.
# The main purpose is to predict "classe" which is a five-class category.
# Therefore I use three multiclass classification strategies: random forest rpart, and svm

# Random forest 
set.seed(1337)
train_control <- trainControl(method = "cv", number = 3, verboseIter = FALSE)
model_rf <- train(classe~., method = "rf", data = training, trControl = train_control)
model_rf$finalModel

pred_rf <- predict(model_rf, newdata = testing)
pred_matrix_rf <- confusionMatrix(pred_rf, testing$classe)

plot(pred_matrix_rf$table, col = pred_matrix_rf$byClass, main = paste("Accuracy Random Forest =",
                                                                      round(pred_matrix_rf$overall['Accuracy'], 4)))

# Random forest 2
model_rf2 <- randomForest(classe ~ ., data = training, ntree = 500, proximity = TRUE, keep.forest = FALSE, importance = TRUE)
varImpPlot(model_rf2)
MDSplot(model_rf2, training$classe)

# Rpart
set.seed(1337)
model_rpa <- rpart(classe~., data = training, method = "class")
fancyRpartPlot(model_rpa)

pred_rpa <- predict(model_rpa, newdata = testing, type = "class")
pred_matrix_rpa <- confusionMatrix(pred_rpa, testing$classe)
pred_matrix_rpa

# SVM
set.seed(1337)
train_control <- trainControl(method = "repeatedcv", number = 3, repeats = 3, verbose = FALSE)
model_svm <- train(classe~., method = "svmRadial", data = training, trControl = train_control)
model_svm$finalModel

pred_svm <- predict(model_svm, newdata = testing)
pred_matrix_svm <- confusionMatrix(pred_svm, testing$classe)
pred_matrix_svm 

plot(pred_matrix_svm$table, col = pred_matrix_svm$byClass, main = paste("support vector machine - Accuracy =",
                                                                      round(pred_matrix_svm$overall['Accuracy'], 4)))
plot(model_svm$finalModel)
