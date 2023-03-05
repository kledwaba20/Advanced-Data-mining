
library(ISLR) 
library(dplyr) 
library(glmnet) 
library(caret)

Carseats <- read.csv("dataset_car_seats.csv") 
Carseats
str(Carseats)
head(Carseats)

Carseats_Filtered <- Carseats %>% select("Sales", "Price", "Advertising","Population","Age","Income","Education")

#Question 1

set.seed(123)
trainIndex <- createDataPartition(Carseats$Sales, p = 0.8, list = FALSE)
trainData <- Carseats[trainIndex, ]
testData <- Carseats[-trainIndex, ]

# Extract predictor variables
predictors <- c("Price", "Advertising", "Population", "Age", "Income", "Education")
trainX <- trainData[, predictors]
testX <- testData[, predictors]

# Scale the predictor variables
preproc <- preProcess(trainX, method = c("center", "scale"))
trainX <- predict(preproc, as.matrix(trainX))
testX <- predict(preproc, as.matrix(testX))


# Use cross-validation to determine the optimal value of lambda
cvModel <- cv.glmnet(x = as.matrix(trainX), y = trainData$Sales, alpha = 1, nfolds = 5)
lambda <- cvModel$lambda.min

# Train the Lasso regression model on the training set using the optimal value of lambda
lassoModel <- glmnet(x = as.matrix(trainX), y = trainData$Sales, alpha = 1, lambda = lambda)

# Print the optimal value of lambda
print(paste("The optimal value of lambda is", lambda))
# Predict on the testing set
testPredictions <- predict(lassoModel, newx = as.matrix(testX), s = lambda)

# Calculate the mean squared error
mse <- mean((testData$Sales - testPredictions)^2)
print(paste("The mean squared error on the testing set is", round(mse, 2)))

# Question 2
lassoCoef <- coef(lassoModel, s = lambda)
lassoCoef["Price",]

#question 3
lassoModel01 <- glmnet(x = trainX, y = trainData$Sales, alpha = 1, lambda = 0.01)
nonZeroCoeffs01 <- sum(coef(lassoModel01) != 0)

# Determine the number of non-zero coefficients for lambda = 0.1
lassoModel1 <- glmnet(x = trainX, y = trainData$Sales, alpha = 1, lambda = 0.1)
nonZeroCoeffs1 <- sum(coef(lassoModel1) != 0)

# Print the number of non-zero coefficients for lambda = 0.01 and lambda = 0.1
print(paste("Number of non-zero coefficients for lambda = 0.01:", nonZeroCoeffs01))
print(paste("Number of non-zero coefficients for lambda = 0.1:", nonZeroCoeffs1))

#question 4
# Build the Elastic-Net model with alpha=0.6
enetModel <- glmnet(x=trainX, y=trainData$Sales, alpha=0.6)

# Find the optimal value of lambda using cross-validation
cvModel <- cv.glmnet(x=trainX, y=trainData$Sales, alpha=0.6, nfolds=5)
lambda <- cvModel$lambda.min

# Print the optimal value of lambda
print(paste("The optimal value of lambda for the Elastic-Net model with alpha=0.6 is", lambda))

