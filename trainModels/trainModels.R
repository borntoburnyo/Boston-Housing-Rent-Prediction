rm(list = ls())
gc(reset = TRUE)

library("plyr")
library("dplyr")
library("lme4")
library("caret")
library("ggplot2")
library("mice")

###########
# load data
###########
 
load("~/Boston.RData")
Bos <- Boston.data 

# remove obs w/ missing values in 'Lat', 'Lon' columns 
Bos <- subset(Bos, Latitude != 0 & Longitude != 0)

# make test dataset by separate obs w/ Rent missing 
rent_miss <- subset(Bos, is.na(Rent))

# check missing values in this sub-dataset 
apply(X <- rent_miss, MARGIN = 2, FUN = function(x) sum(is.na(x)))

# remove obs w/ SqFt missing in this sub-dataset 
# call this test dataset 
# could verify prediction results by query customer service later if interested....
rent_miss <- subset(rent_miss, !is.na(SqFt))
Bos_test <- rent_miss  
save.image("~/Boston_test.RData")

# get training dataset 
# remove obs w/ missing values in SqFt columns
# impute Median.house/fam.income columns later 
train <- subset(Bos, !(is.na(Rent) | is.na(SqFt) | SqFt == 0))

# keep interested features 
train <- subset(train, 
	select = c("Bd.", "SqFt", "Apt.or.House", 
		"Utilities.included", "Latitude", "Longitude", 
		"Median.House.Income", "Median.Family.Income", "Rent"))

# do a little clean up 
train$Bd. <- gsub(" ", "", train$Bd.)
train$Bd.[train$Bd. == "Studio"] <- "1"
train$Bd. <- as.integer(gsub("[a-zA-Z]", "", train$Bd.))

# impute missing values for Income columns 
train <- train %>%
	mice(method = "pmm") %>%
		complete()

# encode categorical columns
train$Apt.or.House <- ifelse(train$Apt.or.House == "Apartment", 0, 1)
train$Utilities.included <- ifelse(train$Utilities.included == "NO", 0, 1)

# separate features and labels 
# since sample size is small (relatively), 
# use cross-validation for hyper parameter tunning

x_train <- train[, seq(1, length(train) - 1)]

y_train <- train$Rent
 
############################
# Multiple Linear Regression
############################

# Introduce L2 penalized linear regression to avoid overfitting 

library("glmnet") 
set.seed(1)

L2Grid <- data.frame(alpha = 0,
	lambda = seq(from = 1e-5, to = 1e3, length.out = 10000))

LRL2Fit <- train(x_train,
	y_train,
	method = "glmnet",
	trControl = trainControl(method = "cv", verbose = TRUE),
	tuneGrid = L2Grid,
	metric = "RMSE",
	maximize = FALSE)

plot(x = LRL2Fit$results$lambda, y = LRL2Fit$results$RMSE, type = "b")


#################
Tree Based Method 
#################

library("rpart")
set.seed(1)

treeGrid <- data.frame(cp = seq(from = 1e-5, to = 0.9, length.out = 10000))

treeFit <- train(x_train,
	y_train,
	method = "rpart",
	trControl = trainControl(method = "cv", verbose = TRUE),
	tuneGrid = treeGrid,
	metric = "RMSE",
	maximize = FALSE)

plot(x = treeFit$results$cp, y = treeFit$results$RMSE, type = "b")

###############
Ensemble Method
###############

library("randomForest")

rfGrid <- data.frame(mtry = seq(1, length(x_train) - 1))

rfFit <- train(x_train,
	y_train,
	method = "rf",
	trControl = trainControl(method = "cv", verbose = TRUE),
	tuneGrid = rfGrid,
	metric = "RMSE",
	maximize = FALSE)

plot(x = rfFit$results$mtry, y = rfFit$results$RMSE, type = 'b')


library("gbm")
library("doMC") # seems need to use parallel computing using my fake 8-core MAC...
library("foreach")


registerDoMC(4)

gbmFit <- foreach(n.trees = seq(5, 20),
	interaction.depth = seq(3, 8),
	shrinkage = seq(1e-4, 0.8, length.out = 1000),
	n.minobsinnode = seq(10, 50),
	.packages = 'gbm', 
	.combine = c) %dopar%
		gbm(Rent ~ Bd. + SqFt + Apt.or.House + Utilities.included 
			+ Latitude + Longitude + Median.House.Income + Median.Family.Income,
			data = train,
			n.trees = n.trees,
			interaction.depth = interaction.depth,
			shrinkage = shrinkage,
			n.minobsinnode = n.minobsinnode,
			cv.folds = 5,
			verbose	= TRUE)


library("xgboost")

xgbData <- xgb.DMatrix(data = x_train, label = y_train)

xbgFit <- foreach(i = seq(5), .combine = c) %:%
	foreach(eta = seq(1e-4, 0.8, length.out = 1000),
		max_depth = seq(3, 8),
		subsample = seq(0.4, 1, length.out = 10),
		colsample_bytree = seq(0.5, 1, length.out = 10),
		.packages = "xgboost",
		.combine = c) %dopar%
			xgb.train(params = list(eta = eta,
				max_depth = max_depth,
				subsample = subsample,
				colsample_bytree = colsample_bytree),
					data = xgbData)
	 





