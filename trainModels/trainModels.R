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
features = c("Bd.", "SqFt", "Apt.or.House", 
		"Utilities.included", "Latitude", "Longitude", 
		"Median.House.Income", "Median.Family.Income")
target = c('Rent')
		
train <- subset(train, select = c(features, target))

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

x_train <- subset(train, select = features)

y_train <- train$Rent
 
############################
#Multiple Linear Regression
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

# seems no penalty needed
plot(x = LRL2Fit$results$lambda, y = LRL2Fit$results$RMSE, type = "b")


############################
#Support Vector Machine  
############################

library('e1071')

SVMFit <- svm(Rent ~ ., data = train)

plot(y_train)

predictions <- predict(SVMFit, x_train)
mse <- sqrt(mean((y_train - predictions)^2))

points(predictions, col = 'red', pch = 16)
	
summary(SVMFit)

#convenient function to tune parameters
tuned_params <- tune.svm(Rent ~ .,
	data = train,
	gamma = 10^(-3:0),
	cost = 10^(0:3))

tuned_model <- tuned_params$best.model 
predictions <- predict(tuned_model, x_train)

mse <- round(sqrt(mean((y_train - predictions)^2)), 2)

ggplot(train, aes(Rent, predictions))+ 
	geom_point()+
	xlab('Actual Rent')+
	ylab('Predicted Rent')+
	xlim(0, 150000) + 
	ylim(0, 15000) + 
	ggtitle(paste('MSE = ', mse, sep = '')) + 
	geom_abline(intercept = 0, slope = 1) + 
	theme(title = element_text(size = 16),
		axis.text = element_text(size = 14))

ggplot(train, aes(Latitude, Longitude, color = predictions)) + 
	geom_point() + 
	scale_colour_gradient(name = 'Predicted Rent') + 
	theme(title = element_text(size = 16),
		axis.text = element_text(size = 14))
	
	
#################
#Decision Tree 
#################

library("rpart")
set.seed(1)

treeGrid <- expand.grid(minsplit = seq(from = 10, to = 200, length.out = 100),
	cp = seq(from = 1e-3, to = 0.99, length.out = 100))

rmse_train_cv = rep(0, times = 5)
rmse_validate_cv = rep(0, times = 5)
rmse_train = rep(0, times = nrow(treeGrid))
rmse_validate = rep(0, times = nrow(treeGrid))

for(i in 1:nrow(treeGrid)){
	kFold = createFolds(y_train, k = 5)
	for(j in 1:5){
		train_cv <- train[kFold[[j]], ]
		validate_cv <- train[-kFold[[j]], ]
		
		treeFit <- rpart(Rent ~ .,
			data = train_cv,
			control = rpart.control(minsplit = treeGrid[i, 'minsplit'],
				cp = treeGrid[i, 'cp']))
				
		prediction_train <- predict(treeFit, train_cv[, features])
		prediction_validate <- predict(treeFit, validate_cv[, features])
		
		rmse_train_cv[j] <- sqrt(mean((prediction_train - train_cv$Rent)^2))
		rmse_validate_cv[j] = sqrt(mean((prediction_validate - validate_cv$Rent)^2))
		}
	rmse_train[i] <- mean(rmse_train_cv)
	rmse_validate[i] <- mean(rmse_validate_cv)
	}

result <- cbind(treeGrid, rmse_train = rmse_train, rmse_validate = rmse_validate)

ggplot(result, aes(x = 1:nrow(result))) + 
	geom_line(aes(y = rmse_train)) + 
	geom_line(aes(y = rmse_validate), color = 'red') 


###############
#Ensemble Method
###############

library("randomForest")

rfGrid <- expand.grid(ntree = seq(from = 5, to = 50, by = 5),
	mtry = seq(from = 3, to = 8),
	sampsize = seq(from = 0.4, to = 0.9, by = 0.1),
	maxnodes = seq(from = 10, to = 100, by = 10))

rmse_train_cv = rep(0, times = 5)
rmse_validate_cv = rep(0, times = 5)
rmse_train = rep(0, times = nrow(rfGrid))
rmse_validate = rep(0, times = nrow(rfGrid))

for(i in 1:nrow(rfGrid)){
	kFold = createFolds(y_train, k = 5)
	for(j in 1:5){
		train_cv <- train[kFold[[j]], ]
		validate_cv <- train[-kFold[[j]], ]
		
		rfFit <- randomForest(Rent ~ .,
			data = train_cv,
			ntree = rfGrid[i, 'ntree'],
			mtry = rfGrid[i, 'mtry'],
			sampsize = rfGrid[i, 'sampsize'],
			nodesize = rfGrid[i, 'nodesize'],
			maxnodes = rfGrid[i, 'maxnodes'])
				
		prediction_train <- predict(treeFit, train_cv[, features])
		prediction_validate <- predict(treeFit, validate_cv[, features])
		
		rmse_train_cv[j] <- sqrt(mean((prediction_train - train_cv$Rent)^2))
		rmse_validate_cv[j] = sqrt(mean((prediction_validate - validate_cv$Rent)^2))
		}
	rmse_train[i] <- mean(rmse_train_cv)
	rmse_validate[i] <- mean(rmse_validate_cv)
	}

result <- cbind(rfGrid, rmse_train = rmse_train, rmse_validate = rmse_validate)

ggplot(result, aes(x = 1:nrow(result))) + 
	geom_line(aes(y = rmse_train)) + 
	geom_line(aes(y = rmse_validate), color = 'red') 


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
	 
