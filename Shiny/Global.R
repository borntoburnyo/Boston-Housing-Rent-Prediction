library(shiny)
library(shinydashboard)
library(leaflet)
library(leaflet.extras)
library(lme4)
library(sp)
library(raster)
library(dplyr)
library(nlme)

# load in data
load("Boston.RData")
# rename data
colnames(Boston.data) <- c("Address", "Town", "Rent", "Bedrooms", "SqFt",
                           "Apt.or.House", "Utilities.Included", 
                           "Latitude", "Longitude", "Zipcode",
                           "State", "Num.Bedrooms")
# assign data into dataframe called data
rent_data <- Boston.data
# NAs introduced by num.bedrooms containing "S"
rent_data <- rent_data %>% 
  # remove column we don't use
  select(-Bedrooms) %>% 
  mutate(Num.Bedrooms = as.factor(Num.Bedrooms))
  # reformat bedrooms from factor to character
  # mutate(Num.Bedrooms=as.character(Num.Bedrooms)) %>%
  # if studio, assign as 0 bedrooms
  # mutate(Num.Bedrooms=ifelse(Num.Bedrooms=='S', '0', Num.Bedrooms)) %>%
  # make numeric
  # mutate(Num.Bedrooms = as.numeric(Num.Bedrooms))
# remove data with missing rent
rent_data <- rent_data %>% filter(!is.na(Rent)) %>% 
  # remove data with missing lat and long
  filter(!is.na(Latitude) & !is.na(Longitude)) %>%
  # remove data with missing square footage
  filter(!is.na(SqFt))
# add single unit info to data for model
temp <- rent_data %>% 
  group_by(Address) %>%
  summarize (count = n())
idx <- which(temp$count == 1)
single.address <- temp$Address[idx]
rent_data$single_unit <- ifelse(rent_data$Address %in% single.address, 1, 0)
# convert towns for model
top5 <- c('Boston', 'Cambridge', 'Waltham', 'Malden', 'Quincy')
rent_data <- rent_data %>% 
  mutate(Town = ifelse(as.vector(Town) %in% top5 , as.vector(Town), "Other"))

rent_data$Town <- as.factor(rent_data$Town)

# load models
baseline_model <- readRDS('baseline_model.RDS')
random_intercept_model <- readRDS('rand_int_model.RDS')
random_intercept_model_zip <- readRDS('intercept_w_zip.RDS')
nested_random_slope_model <- readRDS('nested.RDS')
kriging_model <- readRDS('krig2.RDS')

### make raster image for heatmap ###
# latitude values for predicting
# go from min latitude observed to max, and make 1000 data points
x <- seq(min(rent_data$Latitude, na.rm=T), max(rent_data$Latitude, na.rm=T), length.out=1000)
# longitude values for predicting
# go from min longitude observed to max, and make 1000 data points
y <- seq(min(rent_data$Longitude, na.rm=T), max(rent_data$Longitude, na.rm=T), length.out=1000)
# make latitude values into matrix
X = matrix(rep(x, each = 1000), nrow = 1000)
# make longitude values into matrix
Y = matrix(rep(y, 1000), nrow = 1000)

# make latitude and longitude matrices into data frame
points = data.frame(Longitude = c(Y), Latitude = c(X))
# add predictor values
# initialize bedrooms as 1
points$Num.Bedrooms <- '1'
# initialize towns as other
points$Town <- 'Other'
# initialize as aparments, not houses
points$Apt.or.House <- 'Apartment'
# initialize as no utilities included
points$Utilities.Included <- 'NO'
# initialize to mean squarefootage of one bedroom apartments
points$SqFt <- mean((rent_data %>% filter(Num.Bedrooms=='1'))$SqFt, na.rm=T)
# initialize state to MA
points$State <- 'MA'#ifelse(points$Latitude < 42.7, 'MA', 'NH')
# initialize to not single units
points$single_unit <- 0
# initialize to blank string for address
points$Address <- ''
# initialize to blank string for zip code
points$Zipcode <- '02139'
# calculate predicted rent for this new data
points$pred_rent <- predict(baseline_model, newdata=points, allow.new.levels=TRUE)
rent_data <- rent_data %>% filter(Num.Bedrooms != '9')
rent_data$pred_rent <- predict(kriging_model, newdata=rent_data, allow.new.levels=TRUE)
# calculate error for actual data
rent_data$err_rent <- rent_data$Rent - rent_data$pred_rent

