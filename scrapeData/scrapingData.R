rm(list = ls())
gc(reset = TRUE)

# if you don't have there libraries installed, try:
# install.packages("rvest")
# install.packages("ggmap")

library(rvest)
library(ggmap)

# couple of convenient functions to handle features 

get.state.zip <- function(address){
	nnn <- nchar(address)
	return(substr(address,nnn-7,nnn))
}

mean.rent <- function(rent){
	temp <- as.numeric(unlist(strsplit(rent," ")))
	return(mean(temp,na.rm=T))
}

get.longitude <- function(address){
  address.long <- geocode(address)$lon
  return(address.long)
}

get.latitude <- function(address){
  address.lat <-geocode(address)$lat
  return(address.lat)
}

# url toward search page for Boston, MA area 
url <- "http://www.rentals.com/Massachusetts/Boston/?per_page=3000"	
 
build.dataset <- function(url){
	html <- read_html(url)

	#get a list of href of all properties in this page
	href <- html %>% 
		html_nodes('.property_title a') %>%
			html_attr('href')
			
	#use '.listing_name a' in the bracket if want include spotlight property
	#each href in this list is like '/Massachusetts/Boston/169332/'

	#empty dataframe
	df <- data.frame(u= character(0), x= character(0), y= character(0), 
					 z = character(0), w = character(0), v = character(0), 
					 m = character(0), lat = character(0), long = character(0),
					 stringAsFactors = FALSE)
	names(df) <- c("Address", "Town", "Rent", "Bd.", "SqFt", "Apt or House", "Utilities included","Latitude","Longitude") #column names

	#loop for all properties in the href, build a table of information
	count <- 1
	for(item in href){
		print(count)
		count <- count + 1
		
		# add town name
		junk <- strsplit(item,split="/")
		town <- unlist(junk)[3]
		
  		if(grepl("/lv", item) | grepl("/r", item)){ #check if this property is a house (non-apartment)
    		url <- paste("http://www.rentals.com/", item, sep ='')
    		print(url) #use for debuging
    		
    		#read html
    		page<- read_html(url)
    		
    		#floorplan
    		floorplan <- unlist(strsplit(page %>% 
					     	html_node('#summary_floorplan') %>% 
					     		html_text, '[|]'))
 
    		#address
    		address <- page %>% 
				html_node('#summary_address') %>%
				html_text
    			
     		# add latitude and longitude
    		lat.long <- geocode(as.character(address))
			lat <- lat.long$lat    			
    		long <- lat.long$lon
    			
    		#rental price
    		price <- page %>% 
				html_node('#summary_price strong') %>% 
				html_text

    		#utilities_des

			# get utilities
 			temp <- readLines(url)
  			temp <- grep("[Uu]tilities",temp)
			utilities <- "NO"
   			if (length(temp)>0){
   				utilities <- "YES"
   			}
   				
    		#combine dataframe
    		df_temp <- data.frame(address, town, price, floorplan[1], 
    								  floorplan[3], "House", utilities, lat, long,
    								  stringAsFactors = FALSE)
    		names(df_temp) <- c("Address", "Town", "Rent", "Bd.", "SqFt", 
    								"Apt or House", "Utilities included", "Latitude", "Longitude")
    		df <- rbind(df, df_temp)
  		}
  		else{ #this property is an apartment
  			url <- paste("http://www.rentals.com", item, sep ='')
    		print(url) #use for debuging
    			
    		#read html 
    		page <- read_html(url)
    			
    		# table about appartments
    		tab <- page %>% 
				html_node(xpath='//*[@id="page"]/div[2]/div[2]/div[2]/div/div[1]/div[2]/table') %>% 
				html_table(fill=T)
 
 			# get utilities
 			temp <- readLines(url)
  			temp <- grep("[Uu]tilities",temp)
  			utilities <- "NO"
  			if (length(temp)>0){
  				utilities <- "YES"
  			}
  			
    		# address
    		address <- page %>% 
				html_node('#summary_address') %>% 
				html_text
    			
    		# add latitude and longitude
    		lat.long <- geocode(as.character(address))
			lat <- lat.long$lat    			
    		long <- lat.long$lon
    			
    		#combine dataframe
    		df_temp <- data.frame(address, town, tab$Price, tab$Bd., 
    			tab$Sq, "Apartment", utilities,lat,long, stringAsFactors = FALSE)
   			names(df_temp) <- c("Address", "Town", "Rent", "Bd.", "SqFt", "Apt or House", "Utilities included", "Latitude", "Longitude")
    		df <- rbind(df, df_temp)
			save(df,file="~/tempdata.RData")
  		}
	}
	data <- df
	
	# clean the data a little bit 
	
	# get zip code and state
	addresses <- as.character(data$Address)
	state.and.zip <- as.vector(tapply(addresses,(1:length(addresses)),get.state.zip))
	state.and.zip <- unlist(strsplit(state.and.zip," "))
	data$zip <- state.and.zip[seq(from=2,to=length(state.and.zip),by=2)]
	data$state <- state.and.zip[seq(from=1,to=length(state.and.zip),by=2)]

	# deal w/ rents
	rents <- data$Rent
	# remove dollar signs
	rents <- gsub('\\$','',rents)
	# remove commas
	rents <- gsub(',','',rents)
	# replace dashes with spaces
	rents <- gsub('-',' ',rents)	
	data$Rent <- as.vector(tapply(rents,(1:length(rents)),mean.rent))
	data$Rent[data$Rent=="NaN"] <- NA

	# deal w/ square footage
	sqft <- data$SqFt
	sqft <- gsub(',','',sqft)
	sqft <- gsub('-',' ',sqft)
	sqft <- as.vector(tapply(sqft,(1:length(sqft)),mean.rent))
	sqft[sqft=="NaN"] <- NA
	data$SqFt <- sqft
	
	# deal w/ number of bedrooms
	num.bedrooms <- data$Bd.
	# get rid of non-numbers
	num.bedrooms <- as.factor(gsub('[^0-9]','',num.bedrooms))
	# "studio" is now NA. Make it 1
	num.bedrooms[is.na(num.bedrooms)] <- "S"
	data$num.bedrooms <- num.bedrooms
	
	return(data)
}	

data <- build.dataset(url)

# get some additional data 
# income table for state MA and CT.
url <- "https://en.wikipedia.org/wiki/List_of_Massachusetts_locations_by_per_capita_income"

income_tab_MA <- url %>%
	read_html() %>%  
	html_nodes(css = 'table') %>% 
	html_table()

income_tab_MA <- as.data.frame(income_tab_MA[3])
income_tab_MA <- income_tab_MA[, c(2, 6, 7)]
colnames(income_tab_MA) <- c("Town", "Median.House.Income", "Median.Family.Income")

# clean income columns 
income_tab_MA$Median.House.Income <- as.integer(gsub("\\$|,", "", income_tab_MA$Median.House.Income))
income_tab_MA$Median.Family.Income <- as.integer(gsub("\\$|,", "", income_tab_MA$Median.Family.Income))


# join them by town
data <- merge(data, income_tab_MA, by="Town", all.x = TRUE)
save.image("~/Boston.RData")



























