rm(list = ls())
gc(reset = TRUE)

library("plyr")
library("dplyr")
library("ggplot2")
library("gridExtra")
library("leaflet")
library("leaflet.extras")
library("ggmap")

##############
# load dataset 
##############

load("~/Boston.RData")
Bos <- Boston.data

# check missing values 
apply(X = Bos, MARGIN = 2, FUN = function(x) sum(is.na(x)))

# looks columns `SqFt`, `Rent`, `Latitude`, `Longitude` have missing values 
# for obs with `Rent` missing, leave it as test cases 
# for 'Lat' and 'Lon', remove 
# for 'SqFt', since very large portion missed, 
# try impute / remove in model training stage 

# for visualization, remove all these missing obs temporarily 
Bos <- na.omit(Bos)
Bos <- subset(Bos, SqFt != 0)


############
# Some plots
############

# The more bedrooms, the larger the House/Apt
# The larger the House/Apt, the higher the Rent 

xlims <- range(Bos$SqFt)
ylims <- range(Bos$Rent)

p1 <- Bos %>% 
	subset(Bos$Apt.or.House == "Apartment") %>%
		ggplot(aes(x = SqFt, y = Rent)) + 
  			geom_point(aes(color = num.bedrooms)) + 
  			coord_cartesian(xlim = xlims, ylim = ylims) + 
  			labs(x = "Square Footage of Property",
       			y = "Rents",
       			title = "Rent VS SqFt for Apartment",
       			caption = "Source: http://www.rentals.com/Massachusetts/Boston/")

p2 <- Bos %>% 
	subset(Bos$Apt.or.House == "House") %>%
		ggplot(aes(x = SqFt, y = Rent)) + 
  		geom_point(aes(color = num.bedrooms)) +
  		coord_cartesian(xlim = xlims, ylim = ylims) + 
  		labs(x = "Square Footage of Property",
       		y = "Rents",
       		title = "Rent VS SqFt for House",
       		caption = "Source: http://www.rentals.com/Massachusetts/Boston/")

grid.arrange(p1, p2, nrow = 2, ncol = 1)

# Rents diff cross towns 
# some Houses/Apts with very high rent in center Boston
# need to pay attention when traing model 

ggplot(Bos, aes(x = Town, y = Rent)) +
	geom_boxplot() + 
	theme(axis.text.x = element_text(angle = 45, size = 4, hjust = 1)) +
	labs(x = "Town",
		y = "Rent",
        title = "Rent by Town",
        caption = "Source: http://www.rentals.com/Massachusetts/Boston/")

bos.map <- get_map("Boston", messaging = F)

ggmap(bos.map) + 
	geom_point(aes(x = Longitude, y = Latitude, color = Rent), data = Bos) + 
	scale_color_continuous(low = "yellow", high = "red")  


# leaflet map show rentals on map 

Bos %>%
	leaflet() %>% 
		addTiles() %>% 
			addMarkers(lng = ~Longitude,
				lat = ~Latitude,
            	clusterOptions = markerClusterOptions(),
            	popup = paste("Address:", Bos$Address, 
            		"Property Type:", Bos$Apt.or.House, 
            		"Square Footage:", Bos$SqFt, 
            		"Bedrooms:", Bos$num.bedrooms, 
            		"Utility Included:", Bos$Utilities.included, 
            		"Rent:", paste("$", Bos$Rent, sep = ""), sep = "<br/>")) 



	
