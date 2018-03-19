shinyServer(function(input, output, session) {
  
    ### Rent Map ###
    # function to make new color palette based on what user has filtered to
    colorpal <- reactive({
      return(
        colorNumeric(palette="Spectral", domain=filtered_data()$Rent, reverse=TRUE)
        )
    })
  
    # function to filter data based on user input
    filtered_data <- reactive({
      selected_bed <- as.character(seq(input$bed[1], input$bed[2], by=1))
      selected_bed <- ifelse(selected_bed == '0', 'S', selected_bed)
      return(
        rent_data %>% 
        # filter to rent price choices
        filter(Rent >= input$rent[1] & Rent <= input$rent[2]) %>%
        # filter to number of bedrooms choices
        filter(Num.Bedrooms %in% selected_bed) %>%
        # filter to square footage choices
        filter(SqFt >= input$sqft[1] & SqFt <= input$sqft[2])
      )
    })
    
    # initial map to add points to for rent
    output$rent_chart <- renderLeaflet({
      return(
        leaflet(rent_data) %>% 
          addTiles() %>% 
          setView(lng = -71.0589, lat = 42.3601, zoom = 12)
        )
      })
  
    # observe for color palette and data filter
    observe({
      # get new color palette
      pal <- colorpal()
      # get filtered data
      leafletProxy('rent_chart', data = filtered_data()) %>%
        # remove old points
        clearMarkers() %>%
        # add new points at lat, lon
        addCircleMarkers(~Longitude, ~Latitude, 
                         # text that pops up when you clikc on a point
                         popup = ~paste0('Rent: $', as.character(Rent), '<br />', 
                                         'Number of Bedrooms: ', Num.Bedrooms, '<br />', 
                                         'Square Footage: ', as.character(SqFt), '<br />', 
                                         as.character(Apt.or.House)), 
                         # text that pops up when you hover on a point
                         label = ~as.character(Address), 
                         # aesthetics
                         color=~pal(Rent), stroke = FALSE, fillOpacity = 0.5, radius=7)
    })
  
  # observe for legend
  observe({
    proxy <- leafletProxy('rent_chart', data = filtered_data())
    # remove current legend
    proxy <- proxy %>% clearControls()
    # only if we want to see a legend
    # if (input$legend) {
    if(TRUE) {
      pal <- colorpal()
      # add it in
      proxy <- proxy %>% 
        addLegend("bottomleft", pal = pal, values = ~Rent,
                  title = "Rent",
                  # add $ to labels
                  labFormat = labelFormat(prefix = "$"),
                  opacity = 1
        )
    }
    return(proxy)
  })
  ######
  
  ### Heat Map for Predicted Price ###
  # function to make new color palette based on what user has filtered to
  colorpal_hm <- reactive({
    return(
      colorNumeric(palette="Spectral", domain=values(filtered_data_hm()), reverse=TRUE)
    )
  })
  
  # function to filter data based on user input
  filtered_data_hm <- reactive({
    # get value selected for number of bedrooms
    hm_bed <- input$hm_bed
    if(hm_bed == 0) {
      hm_bed <- 'S'
    }
    points$Num.Bedrooms <- hm_bed
    # get value selected for apartment or house
    points$Apt.or.House <- input$hm_apt
    # get value selected for utilities
    points$Utilities.Included <- toupper(input$hm_utilities)
    # get mean squarefootage for places with given number of bedrooms
    points$SqFt <- mean((rent_data %>% filter(Num.Bedrooms==hm_bed))$SqFt, na.rm=T)
    # get selected value of single units
    points$single_unit <- ifelse(input$hm_single_unit=='Yes', 1, 0)
    points$keep <- TRUE
    # calculate predicted rent for this new data
    if(input$hm_model=='Baseline Model') {
      selected_model <- baseline_model
      points$pred_rent <- predict(selected_model, newdata=points, allow.new.levels=TRUE)
    }
    else if(input$hm_model=='Original Random Intercept Model') {
      selected_model <- random_intercept_model
      points$pred_rent <- predict(selected_model, newdata=points, allow.new.levels=TRUE)
    }
    else if(input$hm_model=='Random Intercept Model with Zip') {
      selected_model <- random_intercept_model_zip
      points$pred_rent <- predict(selected_model, newdata=points, allow.new.levels=TRUE)
    }
    else if(input$hm_model=='Nested Random Slope Model') {
      selected_model <- nested_random_slope_model
      points$pred_rent <- predict(selected_model, newdata=points, allow.new.levels=TRUE)
    }
    else if(input$hm_model=='Kriging Model') {
      selected_model <- kriging_model
      to_add <- rent_data %>% 
        select(Longitude, Latitude, Num.Bedrooms, Town, Apt.or.House, 
               Utilities.Included, SqFt, State, single_unit, Address, 
               Zipcode, pred_rent)
      to_add$keep <- FALSE
      points <- rbind(points, to_add)
      points$Num.Bedrooms <- as.factor(points$Num.Bedrooms)
      points$Town <- as.factor(points$Town)
      points$Apt.or.House <- as.factor(points$Apt.or.House)
      points$Utilities.Included <- as.factor(points$Utilities.Included)
      points$Address <- as.factor(points$Address)
      points$Zipcode <- as.factor(points$Zipcode)
      points$pred_rent <- predict(selected_model, newdata=points, allow.new.levels=TRUE)
    }
    points <- points %>% filter(keep==TRUE)
    print(head(points))
    print(summary(points$pred_rent))
    points <- as.data.frame(points)
    ## create raster grid
    s <- SpatialPixelsDataFrame(points[,c('Longitude', 'Latitude')], data = points)
    # set WGS84 projection
    crs(s) <- sp::CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    # set WGS84 projection
    r <- raster(s)
    # reset values to be predicted rent
    values(r) <- matrix(points$pred_rent, nrow(r), ncol(r), byrow = TRUE)
    return(r)
  })
  
  output$pred_rent_heatmap <- renderLeaflet({
    return(
      ## plot map
      leaflet() %>%
        addTiles() %>%
        setView(lng = -71.0589, lat = 42.3601, zoom = 12)
    )
  })
  
  # observe for color palette and data filter
  observe({
    # get new color palette
    pal <- colorpal_hm()
    # get filtered data
    leafletProxy('pred_rent_heatmap', data = filtered_data_hm()) %>%
      # remove old points
      clearImages() %>%
      addRasterImage(filtered_data_hm(), colors=pal, opacity = 0.6)
  })
  
  # observe for legend
  observe({
    proxy <- leafletProxy('pred_rent_heatmap', data = filtered_data_hm())
    # remove current legend
    proxy <- proxy %>% clearControls()
    # only if we want to see a legend
    # if (input$legend_err) {
    if(TRUE) {
      pal <- colorpal_hm()
      # add it in
      proxy <- proxy %>% 
        addLegend('bottomleft', pal=pal, values=values(filtered_data_hm()), title='Predicted Rent Price')
    }
    return(proxy)
  })
  
  ### Error Map ###
  # function to make new color palette based on what user has filtered to
  colorpal_err <- reactive({
    return(
      colorNumeric(palette="Spectral", domain=filtered_data_err()$err_rent, reverse=TRUE)
    )
  })
  
  # function to filter data based on user input
  filtered_data_err <- reactive({
    selected_bed <- as.character(seq(input$bed_err[1], input$bed_err[2], by=1))
    selected_bed <- ifelse(selected_bed == '0', 'S', selected_bed)
    return(
      rent_data %>% 
        # filter to rent price choices
        filter(err_rent >= input$err[1] & err_rent <= input$err[2]) %>%
        # filter to number of bedrooms choices
        filter(Num.Bedrooms %in% selected_bed) %>%
        # filter to square footage choices
        filter(SqFt >= input$sqft_err[1] & SqFt <= input$sqft_err[2])
    )
  })
  
  # initial map to add points to for error
  output$sq_err_rent_chart <- renderLeaflet({
    return(
      leaflet(rent_data) %>% 
        addTiles() %>% 
        setView(lng = -71.0589, lat = 42.3601, zoom = 12)
    )
  })
  
  # observe for color palette and data filter
  observe({
    # get new color palette
    pal <- colorpal_err()
    # get filtered data
    leafletProxy('sq_err_rent_chart', data = filtered_data_err()) %>%
      # remove old points
      clearMarkers() %>%
    # add new points at lat, lon
      addCircleMarkers(~Longitude, ~Latitude,
                           # text that pops up when you click on a point
                           popup = ~paste0('Rent: $', as.character(Rent), '<br />',
                                           'Predicted Rent: $', as.character(round(pred_rent)), '<br />',
                                           # 'Squared Error of Rent: ', as.character(round(sq_err_rent)), '<br />',
                                           'Error of Rent: $', as.character(round(err_rent))
                                           ),
                           # text that pops up when you hover on a point
                           label = ~as.character(Address),
                           # aesthetics
                           color=~pal(err_rent), stroke = FALSE, fillOpacity = 0.5, radius=7)
  })
  
  # observe for legend
  observe({
    proxy <- leafletProxy('sq_err_rent_chart', data = filtered_data_err())
    # remove current legend
    proxy <- proxy %>% clearControls()
    # only if we want to see a legend
    # if (input$legend_err) {
    if(TRUE) {
      pal <- colorpal_err()
      # add it in
      proxy <- proxy %>% 
        addLegend("bottomleft", pal = pal, values = ~err_rent,
                  title = "Rent",
                  # add $ to labels
                  labFormat = labelFormat(prefix = "$"),
                  opacity = 1
        )
    }
    return(proxy)
  })
  ######
  
})
