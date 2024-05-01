# sufi
# sufi : do image and video manipulation ||apply filters|| extract frames||


# how to use 
## install

    pip install sufi
    
   ## import and use

    from sufi import filters

# convert to image to digital art


    path = "../images/football.jpeg"  #path of image
	img = cv.imread(path)  # read image
	img = filters.digital_art_filter(img)  # apply filter
	filters.display_image(img) #display image
	
# convert image to pencil art
	path = "../images/football.jpeg"  #path of image
	img = cv.imread(path)  # read image
	img = filters.sketch_filter(img)
	filters.display_image(img) #display image
	

# adjust values for different types of images
	
	digital_art_filter(img,sigma_s=10,sigma_r=0.3,blur = 7)
here you supply **sigma_s** and **sigma_r** as per your need.

# blur
# sharp
# canny
# HDR
# sketch
# digital paint




