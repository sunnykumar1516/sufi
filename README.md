# sufi : Do image and video manipulation ||apply filters|| extract frames||



# how to use 
## install

    pip install sufi
    
   ## import and use

    from sufi import filters

  # apply spell filters
  	from sufi import filters
	from sufi import utility

	filters.apply_spell() # casting spell

  # sample output
<img width="1277" alt="spell" src="https://github.com/sunnykumar1516/sufi/assets/25007015/adfaae00-3ea4-4f98-a443-f0420ac41a49">
<img width="1275" alt="shield" src="https://github.com/sunnykumar1516/sufi/assets/25007015/d8633d11-d084-47dc-96e3-1679e90cebd6">

# original image

<img width="560" alt="Screenshot 2023-03-16 at 12 45 51 AM" src="https://github.com/sunnykumar1516/sufi/assets/25007015/22e16e9b-674a-4eb1-8b70-bf850fc2d2ea">

# convert  image to digital art


    path = "../images/football.jpeg"  #path of image
	img = cv.imread(path)  # read image
	img = filters.digital_art_filter(img)  # apply filter
	filters.display_image(img) #display image
 <img width="1119" alt="Screenshot 2024-04-25 at 11 29 28 PM" src="https://github.com/sunnykumar1516/sufi/assets/25007015/21036bd6-596a-4772-85d5-b7a3444026de">

	
# convert image to pencil art
	path = "../images/football.jpeg"  #path of image
	img = cv.imread(path)  # read image
	img = filters.sketch_filter(img)
	filters.display_image(img) #display image
	
 <img width="1112" alt="Screenshot 2024-05-01 at 4 29 31 PM" src="https://github.com/sunnykumar1516/sufi/assets/25007015/cf214e22-883d-4799-946c-0bf2feb6c7ce">

# there are total of 10 filters which can be applied using the package
# blur
# sharp
# canny
# HDR
# sketch
# digital paint
# black and white

# adjust values for different types of images
	
	digital_art_filter(img,sigma_s=10,sigma_r=0.3,blur = 7)
here you supply **sigma_s** and **sigma_r** as per your need.

# extract frames from video and save locally

## if you want to extract images from a video you can use below function from the package

    from sufi import videoProcessing as vd
    vd.extract_frames(path,delay)

**path** is video path , it should be your local file path fo the video
 **dealy**  is which frame you want to extract, like every 10th frame or every 20th frame.


