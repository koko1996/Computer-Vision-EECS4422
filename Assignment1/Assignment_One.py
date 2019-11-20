###################################################################################################
# EECS4422 Assignment 1                                                                           #
# Filename: Assignment_One.py                                                                     #
# Author: NANAH JI, KOKO                                                                          #
# Email: koko96@my.yorku.com                                                                      #
# Student Number: 215168057                                                                       #
###################################################################################################

import cv2
import sys
import math as m
import numpy as np
import matplotlib.pyplot as plt

def Edge_Helper(image):
    # Get mask on the edge of the object
    edges = cv2.Canny(image = image, threshold1 = 100,threshold2 = 200)
    # Convert the mask into binary mask
    edges = edges / 255
    return edges


def Question_4_a(imgin, thick, border_colour, font_colour):
    # sanity clean up
    imgin[imgin < 255 ] = 0

    # get the edges of the objects
    edges = Edge_Helper(imgin)

    # change the image into binary image
    imgin = imgin / 255
    imgin_complement = imgin.copy()
    
    # change the 1's to 0's and 0's to 1's
    imgin[imgin == 0] = 2
    imgin = imgin - 1  
    
    # increase the thickness of the edge to the given thickness
    kernel = np.ones(((thick*2)-1,(thick*2)-1), np.uint8) 
    edges = cv2.dilate(edges, kernel, iterations=1)

    # remove the dialted parts that were diated towards the image itself and not outside
    imgin = imgin.astype(int)   # change it into int to avoid overflowing when subrtracting edges from images
    imgin = imgin - edges 
    imgin[imgin < 0] =0
    imgin = imgin.astype('uint8') # revert back to uint8

    # remove the dialted parts that were diated towards the image itself and not outside
    edges = edges.astype(int)   # change it into int to avoid overflowing when subrtracting edges from images
    edges = edges - imgin_complement
    edges[edges < 0] =0
    edges = edges.astype('uint8') # revert back to uint8

    # Create a 3d (BGR) image based on the colors passed in the argument
    image_3d = np.dstack([imgin * font_colour[2],imgin * font_colour[1],imgin * font_colour[0]])
    edges_3d = np.dstack([edges * border_colour[2],edges * border_colour[1],edges * border_colour[0]])

    # merge the image with the edge
    final_image = edges_3d + image_3d 
  
    # convert the background into white
    # taken from https://stackoverflow.com/questions/52735231/how-to-select-all-non-black-pixels-in-a-numpy-array
    black_pixels_mask = np.all(final_image == [0, 0, 0], axis=-1)
    final_image[black_pixels_mask] = [255, 255, 255]

    # change the image into RGB per question specification
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB) 

    return final_image

# Gauss1D function returns one dimensional gaussian which is taken from the steerable 
# filters demo     
def Gauss1D(x, sigma = 1):
    out = np.exp(-(np.square(x))/(2*(sigma**2)))/(sigma*m.sqrt(2*m.pi))
    return out

# ddxGauss2D function returns gaussian derivative which is taken from the steerable 
# filters demo    
def ddxGauss1D(x, sigma=1):
    out = -x*(np.exp(-(np.square(x))/(2*(sigma**2)))/((sigma**3)*m.sqrt(2*m.pi)))
    return out

# ddxGauss2D function returns gaussian derivative with respect to x which is taken 
# from the steerable filters demo    
def ddxGauss2D(x, y, sigma = 1):
    x_out = ddxGauss1D(x, sigma)
    y_out = Gauss1D(y, sigma)
    out = np.outer(y_out, x_out)
    return out

# ddyGauss2D function returns gaussian derivative with respect to y which is taken 
# from the steerable filters demo    
def ddyGauss2D(x, y, sigma = 1):
    x_out = Gauss1D(x, sigma)
    y_out = ddxGauss1D(y, sigma)
    out = np.outer(y_out, x_out)
    return out

# steerGauss function returns steerable filter based on gaussian derivative which is taken 
# from the steerable filters demo
def steerGauss(x,y,sigma=1,theta=0):
    xpart = ddxGauss2D(x,y,sigma) * np.cos(np.deg2rad(theta))
    ypart = ddyGauss2D(x,y,sigma) * np.sin(np.deg2rad(theta))
    return xpart+ypart

def Question_4_b(imgin, shadow_size, shadow_magnitude,orientation):
    # This function implements steerable filter which is taken 
    # from the steerable filters demo
    # Construct a steering kernel at angle theta
    kdim = (shadow_size )        # the dimensions of the kernel (kdim x kdim). This should be an odd number
    sigma = shadow_magnitude     # the standard deviation of the Gaussian
    theta = (orientation + 180 ) % 360 # the angle in degrees at which we want our kernel

    khw = (kdim-1)/2        # the half-width of the kernel, this is useful for a number of calculations

    # digitize the signal by specifying the x and y values at integer intervals over the kernel dimensions
    x = np.arange(-khw,khw+1,1)
    y = np.arange(-khw,khw+1,1)

    # compute the function at the x and y locations
    steerkern = steerGauss(x,y,sigma,theta)
    # cross correlate the kernel with the image
    steer_img = cv2.filter2D(imgin, -1, steerkern)

    # reverse the colors i.e. the darkest becomes the brightest and second darkest becomes 
    # the second brightest and so on
    max_value = steer_img.max()
    tmp_steer_img = steer_img.copy()
    tmp_steer_img[:] = max_value
    steer_img = tmp_steer_img - steer_img

    unique_values = len(np.unique(steer_img[steer_img!=0])) # number unique non-zero values in steer_img
    # distribute the color across intensities rather than having all the colors from 0 to max_value
    steer_img[steer_img == steer_img.max()] = 255
    steer_img[steer_img != steer_img.max()] *= int((256/unique_values))

    return steer_img

def Parse_Pikachu():
    # Read the pokemon image that is located in current directory
    pokemon = cv2.imread('Pikachu.JPG',1)

    # Create a 1d mask for pokemon
    pokemon_1d = np.copy(pokemon[:, :, 2])
    pokemon_1d[pokemon_1d < 55  ] = 255
    pokemon_1d[pokemon_1d > 184  ] = 255
    pokemon_1d[pokemon_1d < 185  ] = 0
    pokemon_1d[pokemon_1d > 0  ] = 1
    
    # dilate the pokemon mask
    # taken from AlphaMatte with morph demo
    kernel = np.ones((3,3), np.uint8) 
    pokemon_1d = cv2.dilate(pokemon_1d, kernel, iterations=1)
    
    # create a 4d pokemon matrix where first 3 dimensions are the original 3d pokemon
    # the last dimension is an additional dimension that represents the binary mask
    pokemon_4d = np.dstack([pokemon[:, :, 0],pokemon[:, :, 1],pokemon[:, :, 2],pokemon_1d])

    #  crop pokemon based on the mask
    pokemon_4d = pokemon_4d[np.ix_(pokemon_1d.any(1),pokemon_1d.any(0))]

    return pokemon_4d

def Parse_Jigglypuff():
    # Read the pokemon image that is located in current directory
    pokemon = cv2.imread('Jigglypuff.JPG',1)

    # crop the image manually for ease of use (it does not fit on normal monitor screen)
    pokemon = pokemon[450:, :, :]

    # Create a 1d mask for pokemon
    pokemon_1d = np.copy(pokemon[:, :, 2])
    pokemon_1d[pokemon_1d < 70  ] = 255
    pokemon_1d[pokemon_1d > 167  ] = 255
    pokemon_1d[pokemon_1d < 190  ] = 0
    pokemon_1d[pokemon_1d > 0  ] = 1

    # create a 4d pokemon matrix where first 3 dimensions are the original 3d pokemon
    # the last dimension is an additional dimension that represents the binary mask
    pokemon_4d = np.dstack([pokemon[:, :, 0],pokemon[:, :, 1],pokemon[:, :, 2],pokemon_1d])

    #  crop pokemon based on the mask
    pokemon_4d = pokemon_4d[np.ix_(pokemon_1d.any(1),pokemon_1d.any(0))]

    return pokemon_4d

def Parse_Muk():  
    # Read the pokemon image that is located in current directory
    pokemon = cv2.imread('Muk.JPG',1)

    # crop the image manually for ease of use (it does not fit on normal monitor screen)
    pokemon = pokemon[500:, :, :]

    # Create a 1d mask for pokemon by creating seperate 1d masks from
    # each color channel and then combining these masks into one mask
    # Create the first 1d mask from the blue channel
    pokemon_0d = np.copy(pokemon[:, :, 0])
    pokemon_0d[pokemon_0d > 125  ] = 255
    pokemon_0d[pokemon_0d < 65  ] = 255
    pokemon_0d[pokemon_0d < 126  ] = 0

    # Create the second 1d mask from the red channel
    pokemon_2d = np.copy(pokemon[:, :, 2])
    pokemon_2d[pokemon_2d > 215  ] = 255
    pokemon_2d[pokemon_2d < 50  ] = 255
    pokemon_2d[pokemon_2d < 255  ] = 0
    
    # combine the created two masks
    pokemon_mask = ((pokemon_0d / 255) + (pokemon_2d / 255) ) 
    # some value could be equal to two (if it is one in both masks) make them equal to one
    pokemon_mask[pokemon_mask==2] = 1

    # Create the second 1d mask from the green channel
    pokemon_1d = np.copy(pokemon[:, :, 1])
    pokemon_1d[pokemon_1d > 224  ] = 255
    pokemon_1d[pokemon_1d < 155  ] = 255
    pokemon_1d[pokemon_1d < 255  ] = 0

    # combine the created mask with the other masks
    pokemon_mask = ((pokemon_mask) + (pokemon_1d / 255) ) 
    # some value could be equal to two (if it is one in both masks) make them equal to one
    pokemon_mask[pokemon_mask==2] = 1

    # dilate the pokemon mask
    # taken from AlphaMatte with morph demo
    kernel = np.ones((5,5), np.uint8) 
    pokemon_mask = cv2.dilate(pokemon_mask, kernel, iterations=1)
 
    # create a 4d pokemon matrix where first 3 dimensions are the original 3d pokemon
    # the last dimension is an additional dimension that represents the binary mask
    pokemon_4d = np.dstack([pokemon[:, :, 0],pokemon[:, :, 1],pokemon[:, :, 2],pokemon_mask])

    #  crop pokemon based on the mask
    pokemon_4d = pokemon_4d[np.ix_(pokemon_mask.any(1),pokemon_mask.any(0))]

    return pokemon_4d  


def Parse_Flygon():  
    # Read the pokemon image that is located in current directory
    pokemon = cv2.imread('Flygon.JPG',1)

    # crop the image manually for ease of use 
    pokemon = pokemon[50:-175, 200:-260, :]

    # Create a 1d mask for pokemon by creating seperate 1d masks from
    # each color channel and then combining these masks into one mask
    # Create the first 1d mask from the blue channel
    pokemon_0d = np.copy(pokemon[:, :, 0])
    pokemon_0d[pokemon_0d > 110  ] = 255
    pokemon_0d[pokemon_0d < 70  ] = 255
    pokemon_0d[pokemon_0d != 255  ] = 0

    # Create the second 1d mask from the red channel
    pokemon_2d = np.copy(pokemon[:, :, 2])
    pokemon_2d[pokemon_2d > 125 ] = 255
    pokemon_2d[pokemon_2d < 75  ] = 255
    pokemon_2d[pokemon_2d != 255  ] = 0
    
    # combine the created two masks
    pokemon_mask = ((pokemon_0d / 255) + (pokemon_2d / 255) ) 
    # some value could be equal to two (if it is one in both masks) make them equal to one
    pokemon_mask[pokemon_mask==2] = 1

    # Create the second 1d mask from the green channel
    pokemon_1d = np.copy(pokemon[:, :, 1])
    pokemon_1d[pokemon_1d < 181  ] = 255
    pokemon_1d[pokemon_1d != 255  ] = 0

    # combine the created mask with the other masks
    pokemon_mask = ((pokemon_mask) + (pokemon_1d / 255) ) 
    # some value could be equal to two (if it is one in both masks) make them equal to one    
    pokemon_mask[pokemon_mask==2] = 1

    # dilate the pokemon mask
    # taken from AlphaMatte with morph demo
    kernel = np.ones((3,3), np.uint8) 
    pokemon_mask = cv2.dilate(pokemon_mask, kernel, iterations=1)
 
    # Erode the mask
    # taken from AlphaMatte with morph demo
    morph_size = 3
    # Create a structuring element
    morph_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size,morph_size))
    # We want to get rid of the little dots in the background, so that would be an opening operation. Let's start with the erosion
    pokemon_mask = cv2.erode(pokemon_mask.astype(np.uint8), morph_kern, iterations=1)
    
    # create a 4d pokemon matrix where first 3 dimensions are the original 3d pokemon
    # the last dimension is an additional dimension that represents the binary mask
    pokemon_4d = np.dstack([pokemon[:, :, 0],pokemon[:, :, 1],pokemon[:, :, 2],pokemon_mask])

    #  crop pokemon based on the mask
    pokemon_4d = pokemon_4d[np.ix_(pokemon_mask.any(1),pokemon_mask.any(0))]

    return pokemon_4d    

def Question_5(scene_img,pokemon_string,location,width):   
    
    ##################################################################################################
    ################################# Pokemon Preprocessing ##########################################
    ##################################################################################################

    if pokemon_string == "Pikachu":
        pokemon_4d = Parse_Pikachu()
    elif pokemon_string == "Jigglypuff":
        pokemon_4d = Parse_Jigglypuff()
    elif pokemon_string == "Muk":
        pokemon_4d = Parse_Muk()
    elif pokemon_string == "Flygon":
        pokemon_4d = Parse_Flygon()
    else:
        raise ValueError("Pokemon name must be one of Flygon, Jigglypuff, Muk, Pikachu")

    scene_img = cv2.cvtColor(scene_img, cv2.COLOR_RGB2BGR)  
    (sy,sx,sc) = scene_img.shape # get the height, width, and number of channels in the image

    ##################################################################################################
    ############################# Insert Pokemon Into Scene Image ####################################
    ##################################################################################################
    
    # Resize the given cropped pokemon to the appropriate size 
    (py,px) = pokemon_4d[:, :, 3].shape         # get height and widht of the cropped pokemon
    height = int(py*width/px)                   # calculate the new height scaled according to its weight
    pokemon_4d = cv2.resize(pokemon_4d, (width,height)) # resize pokemon
    

    # Calculate the centroid of the pokemon taken from https://www.learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
    M = cv2.moments(pokemon_4d[:, :, 3])    # calculate moments of binary image
    cX = int(M["m10"] / M["m00"])           # calculate x coordinate of centeroid
    cY = int(M["m01"] / M["m00"])           # calculate y coordinate of centeroid


    # Resize the pokemon matrix size to the size of the scene image and add one width and one height on each side 
    # so that placing pokemon on the boundary won't cause any errors, since the user can specify location such that 
    # the pokemon does not fully fit in the image 
    # We crop the added extra height and width at the end
    resized_pokemon = np.zeros((sy+ (2* height), sx+ (2*width), 4), dtype='uint8')
    # Copy the Pokemon to the appropriate location on the matrix (needed for Alpha Matting )
    resized_pokemon[:, :, 0][height-cY+location[1]:height-cY+location[1]+pokemon_4d.shape[0], width-cX+location[0]:width-cX+location[0]+pokemon_4d.shape[1]] = pokemon_4d[:, :, 0] * pokemon_4d[:, :, 3]
    resized_pokemon[:, :, 1][height-cY+location[1]:height-cY+location[1]+pokemon_4d.shape[0], width-cX+location[0]:width-cX+location[0]+pokemon_4d.shape[1]] = pokemon_4d[:, :, 1] * pokemon_4d[:, :, 3]
    resized_pokemon[:, :, 2][height-cY+location[1]:height-cY+location[1]+pokemon_4d.shape[0], width-cX+location[0]:width-cX+location[0]+pokemon_4d.shape[1]] = pokemon_4d[:, :, 2] * pokemon_4d[:, :, 3]
    resized_pokemon[:, :, 3][height-cY+location[1]:height-cY+location[1]+pokemon_4d.shape[0], width-cX+location[0]:width-cX+location[0]+pokemon_4d.shape[1]] = pokemon_4d[:, :, 3]
    
    # Resize the scene image by adding one width and one height to each side for the same reason as pokemon
    resized_scene_img = np.zeros((sy+ (2* height), sx+ (2*width), 3), dtype='uint8')
    # copy the scene image to the appropriate location
    resized_scene_img[:, :, :][height:scene_img.shape[0]+height,width:scene_img.shape[1]+width] = scene_img[:, :, :]

    ## AlphaMatte the images, taken from AlphaMatte demo
    # create a 3d mask from 1d mask (4th dimension in resized_pokemon is the 1d mask)
    gs_mask3c = cv2.cvtColor(resized_pokemon[:, :, 3].astype('uint8'), cv2.COLOR_GRAY2BGR)
    # create a 3d composite of two images based on mask
    composite = resized_pokemon[:, :, 0:3]*gs_mask3c + resized_scene_img*(1-gs_mask3c)
    # crop the final image by removing the extra added widths and heights
    crop_composite = composite[height:scene_img.shape[0]+height,width:scene_img.shape[1]+width]

    # convert the image into RGB
    crop_composite = cv2.cvtColor(crop_composite, cv2.COLOR_BGR2RGB)  

    return crop_composite    
  


# if __name__ == '__main__':  
#     ### Question 4 part a ###
#     image = cv2.imread('question_4_output/sample_letters.png',0)

#     # Test case 1
#     thick = 2
#     border_colour = [0,0,200]
#     font_colour = [200,0,0]
#     img_out = Question_4_a(image, thick, border_colour, font_colour)
#     cv2.imshow('Question 4 part a Test case 1',img_out)
#     plt.figure()
#     plt.imshow(img_out) 
#     plt.show()  # display it

#     # Test case 2
#     thick = 5
#     border_colour = [0,100,0]
#     font_colour = [50,0,50]
#     img_out = Question_4_a(image, thick, border_colour, font_colour)
#     cv2.imshow('Question 4 part a Test case 2',img_out)
#     plt.figure()
#     plt.imshow(img_out) 
#     plt.show()  # display it

#     ### Question 4 part b ###
#     # Test case 1
#     shadow_size = 11
#     shadow_magnitude = 3
#     orientation = 45
#     img_out = Question_4_b(image, shadow_size, shadow_magnitude,orientation)
#     cv2.imshow('Question 4 part b Test case 1', img_out)
    

#     # Test case 2
#     shadow_size = 25
#     shadow_magnitude = 8
#     orientation = 65
#     img_out = Question_4_b(image, shadow_size, shadow_magnitude,orientation)
#     cv2.imshow('Question 4 part b Test case 2', img_out)

#     # Test case 3
#     shadow_size = 25
#     shadow_magnitude = 4
#     orientation = 0
#     img_out = Question_4_b(image, shadow_size, shadow_magnitude,orientation)
#     cv2.imshow('Question 4 part b Test case 3', img_out)

#     ## Question 5
#     scene = cv2.imread('Q5_samples/image_1299.png',1)
#     scene = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)     

#     pokemon_string = "Flygon"
#     pokemon_string = "Pikachu"
#     pokemon_string = "Jigglypuff"
#     pokemon_string = "Muk"
#     location = (100,100)
#     width = (280)

#     img_out = Question_5(scene, pokemon_string,location,width)
#     plt.figure()
#     plt.imshow(img_out) 
#     plt.show()  # display it
#     cv2.imshow('Question 5', img_out)

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()