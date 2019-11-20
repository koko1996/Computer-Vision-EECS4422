EECS 4422 Computer Vision

How To Run:
- ipython Assignment_One.py

How To Test:
- The code in the main function tests the functionality for both questions (currently commented out)

Main Solution Ideas:
# Question 4 part a:
- Find the edges of the objects with canny edge detector, increase the thickness of the edge by dilating the image with kernel of size equals to (2*thickness-1). Lastly, emove from the dilated image the parts which fall outside the original object.

# Question 4 part b:
- Find the edges of the objects using steerable Gaussian filter.

# Question 5:
- Preprocess each Pokemon image seperately (assuming the images are in the current working directory) creating a different binary mask for each Pokemon image. Add the mask to the image matrix as the forth dimension to make the rest of the problem easier to handle. Resize the scene image by adding to the height twice the height of the Pokemon and to the width twice the width of the Pokemon to handle the case where we place the Pokemon close to the edges that some parts of the Pokemon fall outside the image. Lastly, crop the scene image to its original size.

## References 
* https://stackoverflow.com/questions/52735231/how-to-select-all-non-black-pixels-in-a-numpy-array
* https://www.learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
* Alpha Matte Demo from class
* Alpha Matte With Morph Demo
* Steerable Filters Demo from class