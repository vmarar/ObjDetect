# Contact Tracing Using object Detection
Maintaing Social Distancing Using ObjDetection

Using ImageAI, OpenCV and Matplotlib to detect people in a space, this script uses eucledian distance between centroids
to establish a social distancing threshold. Using the centroid points and data we create graphs showing centroid 
movement and interaction, and use OpenCV VideoWriter to create a 'movie' of centroid movements from matplotlib graphs. 

Method : 
Using a pretrained YoloV3 Model and ImageAI's object detection framework, OpenCV is used to execute a frame by frame function that
identifies the probability, box points, and name of every object identified in each frame. The box points of each object are used 
to find the centroid of each object. The centroids of each object in the frame are compared with each other, and those that do not meet 
'social distancing' or eucledian distance criteria are then visualized using matplotlib. Using the various plots of these centroids Video Writer 
is used to create a movie of centroid movements to visualize contact between centroids. 

