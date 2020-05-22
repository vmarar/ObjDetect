from imageai.Detection import VideoObjectDetection
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

data = []

# func to calculate eucledian distance using np einsum
def euc_dist(a , b, metric='eucledian'):
    a = np.asarray(a)
    b = np.atleast_2d(b)
    a_dim = a.ndim
    b_dim = b.ndim
    if a_dim == 1:
        a = a.reshape(1, 1, a.shape[0])
    if a_dim >= 2:
        a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    if b_dim > 2:
        b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
    diff = a - b
    dist_arr = np.einsum('ijk,ijk->ij', diff, diff)
    if metric[:1] == 'e':
        dist_arr = np.sqrt(dist_arr)
    dist_arr = np.squeeze(dist_arr)
    return dist_arr


# Eucledian dist for 1D points
def find_dist_ratio(x1,y1):
     length = abs(x1 - y1)
     return length


# per frame function which calculates centroid, euc_dist between all persons and uses dist threshold
def dist_func(counting, output_objects_array,output_objects_count):
     a =[]
     points =[]
     for d in output_objects_array:
         x1 = d['box_points'][0]
         y1 = d['box_points'][1]
         x2 = d['box_points'][2]
         y2 = d['box_points'][3]
         centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
         #print(type(centroid))
         a.append(centroid)
     for i in range(len(a)):
         for j in range(i+1, len(a)):
            distance = euc_dist(a[i],a[j])
            if distance > find_dist_ratio(x1, y1):
                print('Social Distancing Not Maintained')
                pair1 = []
                pair2 = []
                x, y = a[i]
                x2, y2 = a[j]
                pair1.append(x)
                pair1.append(y)
                pair2.append(x2)
                pair2.append(y2)
                points.append([pair1, pair2])
     data.append(points)




execution_path = os.getcwd()
results_dir = os.path.join(execution_path, 'Results/')
detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, 'yolo.h5'))
detector.loadModel()
custom = detector.CustomObjects(person=True)
detection = detector.detectCustomObjectsFromVideo(custom_objects=custom,
                                                  input_file_path=os.path.join(execution_path,'traffic-mini.mp4'),
                                                  output_file_path=os.path.join(execution_path,'OUTPUT4.mp4'),
                                                  frames_per_second=30, per_frame_function=dist_func,
                                                  minimum_percentage_probability=60,
                                                  log_progress=True)



# creating a plot using the centroid pairs of each frame
def plot_points(data_points):
    print(data_points)
    for set in data_points:
       if set:
           for pairs in set :
               pair1 = pairs[0]
               pair2 = pairs[1]
               x, y = pair1
               x2, y2 = pair2
               plt.plot([x, x2], [y, y2], zorder=2)
               plt.scatter([x, x2], [y, y2])
           plt.savefig(os.path.join(results_dir,"{}Frame.png".format(pairs)))
           plt.show()

plot_points(data)

# creating a movie from graphs to show movement of centroids
images = [img for img in os.listdir(results_dir)]
frame = cv2.imread(os.path.join(results_dir, images[0]))
h , w, l = frame.shape
video = cv2.VideoWriter('graph-video.avi', 0, 1, (w, h))
for image in images:
    video.write(cv2.imread(os.path.join(results_dir,image)))
cv2.destroyAllWindows()
video.release()
