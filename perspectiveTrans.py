import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop(depth_img, output_size):
    h,w = depth_img.shape # (rows, columns)
    center = (int(w/2),int(h/2)) # (x,y)     
    left = int(max(0, min(center[0] - output_size // 2, w - output_size)))
    top = int(max(0, min(center[1] - output_size // 2, h - output_size)))
    # cv2.rectangle(depth_img, (left,top), (min(w, left + 300),min(h, top + 300)), color=(255, 0, 255), thickness=2) # test crop
    # plt.imshow(depth_img)
    # plt.title("depth img crop test")
    # plt.show()
    depth_crop = depth_img[top:min(h, top + 300), left:min(w, left + 300)] # crop = image[y:y+h, x:x+w]
    return depth_crop

# Load the image
img = cv2.imread('block_3.png') 
 
# Create a copy of the image
img_copy = np.copy(img[:,:,0])
# img_copy = crop(img_copy,300)
# Convert to RGB so as to display via matplotlib
# Using Matplotlib we can easily find the coordinates
# of the 4 points that is essential for finding the 
# transformation matrix
img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img_copy)
# All points are in format [cols, rows]
pt_A = [104,70]
pt_B = [173,371]
pt_C = [448,369]
pt_D = [618,120]


maxWidth = 300
maxHeight = 300

input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
output_pts = np.float32([[0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0]])
# Compute the perspective transform M
M = cv2.getPerspectiveTransform(input_pts,output_pts)
out = cv2.warpPerspective(img_copy,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
plt.figure()
plt.imshow(out)
plt.show()