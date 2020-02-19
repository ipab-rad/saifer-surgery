#!/usr/bin/env python
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import numpy as np
import glob
import random
import os 
import copy
#import skimage 
import matplotlib.pyplot as plt 


def generate_occluded_images(target_index, image_files="./*.jpg", occlusions_per_target=10):

    image_paths = glob.glob(image_files)
    print(image_paths)

    w = 640
    h = 480
    offset = 150
    sigma = 55

    #target_index = 0
    for image_path in sorted(image_paths):

        for n in range(occlusions_per_target):
            image = cv2.imread(image_path)

            
            # num_occlusions = random.randint(2, 5)
            # for i in range(num_occlusions):
            #     shape = random.randint(0, 3)
            #     print("shape: " + str(shape))
            #     color = np.random.randint(0, 256, 3)
            #     print("color " + str(color))
            #     start = (random.randint(offset, w - offset), random.randint(offset, h - offset))
            #     if shape == 0:
            #         end = (random.randint(offset, w - offset), random.randint(offset, h - offset))
            #         cv2.line(image, start, end, color, random.randint(50,100))

            #     elif shape == 1:
            #         end = (random.randint(offset, w - offset), random.randint(offset, h - offset))
            #         cv2.rectangle(image, start, end, color, -1)

            #     elif shape == 2:
            #         radius = random.randint(80, 200)
            #         cv2.circle(image, start, radius, color, -1)

            image = generate_occluded_image(image)
            blurred = cv2.GaussianBlur(image,(sigma,sigma),cv2.BORDER_DEFAULT) 
            #plt.imshow(blurred)
            #plt.show()

            print("../target_{}/occluded/occ_{}".format(target_index, n))
            #cv2.imwrite("../target_{}/occluded/occ_{}.jpg".format(target_index, n), image)  
            cv2.imwrite("../occluded/occ_b{}.jpg".format(n), blurred)
              
        #target_index += 1


def generate_occluded_image(image):

    w = 640
    h = 480
    offset = 150

    # occluder_paths = ['/home/ultrasound/Downloads/hand-removebg-preview.png']
    # occluder = occluder_paths[random.randint(0, len(occluder_paths) - 1)]

    # print(occluder)

    # background = copy.deepcopy(image)
    # overlay = cv2.imread(occluder)

    # rows,cols,channels = overlay.shape

    # row_start = random.randint(200, 300)
    # col_start = random.randint(0, 200)
    # overlay=cv2.addWeighted(background[row_start:row_start+rows, col_start:col_start+cols],1.0,overlay,1.0,0)

    # background[row_start:row_start+rows, col_start:col_start+cols ] = overlay


    num_occlusions = random.randint(2, 4)
    for i in range(num_occlusions):
        shape = random.randint(0, 1)
        shape = 0
        color_type = random.randint(0, 1)
        print("shape: " + str(shape))
        print("color type " + str(color_type))
        if color_type == 0:
            R = random.randint(50, 255)
            G = random.randint(int(.75 * R), int(.9 * R))
            B = random.randint(int(.85 * G), int(.9 * G))
            color = (B, G, R)
        elif color_type == 1:
            R = random.randint(0, 255)
            color = (random.randint(max(R - 5, 0), min(R + 5, 255)), random.randint(max(R - 5, 0), min(R + 5, 255)), R)
        print("color " + str(color))

        #color = (120, 138, 180)
        start = (random.randint(offset, w - offset), random.randint(offset, h - offset))
        if shape == 0:
            end = (random.randint(offset, w - offset), random.randint(offset, h - offset))
            cv2.line(image, start, end, color, random.randint(50,100))
        elif shape == 1:
            radius = random.randint(80, 150)
            cv2.circle(image, start, radius, color, -1)

    return image 
    #return background

def generate_blurred_images(image_files="*.jpg"):
    image_paths = glob.glob(image_files)
    sigma = 81

    for image_path in image_paths:
        image = cv2.imread(image_path)
        
        topLeft = (random.randint(250, 350), random.randint(100, 250))
        x, y = topLeft[0], topLeft[1]
        w, h = random.randint(100, 200), random.randint(100, 200)

        # Grab ROI with Numpy slicing and blur
        ROI = image[y:y+h, x:x+w]
        blur = cv2.GaussianBlur(ROI, (sigma,sigma), 0) 

        # Insert ROI back into image
        image[y:y+h, x:x+w] = blur

        cv2.imwrite("./b_{}".format(image_path), image)

def generate_blurred_image(image):
    sigma = 81

        #image = cv2.imread(image_path)
        
    topLeft = (random.randint(250, 350), random.randint(100, 250))
    x, y = topLeft[0], topLeft[1]
    w, h = random.randint(100, 200), random.randint(100, 200)
    # Grab ROI with Numpy slicing and blur
    ROI = image[y:y+h, x:x+w]
    blur = cv2.GaussianBlur(ROI, (sigma,sigma), 0) 
    # Insert ROI back into image
    image[y:y+h, x:x+w] = blur
    #cv2.imwrite("./b_{}".format(image_path), image)

    return image

def generate_warped_images(image_files="*.jpg"):

    image_paths = glob.glob(image_files)

    w = 640
    h = 480
    offset = 100

    affine_points_list = [\
        np.float32([[0,0], [int(.7*w-1),0], [int(.3*w-1),h-1]]), \
        np.float32([[int(.3*w-1),0], [w-1,0], [0,h-1]]), \
        np.float32([[int(.7*w-1),0], [0,0], [w-1,h-1]]), \
        np.float32([[w-1,0], [int(.3*w-1),0], [int(.7*w-1),h-1]]), \
        np.float32([[0,int(.3*(h-1))], [w-1,0], [0,h-1]]), \
        np.float32([[0,0], [w-1,int(.3*(h-1))], [0,.7*h]]) \
        ]

    perspective_pts_list = [\
        np.float32([[int(.3*w),0], [0,h-1], [int(.7*w),0], [w-1,h-1]]), \
        np.float32([[int(.3*w),h-1], [0,0], [int(.7*w),h-1], [w-1,0]]), \
        np.float32([[0,int(.3*h)], [0,int(.7*h)], [w-1,0], [w-1,h-1]]), \
        np.float32([[0,0], [0,h-1], [w-1,int(.3*h)], [w-1,int(.7*h)]]) \
    ]

    target_index = 0
    for image_path in sorted(image_paths):
        if not os.path.isdir("../target_{}".format(target_index)):
            os.mkdir("../target_{}".format(target_index))
        if not os.path.isdir("../target_{}/warped".format(target_index)):
            os.mkdir("../target_{}/warped".format(target_index))

        index = 0
        image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        src_points = np.float32([[0,0], [w-1,0], [0,h-1]])
        #dst_points = np.float32([[random.randint(0, w - 1), random.randint(0, h - 1)], [random.randint(0, w - 1), random.randint(0, h - 1)], [random.randint(0, w - 1), random.randint(0, h - 1)]])
        for i in range(len(affine_points_list)):
            affine_points = affine_points_list[i]
            
            affine_matrix = cv2.getAffineTransform(src_points, affine_points)
            img_output = cv2.warpAffine(image, affine_matrix, (w,h))

            cv2.imwrite("../target_{}/warped/warp_{}_{}".format(target_index, index, image_path), img_output)  
            index += 1 

        p_src_points = np.float32([[0,0], [0,h-1], [w-1,0], [w-1,h-1]]) 

        for i in range(len(perspective_pts_list)):
            perspective_points = perspective_pts_list[i]
            perspective_matrix = cv2.getPerspectiveTransform(p_src_points, perspective_points)
            img_output = cv2.warpPerspective(image, perspective_matrix, (w,h))

            cv2.imwrite("../target_{}/warped/warp_{}_{}".format(target_index, index, image_path), img_output)
            index += 1
        
        target_index += 1

if __name__ == "__main__":

    # pngs = glob.glob('./*.png')

    # for j in pngs:
    #     img = cv2.imread(j)
    #     cv2.imwrite(j[:-3] + 'jpg', img)

    #generate_occluded_image(cv2.imread('liquid.jpg'))
    #generate_warped_images()

    generate_occluded_images(4)
    #generate_blurred_images()

