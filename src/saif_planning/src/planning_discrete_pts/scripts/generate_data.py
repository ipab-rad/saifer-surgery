#!/usr/bin/env python
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import numpy as np
import glob
import random


def generate_occluded_images(image_files="*.jpg", occlusions_per_target=20):

    image_paths = glob.glob(image_files)

    w = 640
    h = 480
    offset = 150

    target_index = 0
    for image_path in sorted(image_paths):

        for n in range(occlusions_per_target):
            image = cv2.imread(image_path)
            num_occlusions = random.randint(2, 5)
            for i in range(num_occlusions):
                shape = random.randint(0, 3)
                print("shape: " + str(shape))
                color = np.random.randint(0, 256, 3)
                print("color " + str(color))
                start = (random.randint(offset, w - offset), random.randint(offset, h - offset))
                if shape == 0:
                    end = (random.randint(offset, w - offset), random.randint(offset, h - offset))
                    cv2.line(image, start, end, color, random.randint(50,100))

                elif shape == 1:
                    end = (random.randint(offset, w - offset), random.randint(offset, h - offset))
                    cv2.rectangle(image, start, end, color, -1)

                elif shape == 2:
                    radius = random.randint(80, 200)
                    cv2.circle(image, start, radius, color, -1)

            print("../target_{}/occluded/occ_{}".format(target_index, n))
            cv2.imwrite("../target_{}/occluded/occ_{}.jpg".format(target_index, n), image)  
              
        target_index += 1


def generate_warped_images(image_files="*.jpg"):

    image_paths = glob.glob(image_files)

    w = 640
    h = 480
    offset = 100

    affine_points_list = [\
        np.float32([[0,0], [int(.6*w-1),0], [int(.4*w-1),h-1]]), \
        np.float32([[int(.4*w-1),0], [w-1,0], [0,h-1]]), \
        np.float32([[int(.6*w-1),0], [0,0], [w-1,h-1]]), \
        np.float32([[w-1,0], [int(.4*w-1),0], [int(.6*w-1),h-1]]), \
        np.float32([[0,int(.4*(h-1))], [w-1,0], [0,h-1]]), \
        np.float32([[0,0], [w-1,int(.4*(h-1))], [0,.6*h]]) \
        ]

    perspective_pts_list = [\
        np.float32([[int(.3*w),0], [0,h-1], [int(.7*w),0], [w-1,h-1]]), \
        np.float32([[int(.3*w),h-1], [0,0], [int(.7*w),h-1], [w-1,0]]), \
        np.float32([[0,int(.3*h)], [0,int(.7*h)], [w-1,0], [w-1,h-1]]), \
        np.float32([[0,0], [0,h-1], [w-1,int(.3*h)], [w-1,int(.7*h)]]) \
    ]

    target_index = 0
    for image_path in sorted(image_paths):
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

    pngs = glob.glob('./*.png')

    for j in pngs:
        img = cv2.imread(j)
        cv2.imwrite(j[:-3] + 'jpg', img)

    generate_occluded_images()
    generate_warped_images()

