"""
CS4391 Homework 4 Programming
Epipolar Geometry
"""

import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


#TODO
# use your backproject function in homework 4, problem 1
from backproject import backproject
    
    
# read rgb, depth, mask and meta data from files
def read_data(file_index):

    # read the image in data
    # rgb image
    rgb_filename = 'data/%06d-color.jpg' % file_index
    im = cv2.imread(rgb_filename)
    
    # depth image
    depth_filename = 'data/%06d-depth.png' % file_index
    depth = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
    depth = depth / 1000.0
    
    # read the mask image
    mask_filename = 'data/%06d-label-binary.png' % file_index
    mask = cv2.imread(mask_filename)
    mask = mask[:, :, 0]
    
    # erode the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    
    # load matedata
    meta_filename = 'data/%06d-meta.mat' % file_index
    meta = scipy.io.loadmat(meta_filename)
    
    return im, depth, mask, meta
    
    
#TODO: implement this function to compute the fundamental matrix
# Follow lecture 14, the 8-point algorithm
# xy1 and xy2 are with shape (n, 2)
def compute_fundamental_matrix(xy1, xy2):

    # step 1: construct the A matrix
    n = xy1.shape[0]
    A = np.zeros((n,9))
    for i in range(0,n):
        x, y = xy1[i]
        xprime, yprime = xy2[i]
        ithrow = [x*xprime, xprime * y, xprime, yprime*x, yprime*y, yprime, x, y,1]
        A[i] = ithrow

    # step 2: SVD of A
    # use numpy function for SVD
    U,D,Vtranspose = np.linalg.svd(A)

    # step 3: get the last column of V
    rows, columns = Vtranspose.shape
    f = Vtranspose[-1]
    f = f.reshape(3,3)# reshape into 3x3 matrix, first 3 become first row and so on

    # step 4: SVD of F
    U,D,Vt= np.linalg.svd(f) # svd on the new matrix

    # step 5: mask the last element of singular value of F
    D[-1] = 0

    # step 6: reconstruct F
    F = np.dot(U, np.dot(np.diag(D),Vt))
    return F  


# main function
if __name__ == '__main__':

    # read image 1
    im1, depth1, mask1, meta1 = read_data(6)
    
    # read image 2
    im2, depth2, mask2, meta2 = read_data(7)
    
    # intrinsic matrix
    intrinsic_matrix = meta1['intrinsic_matrix']
    print('intrinsic_matrix')
    print(intrinsic_matrix)
        
    # get the point cloud from image 1
    pcloud = backproject(depth1, intrinsic_matrix)
    
    # find the boundary of the mask 1
    boundary = np.where(mask1 > 0)
    x1 = np.min(boundary[1])
    x2 = np.max(boundary[1])
    y1 = np.min(boundary[0])
    y2 = np.max(boundary[0])
    
    # sample n pixels (x, y) inside the bounding box of the cracker box
    # due to the randomness here, you may not get the same figure as mine
    # this is fine as long as your result is correct    
    n = 10
    height = im1.shape[0]
    width = im1.shape[1]
    x = np.random.randint(x1, x2, n)
    y = np.random.randint(y1, y2, n)
    index = np.zeros((n, 2), dtype=np.int32)
    index[:, 0] = x
    index[:, 1] = y
    print(index, index.shape)

    # get the coordinates of the n pixels
    pc1 = np.ones((4, n), dtype=np.float32)
    for i in range(n):
        x = index[i, 0]
        y = index[i, 1]
        print(x, y)
        pc1[:3, i] = pcloud[y, x, :]
    print('pc1', pc1)
    
    # filter zero depth pixels
    ind = pc1[2, :] > 0
    pc1 = pc1[:, ind]
    index = index[ind]
    xy1 = index
    # xy1 is a set of pixels on image 1
    # we will find the correspondences of these pixels

    # transform the points to another camera
    RT1 = meta1['camera_pose']
    RT2 = meta2['camera_pose']
    print(RT1.shape, RT2.shape)
    xy2 = []

    for i in range(0,index.shape[0]):
        x = index[i, 0]
        y = index[i, 1]
        pixel = pcloud[y, x, :]
        xy2.append(pixel)
    
    #TODO
    # use your code from homework 4, problem 2 to find the correspondences of xy1
    # let the corresponding pixels on image 2 be xy2 with shape (n, 2)
    xy2 = np.array(xy2)
    worldtocamer2 = []
    for pixel in xy2:
        homogenous = np.append(pixel, 1)  # homogenous coordinates
        worldcoord = np.dot(np.linalg.inv(RT1), homogenous)  # camera1 coordinate system --> world coord. sys
        camera2 = np.dot(RT2, worldcoord)  # world coord. sys --> camera 2 coord. sys
        worldtocamer2.append(camera2)
    worldtocamer2 = np.array(worldtocamer2)

    camera2_points = worldtocamer2[:, :3]  # extract camera 2 coord
    temp = []
    for point in camera2_points:
        proj = np.dot(intrinsic_matrix, point)  # project onto image 2
        x, y, z = proj
        pixel = [x / z, y / z]  # divide by z
        temp.append(pixel)

    temp = np.array(temp)
    xy2 = temp


    #TODO
    # implement this function: compute fundamental matrix
    F = compute_fundamental_matrix(xy1, xy2)
    
    # visualization for your debugging
    fig = plt.figure()
        
    # show RGB image 1 and sampled pixels
    ax = fig.add_subplot(2, 2, 1)
    plt.imshow(im1[:, :, (2, 1, 0)])
    ax.set_title('image 1: correspondences', fontsize=15)
    plt.scatter(x=xy1[:, 0], y=xy1[:, 1], c='y', s=20)
    
    # show RGB image 2 and sampled pixels
    ax = fig.add_subplot(2, 2, 2)
    plt.imshow(im2[:, :, (2, 1, 0)])
    ax.set_title('image 2: correspondences', fontsize=15)
    plt.scatter(x=xy2[:, 0], y=xy2[:, 1], c='g', s=20)
    
    # show three pixels on image 1
    ax = fig.add_subplot(2, 2, 3)
    plt.imshow(im1[:, :, (2, 1, 0)])
    ax.set_title('image 1: sampled pixels', fontsize=15)
    
    # compute epipolar lines of three sampled points
    px = 233
    py = 145
    p = np.array([px, py, 1]).reshape((3, 1))
    l1 = np.matmul(F, p)
    print(p.shape)
    print(l1) 
    plt.scatter(x=px, y=py, c='r', s=40)
    
    px = 240
    py = 245
    p = np.array([px, py, 1]).reshape((3, 1))
    l2 = np.matmul(F, p)
    plt.scatter(x=px, y=py, c='g', s=40)
    
    px = 326
    py = 268
    p = np.array([px, py, 1]).reshape((3, 1))
    l3 = np.matmul(F, p)
    plt.scatter(x=px, y=py, c='b', s=40)    
    
    # draw the epipolar lines of the three pixels
    ax = fig.add_subplot(2, 2, 4)
    plt.imshow(im2[:, :, (2, 1, 0)])
    ax.set_title('image 2: epipolar lines', fontsize=15)
    
    for x in range(width):
        y1 = (-l1[0] * x - l1[2]) / l1[1]
        if y1 > 0 and y1 < height-1:
            plt.scatter(x, y1, c='r', s=1)
            
        y2 = (-l2[0] * x - l2[2]) / l2[1]
        if y2 > 0 and y2 < height-1:
            plt.scatter(x, y2, c='g', s=1)
            
        y3 = (-l3[0] * x - l3[2]) / l3[1]
        if y3 > 0 and y3 < height-1:
            plt.scatter(x, y3, c='b', s=1)                        
                  
    plt.show()
