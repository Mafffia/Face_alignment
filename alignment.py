"""This is the py file for face align, we try to use the align tech in ArcFace, rather than the align in DeepFace"""
import cv2
import mtcnn
import numpy as np
from skimage import transform as trans
import cv2
from PIL import Image

import math

def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def align_rotate(img,landmarks):
    left_eye_center = (landmarks[0][0],landmarks[0][1])
    right_eye_center = (landmarks[1][0],landmarks[1][1])
    left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
    right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
        # print("rotate to clock direction")
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock
        # print("rotate to inverse clock direction")
    # cv2.circle(img, point_3rd, 2, (255, 0, 0) , 2)

    # cv2.line(img,right_eye_center, left_eye_center,(67,67,67),2)
    # cv2.line(img,left_eye_center, point_3rd,(67,67,67),2)
    # cv2.line(img,right_eye_center, point_3rd,(67,67,67),2)

    a = euclidean_distance(left_eye_center, point_3rd)
    b = euclidean_distance(right_eye_center, left_eye_center)
    c = euclidean_distance(right_eye_center, point_3rd)
    cos_a = (b*b + c*c - a*a)/(2*b*c)
    angle = np.arccos(cos_a)
    angle = (angle * 180) / math.pi
    if direction == -1:
        angle = 90 - angle
    new_img = Image.fromarray(img)
    new_img = np.array(new_img.rotate(direction * angle))
    return new_img


def detect_landmark(image, detector):
    '''
    image as numpy format with RGB format
    note that cv2 read is BGR format
    '''
    face = detector.detect_faces(image)[0]

    #draw points
    left_eye = face["keypoints"]["left_eye"]
    right_eye = face["keypoints"]["right_eye"]
    nose = face["keypoints"]["nose"]
    mouth_left = face["keypoints"]["mouth_left"]
    mouth_right = face["keypoints"]["mouth_right"]
    landmark = [[left_eye[0], left_eye[1]],
                   [right_eye[0], right_eye[1]],
                   [nose[0], nose[1]],
                   [mouth_left[0], mouth_left[1]],
                   [mouth_right[0], mouth_right[1]]]
    return landmark


#<--left profile
src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)

# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode='arcface',method='affine'):
    assert lmk.shape == (5, 2)
    if(method == 'affine'):
        tform = trans.AffineTransform()
    elif(method == 'similarity'):
        tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        assert image_size == 112
        src = arcface_src
    else:
        src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index

def norm_crop(img, landmark, image_size=112, mode='arcface',method='affine'):
    M, pose_index = estimate_norm(landmark, image_size, mode,method)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


""""Input: the dir of the file
    Output: cv2 image with already aligned face
"""
def get_face_align(cropface,landmark,method='affine'):
    if(method=='rotate'):
        return align_rotate(cropface,landmark)
    img = cv2.cvtColor(cropface, cv2.COLOR_BGR2RGB)  # To RGB
    """"landmark format:
    landmark = [[left_eye[0], left_eye[1]],
                   [right_eye[0], right_eye[1]],
                   [nose[0], nose[1]],
                   [mouth_left[0], mouth_left[1]],
                   [mouth_right[0], mouth_right[1]]]
    """    
    # landmark = detect_landmark(img, detector)
    wrap = norm_crop(img, np.array(landmark), image_size=112, mode='arcface',method=method)
    wrap = cv2.cvtColor(wrap,cv2.COLOR_BGR2RGB)
    return wrap