from GUI import GUI
from HAL import HAL
from datetime import datetime

import numpy as np
import cv2 as cv

print(str(datetime.now()) + " [INFO] Pipeline started...")

# WORKS!
def get_ray(camera, img_p2d):
    ''' Image point from Numpy array directly '''
    global HAL, np
    
    y, x = img_p2d[:2]
    
    p1 = HAL.backproject(camera, HAL.graficToOptical(camera, [x, y, 1]))[:3]
    p0 = HAL.getCameraPosition(camera)
    v = p1 - p0
    
    return np.append(v, [1]), np.append(p0, [1])

# WORKS!   
def get_ray_projected(camera, ray3d):
    ''' Get projected line as two points in grafic plane '''
    global HAL, np
    
    k = 100
    
    p0 = ray3d[0] + ray3d[1]
    p0 = HAL.opticalToGrafic(camera, HAL.project(camera, p0))
    p1 = (k * ray3d[0]) + ray3d[1]
    p1 = HAL.opticalToGrafic(camera, HAL.project(camera, p1))
    v = p1 - p0
    
    fx = lambda x: (v[1] * (x - p0[0]) / v[0]) + p0[1]
    x_end = HAL.getImage(camera).shape[1]
    p1 = np.array([x_end, fx(x_end)])
    p0 = np.array([0, fx(0)])
    
    return p1.astype(np.int), p0.astype(np.int)

# WORKS!  
def get_ray_mask(camera, ray2d, thickness = 8):
    global HAL, np, cv
    
    image = HAL.getImage(camera)
    mask = np.zeros_like(image)
    
    cv.line(mask, tuple(ray2d[0]), tuple(ray2d[1]), (1, 1, 1), thickness)
    image *= mask
    
    return mask, image

# WORKS!    
def get_homologue(img_p2d, camera_src, camera_target):
    global GUI, HAL, get_ray, get_ray_projected, get_ray_mask
    
    template_shape = 8
    
    ray3d = get_ray(camera_src, img_p2d)
    ray2d = get_ray_projected(camera_target, ray3d)
    mask, img_target_masked = get_ray_mask(camera_target, ray2d)
    
    pad = template_shape // 2
    image_src = HAL.getImage(camera_src)
    x, y = img_p2d[:2]
    template = image_src[x - pad:x + 1 + pad, y - pad:y + 1 + pad]
    res = cv.matchTemplate(HAL.getImage(camera_target), template, cv.TM_CCOEFF_NORMED)
    _, coeff, _, top_left = cv.minMaxLoc(res)
    top_left = np.array(top_left)
    
    match_point = top_left[::-1] + pad
    bbox_target = (top_left, top_left + template_shape)
    bbox_src = ((img_p2d[1] - pad, img_p2d[0] - pad), (img_p2d[1] + pad, img_p2d[0] + pad))
    return match_point, coeff, bbox_target, bbox_src
    
def triangulate(camera_src, camera_target, img_p2d_src, img_p2d_target):
    ray3d_src = get_ray(camera_src, img_p2d_src)[0][:3]
    ray3d_target = get_ray(camera_target, img_p2d_target)[0][:3]
    
    n = np.cross(ray3d_src, ray3d_target)
    A = np.array([ray3d_src, n, -ray3d_target]).T
    b = HAL.getCameraPosition(camera_target) - HAL.getCameraPosition(camera_src)
    
    eps1, eps2, eps3 = np.linalg.lstsq(A, b)[0]
    p = (eps1 * ray3d_src) + ((eps2 / 2) * n)
    
    return p

camera_src, camera_target = 'left', 'right'
img_src = HAL.getImage(camera_src)
mask = cv.Canny(img_src, 100, 200)
points = [[x, y] for x, y in np.ndindex(*img_src.shape[:2]) if mask[x, y]]

for img_p2d in points:
    p_match, coeff, _, _ = get_homologue(img_p2d, camera_src, camera_target)
    if coeff > 0.9:
		p = triangulate(camera_src, camera_target, img_p2d, p_match)
		p = HAL.project3DScene(p).tolist()
		color = HAL.getImage(camera_src)[img_p2d[0], img_p2d[1]].tolist()
		GUI.ShowNewPoints([p + color[::-1]])

print(str(datetime.now()) + " [INFO] Finished successfully!")
