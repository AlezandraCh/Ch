import cv2 
import numpy as np

img1_color = cv2.imread("img7.jpg")  
img2_color = cv2.imread("img8.jpg") 

img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY) 
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY) 
height, width = img2.shape 

#orb
orb = cv2.ORB_create() 
kp1_orb, d1_orb = orb.detectAndCompute(img1, None) 
kp2_orb, d2_orb = orb.detectAndCompute(img2, None) 
matcher_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
matches_orb = matcher_orb.match(d1_orb, d2_orb) 
matches_orb.sort(key = lambda x: x.distance) 
matches_orb = matches_orb[:int(len(matches_orb)*90)] 
no_of_matches_orb = len(matches_orb) 
p1_orb = np.zeros((no_of_matches_orb, 2)) 
p2_orb = np.zeros((no_of_matches_orb, 2)) 
for i in range(len(matches_orb)): 
	p1_orb[i, :] = kp1_orb[matches_orb[i].queryIdx].pt 
	p2_orb[i, :] = kp2_orb[matches_orb[i].trainIdx].pt
homography_orb, mask_orb = cv2.findHomography(p1_orb, p2_orb, cv2.RANSAC) 
img_orb = cv2.warpPerspective(img1_color, homography_orb, (width, height)) 
img3_orb = cv2.drawMatches(img1,kp1_orb,img2,kp2_orb,matches_orb[:500],None)
output_orb = np.hstack([img3_orb, img_orb])
cv2.imwrite('orb.jpg', output_orb) 


#brisk
brisk=cv2.BRISK_create()
kp1_brisk, d1_brisk = brisk.detectAndCompute(img1,None)
kp2_brisk, d2_brisk = brisk.detectAndCompute(img2,None)
matcher_brisk = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
matches_brisk = matcher_brisk.match(d1_brisk, d2_brisk)
matches_brisk.sort(key = lambda x: x.distance)
matches_brisk = matches_brisk[:int(len(matches_brisk)*90)]
no_of_matches_brisk = len(matches_brisk) 
p1_brisk = np.zeros((no_of_matches_brisk, 2)) 
p2_brisk = np.zeros((no_of_matches_brisk, 2)) 
for i in range(len(matches_brisk)):
	p1_brisk[i, :] = kp1_brisk[matches_brisk[i].queryIdx].pt 
	p2_brisk[i, :] = kp2_brisk[matches_brisk[i].trainIdx].pt 
homography_brisk, mask_brisk = cv2.findHomography(p1_brisk, p2_brisk, cv2.RANSAC) 
img_brisk = cv2.warpPerspective(img1_color, homography_brisk, (width, height)) 
img3_brisk = cv2.drawMatches(img1,kp1_brisk,img2,kp2_brisk,matches_brisk[:500],None)
output_brisk = np.hstack([img3_brisk, img_brisk])
cv2.imwrite('brisk.jpg', output_brisk) 



#akaze
akaze=cv2.AKAZE_create()
kp1_akaze, d1_akaze = akaze.detectAndCompute(img1,None)
kp2_akaze, d2_akaze = akaze.detectAndCompute(img2,None)
matcher_akaze = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
matches_akaze = matcher_akaze.match(d1_akaze, d2_akaze)
matches_akaze.sort(key = lambda x: x.distance)
matches_akaze = matches_akaze[:int(len(matches_akaze)*90)]
no_of_matches_akaze = len(matches_akaze) 
p1_akaze = np.zeros((no_of_matches_akaze, 2)) 
p2_akaze = np.zeros((no_of_matches_akaze, 2)) 
for i in range(len(matches_akaze)): 
	p1_akaze[i, :] = kp1_akaze[matches_akaze[i].queryIdx].pt 
	p2_akaze[i, :] = kp2_akaze[matches_akaze[i].trainIdx].pt 

homography_akaze, mask_akaze = cv2.findHomography(p1_akaze, p2_akaze, cv2.RANSAC) 
img_akaze = cv2.warpPerspective(img1_color, homography_akaze, (width, height)) 
img3_akaze = cv2.drawMatches(img1,kp1_akaze,img2,kp2_akaze,matches_akaze[:500],None)
output_akaze = np.hstack([img3_akaze, img_akaze])
cv2.imwrite('akaze.jpg', output_akaze)

print('успешно')