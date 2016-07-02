import cv2
import imutils

import numpy as np

def camera():
	cap=cv2.VideoCapture(0)
	while True:
		ret,frame=cap.read()

		cv2.imshow('frame',frame)

		if cv2.waitKey(1)== ord('q'):
			
			cv2.imwrite("/home/astha/Documents/shradha1.jpg", frame)
			
			break

	cap.release()
	cv2.destroyAllWindows()

	

def contours():

	image1 = cv2.imread("index1.jpeg")
	gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
	 
	# threshold the image, then perform a series of erosions +
	# dilations to remove any small regions of noise
	thresh1 = cv2.threshold(gray1, 160, 255, cv2.THRESH_BINARY)[1]
	thresh1 = cv2.erode(thresh1, None, iterations=2)
	thresh1 = cv2.dilate(thresh1, None, iterations=2)

	# find contours in thresholded image, then grab the largest
	# one
	cnts1 = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts1= cnts1[0] if imutils.is_cv2() else cnts1[1]
	c1 = max(cnts1, key=cv2.contourArea)
	extLeft1 = tuple(c1[c1[:, :, 0].argmin()][0])
	extRight1 = tuple(c1[c1[:, :, 0].argmax()][0])
	extTop1 = tuple(c1[c1[:, :, 1].argmin()][0])
	extBot1 = tuple(c1[c1[:, :, 1].argmax()][0])


	 
	#cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
	cv2.circle(image1, extLeft1, 8, (0, 0, 255), -1)
	cv2.circle(image1, extRight1, 8, (0, 255, 0), -1)
	cv2.circle(image1, extTop1, 8, (255, 0, 0), -1)
	cv2.circle(image1, extBot1, 8, (255, 255, 0), -1)

	rect1 = cv2.minAreaRect(c1)
	box1 = cv2.boxPoints(rect1)
	print(box1)
	box1 = np.int0(box1)
	cv2.drawContours(image1, [box1], 0,(0,0,255),2)
	print('xleft',extLeft1[0])
	print('xright',extRight1[0])
	print('xtop',extTop1[0])
	print('xbot',extBot1[0])
	# show the output image

	cv2.imwrite("/home/astha/Virtual-Online-Shopping-/vtr/index2.jpg", image1)

	cv2.waitKey(0)

	image = cv2.imread("shradha1.jpg")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	 
	# threshold the image, then perform a series of erosions +
	# dilations to remove any small regions of noise
	thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	c = max(cnts, key=cv2.contourArea)
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])

	cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
	cv2.circle(image, extRight, 8, (0, 255, 0), -1)
	cv2.circle(image, extTop, 8, (255, 0, 0), -1)
	cv2.circle(image, extBot, 8, (255, 255, 0), -1)

	rect = cv2.minAreaRect(c)
	box = cv2.boxPoints(rect)
	print(box)
	box = np.int0(box)
	cv2.drawContours(image, [box], 0,(0,0,255),2)
	print('xleft',extLeft[0])
	print('xright',extRight[0])
	print('xtop',extTop[0])
	print('xbot',extBot[0])
	x=extRight[0]-extLeft[0]
	# show the output image

	cv2.imwrite("/home/astha/Virtual-Online-Shopping-/vtr/astha4.jpg", image)
	cv2.waitKey(0)

	x1 = (extRight1[0]-extLeft1[0]) 


	# find contours in thresholded image, then grab the largest
	# one

	img1 = cv2.imread('astha4.jpg')

	#the dress will be given by the user through the webpage
	img2 = cv2.imread('index1.jpeg')
	a=((x-x1)/2 + extLeft[0])
	b=extTop[1] + 30

	# I want to put logo on top-left corner, So I create a ROI
	rows,cols,channels = img2.shape
	roi = img1[b:rows+b, a:cols+a ]#the region of interest is x=180 to 439 and y=180 to 374  
	#the area of interest will be given by function above
	print('Rows:',rows)
	print('cols:', cols)
	# Now create a mask of logo and create its inverse mask
	img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


	# add a threshold
	ret, mask1 = cv2.threshold(img2gray, 100, 255, cv2.THRESH_BINARY_INV)

	mask_inv = cv2.bitwise_not(mask1)

	# Now black-out the area of logo in ROI
	img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

	# Take only region of logo from logo image.
	img2_fg = cv2.bitwise_and(img2,img2,mask = mask1)


	dst = cv2.add(img1_bg,img2_fg)
	img1[b:rows+b, a:cols+a ] = dst

	cv2.imshow('res',img1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ =='__main__':
	#camera()
	contours()
