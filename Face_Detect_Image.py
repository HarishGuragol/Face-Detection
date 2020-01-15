# -*- coding: utf-8 -*-
"""
@author: Harish Guragol
"""
#Importing the opencv library
import cv2

# Get user supplied values
#imagePath = sys.argv[1]
#cascPath = sys.argv[2]
cascPath = "C:\\Users\\SRI GANESH\\Downloads\\Face Haarcascade Classifier.xml"

# Creating the haar cascade 
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread("C:\\Users\\SRI GANESH\\Downloads\\abba.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

#Print the number of faces found in the image
print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
