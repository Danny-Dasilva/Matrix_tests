import cv2

import glob

images = glob.glob('./calibration_images/*.jpg')
for fname in images:

    image = cv2.imread(fname)

    cv2.imshow('test', image)


    cv2.waitKey(0)
cv2.destroyAllWindows()
