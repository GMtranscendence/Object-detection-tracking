import cv2 as cv
import numpy as np

def findDesctriptors(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
    orb = cv.ORB_create(nfeatures=2000)
    keypoints, desctiptors = orb.detectAndCompute(gray, None)
    image = cv.drawKeypoints(image, keypoints, None)
    return (image, keypoints, desctiptors)

def find_matches(image1,image2, descriptors, keypoints1):
    gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY) 
    orb = cv.ORB_create(nfeatures=2000)
    keypoints2, descriptors2 = orb.detectAndCompute(gray, None)
    bf = cv.BFMatcher(cv.NORM_HAMMING) 
    matches = bf.knnMatch(descriptors, descriptors2, k=2)
    good_matches = []
    good_without_list = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
            good_without_list.append(m)
    if len(good_without_list)>50:
        src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good_without_list]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good_without_list]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = image1.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        image2 = cv.polylines(image2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    return keypoints2, good_matches, image2


if __name__ == '__main__':

    img = cv.imread('./marker.jpg')
    cap = cv.VideoCapture('./find_chocolate.mp4')

    img_desc = findDesctriptors(img)
    
    # show keypoints on the original image
    #cv.imshow('orig_keypoints', img_desc[0])

    while True:
        ret, frame = cap.read()
        if ret:
            kp2 = find_matches(img_desc[0], frame, img_desc[2], img_desc[1])
            cv.imshow('res', kp2[2])

            # show keypoints matching on the video
    #        cv.imshow('Match', cv.drawMatchesKnn(img_desc[0], img_desc[1], frame, kp2[0], kp2[1], None, flags=2))

            k = cv.waitKey(30) & 0xFF
            if k == ord('q'):
                break
        else:
            break
    cap.release
    cv.destroyAllWindows()











