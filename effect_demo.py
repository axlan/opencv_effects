import cv2
import sys
import numpy as np
import glob

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def draw_on_image(bottom, top, x=0, y=0):
    (h, w) = top.shape[:2]
    y1, y2 = y, y + h
    x1, x2 = x, x + w
    
    x_lim = 0
    y_lim = 0
    if x2 >= bottom.shape[1]:
        x_lim = x2 - bottom.shape[1] + 1
        w -= x_lim
        x2 -= x_lim
    if y2 >= bottom.shape[0]:
        y_lim = y2 - bottom.shape[0] + 1
        h -= y_lim
        y2 -= y_lim

    alpha_top = top[:h, :w, 3] / 255.0
    alpha_bottom = 1.0 - alpha_top
    for c in range(0, 3):
        bottom[y1:y2, x1:x2, c] = (alpha_top * top[:h, :w, c] +
                                   alpha_bottom * bottom[y1:y2, x1:x2, c])
    if bottom.shape[2] == 4:
        bottom[y1:y2, x1:x2, 3] = np.maximum(top[:h, :w, 3], bottom[y1:y2, x1:x2, 3])

def main():
    cascPath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(0)

    video_capture.set(3,1280)
    video_capture.set(4,720)

    rotation = 0.0
    rotation_rate = -2.0
    face_img = cv2.imread("images/face.png", -1)
    text_img = cv2.imread("images/text_centered.png", -1)

    magic_frame_index = 0
    magic_frame_files = glob.glob('images/magic_circle_frames/magic*')
    magic_frames = [cv2.imread(filename) for filename in magic_frame_files]

    #Load the dictionary that was used to generate the markers.
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    # Initialize the detector parameters using default values
    parameters =  cv2.aruco.DetectorParameters_create()


    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            w = int(w * 1.2)
            h = int(h * 1.2)

            combined = np.zeros(face_img.shape)

            rotated = rotate_image(text_img, rotation)

            draw_on_image(combined, rotated, 22, 26)
            draw_on_image(combined, face_img)
            face_resized = cv2.resize(combined, (w, h))

            draw_on_image(frame, face_resized, x, y)
        rotation += rotation_rate

        if True:
            # for simple noise reduction we use deque
            from collections import deque
            # simple noise reduction
            h_array = deque(maxlen = 5)

           
        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

        for corners in markerCorners:

            im_src = magic_frames[magic_frame_index]
            (h, w) = im_src.shape[:2]

            pts_dst = np.concatenate(corners, axis = 1)
            pts_src = np.float32([[0, 0], [0, w], [h , w], [h, 0]])


            RES_SIZE = (frame.shape[1], frame.shape[0])

            M = cv2.getPerspectiveTransform(pts_src, pts_dst)

            warped = cv2.warpPerspective(im_src, M, RES_SIZE, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
            warped = cv2.cvtColor(warped, cv2.COLOR_RGB2RGBA)
            warped[warped[:,:,0] == 255] = 0

            draw_on_image(frame, warped)
            

        magic_frame_index += 1
        magic_frame_index %= len(magic_frames)



        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

main()
