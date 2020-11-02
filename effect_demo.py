import cv2
import numpy as np

import sys
import glob


# Rotate an image around its center
# angle is given in degrees
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

# Draw the image top over the image bottom, offset by x,y
# Handles transparency in the top image
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

# Draw an animated laughing man effect over faces detected in a frame
class LaughingFaceEffect:
    ROTATION_RATE = -2.0
    FACE_IMAGE_PATH = "images/face.png"
    TEXT_IMAGE_PATH = "images/text_centered.png"
    CASCADE_XML_PATH = 'haarcascade_frontalface_default.xml'
    
    def __init__(self):
        self.rotation = 0.0
        self.face_img = cv2.imread(LaughingFaceEffect.FACE_IMAGE_PATH, -1)
        self.text_img = cv2.imread(LaughingFaceEffect.TEXT_IMAGE_PATH, -1)
        self.faceCascade = cv2.CascadeClassifier(LaughingFaceEffect.CASCADE_XML_PATH)

    def process_frame(self, frame):
        
        # get greyscale frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # run face detection
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # draw laughing man on each face
        for (x, y, w, h) in faces:
            # Scale image to be larger then detected face
            w = int(w * 1.2)
            h = int(h * 1.2)

            # Create empty image to draw on
            combined = np.zeros(self.face_img.shape)
            # Draw on rotated text
            rotated = rotate_image(self.text_img, self.rotation)
            draw_on_image(combined, rotated, 22, 26)
            # Draw on face template
            draw_on_image(combined, self.face_img)
            # Resize to detected face
            face_resized = cv2.resize(combined, (w, h))
            # Draw on original frame
            draw_on_image(frame, face_resized, x, y)
        self.rotation += LaughingFaceEffect.ROTATION_RATE

# Draw an animated magic circle over ArUco markers in frame
class MagicCircleEffect:
    ROTATION_RATE = -2.0
    CIRLCE_FRAMES_PATH = 'images/magic_circle_frames/magic*'
    
    def __init__(self):
        self.magic_frame_index = 0
        # Load animation frames from path
        magic_frame_files = glob.glob(MagicCircleEffect.CIRLCE_FRAMES_PATH)
        self.magic_frames = [cv2.imread(filename) for filename in magic_frame_files]
        #Load the dictionary that was used to generate the markers.
        self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        # Initialize the detector parameters using default values
        self.parameters =  cv2.aruco.DetectorParameters_create()

    def process_frame(self, frame):
        
        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame,
            self.dictionary,
            parameters=self.parameters)

        # Draw image over each marker
        for corners in markerCorners:
            # Current frame of circle animation 
            im_src = self.magic_frames[self.magic_frame_index]
            (h, w) = im_src.shape[:2]

            # Find transform between source image and marker in frame
            pts_dst = np.concatenate(corners, axis = 1)
            pts_src = np.float32([[0, 0], [0, w], [h , w], [h, 0]])
            M = cv2.getPerspectiveTransform(pts_src, pts_dst)

            RES_SIZE = (frame.shape[1], frame.shape[0])
            # create a projection of im_src with a white background
            warped = cv2.warpPerspective(im_src, M, RES_SIZE, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
            # Add an alpha channel and make white points transparent
            warped = cv2.cvtColor(warped, cv2.COLOR_RGB2RGBA)
            warped[(warped[:,:,0] == 255) & (warped[:,:,1] == 255) & (warped[:,:,2] == 255)] = 0

            # Draw on original frame
            draw_on_image(frame, warped)
            
        self.magic_frame_index += 1
        self.magic_frame_index %= len(self.magic_frames)

def main():

    show_face_effect = False
    show_circle_effect = False

    # Initialize camera
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3,1280)
    video_capture.set(4,720)

    laughing_face_effect = LaughingFaceEffect()
    magic_circle_effect = MagicCircleEffect()

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if show_face_effect:
            laughing_face_effect.process_frame(frame)
        if show_circle_effect:
            magic_circle_effect.process_frame(frame)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Check for key presses
        key_press = cv2.waitKey(1) & 0xFF
        # Exit on 'q'
        if key_press == ord('q'):
            break
        # toggle show_face_effect on 'f'
        elif key_press == ord('f'):
            show_face_effect = not show_face_effect
        # toggle show_circle_effect on 'm'
        elif key_press == ord('m'):
            show_circle_effect = not show_circle_effect

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
