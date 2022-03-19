import cv2
import mediapipe as mp
import time

 
class handDetector():
    def __init__(self, mode=False, maxHands=2, maxFaces=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.maxfaces = maxFaces
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpfaces = mp.solutions.face_mesh
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.faces = self.mpfaces.FaceMesh(False, self.maxfaces, True, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.mpStyle = mp.solutions.drawing_styles
 
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.hand_results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
 
        if self.hand_results.multi_hand_landmarks:
            for handLms in self.hand_results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img
 
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.hand_results.multi_hand_landmarks:
            myHand = self.hand_results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
 
        return lmList

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.face_results = self.faces.process(imgRGB)
        if self.face_results.multi_face_landmarks:
            for face_landmarks in self.face_results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=self.mpfaces.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mpStyle.get_default_face_mesh_tesselation_style())
                    self.mpDraw.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=self.mpfaces.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mpStyle.get_default_face_mesh_contours_style())
                    self.mpDraw.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=self.mpfaces.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mpStyle.get_default_face_mesh_iris_connections_style())
        return img
 
 
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        img = detector.findFaces(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
 
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
 
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
 
        cv2.imshow("Image", img)
        cv2.waitKey(1)
 
 
if __name__ == "__main__":
    main()