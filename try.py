import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0) # 0을 사용하는 것은 기본 카메라를 사용하겠다는 의미
hand_detector = mp.solutions.hands.Hands()


while True:
    _, frame = cap.read() # _: 프레임이 잘 읽혔는지 확인(True or False)
                          # frame: 이미지 프레임. Numpy 배열 형태로 저장
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    handedness = output.multi_handedness 
    # drawing_utils = mp.solutions.drawing_utils

    if hands:
        for idx, hand in enumerate(hands):
            # 왼손 또는 오른손 정보 가져오기
            hand_label = handedness[idx].classification[0].label
            # 손의 랜드마크 그리기
            for lm_idx, landmark in enumerate(hand.landmark):
                if lm_idx in [4, 8, 12, 9]:  # 원하는 랜드마크 인덱스
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    # 왼손은 파란색, 오른손은 빨간색으로 표시
                    if hand_label == 'right':
                        color = (255, 0, 0)  # 파란색
                    else:
                        color = (0, 0, 255)  # 빨간색
                    cv2.circle(frame, (cx, cy), 5, color, cv2.FILLED)
            # 손 종류 표시
            cv2.putText(frame, hand_label, (cx + 10, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if not _:
        print("웹캠 확인 필요")
        break

    cv2.imshow("Cam", frame) # 창이름, 이미지 프레임
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("종료")
        break

cap.release()
cv2.destroyAllWindows()
