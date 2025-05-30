import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from ultralytics import YOLO
import matplotlib.pyplot as plt

class Yolo8Svetofor:
    def __init__(self, video_manzili="video3.mp4"):
        self.model = self._modelni_yaratish()
        self.min_yashil = 5
        self.max_yashil = 60
        self.video = cv2.VideoCapture(video_manzili)
        if not self.video.isOpened():
            raise ValueError("Video ochilmadi.")
        self.tarix = []

        # YOLOv8 modelini yuklash
        self.yolo_model = YOLO('yolov8n.pt')  # 'yolov8n.pt' ni joylashtiring

    def _modelni_yaratish(self):
        X = np.array([[0], [5], [10], [15], [20], [30], [40], [50]])
        y = np.array([5, 10, 18, 25, 30, 40, 50, 60])
        model = LinearRegression()
        model.fit(X, y)
        return model

    def avtomobil_sonini_hisoblash(self, frame):
        results = self.yolo_model(frame, verbose=False)[0]
        detected_frame = frame.copy()
        count = 0

        for box in results.boxes:
            cls_id = int(box.cls)
            if self.yolo_model.model.names[cls_id] in ['car', 'truck', 'bus', 'motorbike']:  # faqat transport
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(detected_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(detected_frame, self.yolo_model.model.names[cls_id],
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return count, detected_frame

    def bashorat_qilish(self, mashinalar_soni):
        vaqt = self.model.predict([[mashinalar_soni]])[0]
        return np.clip(vaqt, self.min_yashil, self.max_yashil)

    def monitoring(self):
        plt.figure(figsize=(12, 6))
        mashina_soni = [x[0] for x in self.tarix]
        yashil_vaqt = [x[1] for x in self.tarix]

        plt.subplot(1, 2, 1)
        plt.plot(mashina_soni, 'b-', label='Avtomobillar soni')
        plt.title('Avtomobillar soni dinamikasi')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(yashil_vaqt, 'g-', label='Yashil vaqt')
        plt.title('Yashil chiroq vaqti')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def ishga_tushirish(self):
        print("Iltimos kuting, video ishlanmoqda... 'q' tugmasini bosing")
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break

            mashinalar_soni, detected_frame = self.avtomobil_sonini_hisoblash(frame)
            yashil_vaqt = self.bashorat_qilish(mashinalar_soni)
            self.tarix.append((mashinalar_soni, yashil_vaqt))

            cv2.putText(detected_frame, f"Avtomobillar: {mashinalar_soni}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(detected_frame, f"Yashil vaqt: {yashil_vaqt:.1f}s", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("YOLOv8 Avtomobil aniqlash", detected_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()
        self.monitoring()

if __name__ == "__main__":
    try:
        svetofor = Yolo8Svetofor("video3.mp4")  # 0 yoki video yo‘li: 'videos/my_video.mp4'
        svetofor.ishga_tushirish()
    except Exception as e:
        print("Xatolik:", str(e))
