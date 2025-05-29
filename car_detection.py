import cv2
import torch

car_cascade = cv2.CascadeClassifier("cars.xml")
cap = cv2.VideoCapture(r"C:\Users\acren\OneDrive\Desktop\yol.mp4")

# 2. YOLOv5 modelini yuklash (internet bo‘lsa birinchi marta yuklanadi)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

if not cap.isOpened():
    print("❌ Video ochilmadi!")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("🎬 Video tugadi.")
            break

        # Cascade uchun grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 3)

        # YOLO uchun kadrni RGB ga o‘zgartiramiz
        rgb_frame = frame[..., ::-1]

        # YOLO modeliga kadrni beramiz
        results = model(rgb_frame)

        # Pandas dataframe ko‘rinishidagi natijalar
        df = results.pandas().xyxy[0]

        # Transport vositalarini filterlaymiz
        vehicles = df[df['name'].isin(['car', 'truck', 'bus', 'motorcycle'])]

        # Cascade bilan aniqlangan joylarni chizamiz (yashil to‘rtburchak)
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # YOLO orqali aniqlangan transportni chizamiz va turini yozamiz (qizil rang)
        for _, row in vehicles.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # qizil rang
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Mashinalarni aniqlash va turini ko'rsatish", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
