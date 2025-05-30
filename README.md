🚦 Smart Svetofor Tizimi - YOLOv8 yordamida avtomobil soniga asoslangan
Ushbu loyiha video orqali harakatlanayotgan transport vositalarini YOLOv8 yordami bilan aniqlaydi va ularning soniga qarab svetaforning yashil chiroq vaqtini avtomatik ravishda belgilaydi. Loyihaning maqsadi — tirbandlikni kamaytirish va harakatni samarali boshqarish.

📌 Asosiy imkoniyatlar
📷 Video orqali real vaqtli avtomobil aniqlash (YOLOv8 bilan)

🔢 Avtomobil sonini hisoblash

⏱ Avtomobil soniga qarab yashil chiroq vaqtini bashorat qilish (Linear Regression)

📈 Statistikani grafik ko‘rinishda chiqarish

🖥 Real vaqtli ko‘rsatish oynasi (OpenCV)


🧱 Loyiha tuzilmasi
.
├── trafic.py           # Asosiy kod
├── video3.mp4          # Sinov uchun video
├── yolov8n.pt          # YOLOv8 model fayli
├── env.py              # Maxfiy ma'lumotlar (agar kerak bo‘lsa)
├── requirements.txt    # Kerakli kutubxonalar
└── README.md           # Loyiha tavsifi


▶️ Dasturdan foydalanish
1. Repositoryni yuklab oling

git clone https://github.com/foydalanuvchi/smart-svetofor.git
cd smart-svetofor

2. Virtual muhit (ixtiyoriy, ammo tavsiya etiladi)

python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

3. Kutubxonalarni o‘rnating

pip install -r requirements.txt

4. YOLOv8 modelini tayyorlash
Agar sizda yolov8n.pt fayli yo‘q bo‘lsa, quyidagicha avtomatik yuklab olishingiz mumkin:


from ultralytics import YOLO
YOLO('yolov8n.pt')
Yoki https://github.com/ultralytics/ultralytics sahifasidan yuklab oling.

5. Dastur ishga tushurish

python trafic.py
q tugmasi bosilganda dastur to‘xtaydi.

Yakunda grafik ko‘rinishda avtomobil soni va yashil vaqt statistikasi ko‘rsatiladi.

📊 Model haqida
Yashil chiroq davomiyligi quyidagi asosiy qoidaga tayangan:

Avtomobil soni ko‘paygan sayin, yashil chiroq vaqtining oshirilishi kerak.

Bu uchun LinearRegression modeli ishlatilgan va quyidagicha mashq qilingan:


X = [[0], [5], [10], [15], [20], [30], [40], [50]]
y = [5, 10, 18, 25, 30, 40, 50, 60]

🧪 Namuna oynasi
Dastur oynada real vaqtli video ko‘rsatadi:

Avtomobillar soni: O‘ng yuqori burchakda

Yashil chiroq davomiyligi (sekundlarda): Pastki chap burchakda

📦 Zarur kutubxonalar
Quyidagilarni requirements.txt orqali o‘rnating:


opencv-python
numpy
scikit-learn
ultralytics
matplotlib