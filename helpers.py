import matplotlib.pyplot as plt
import numpy as np
import cv2

# Rasmni ko‘rsatish uchun oddiy yordamchi funksiya
def show_image(image, title="Image"):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Rasmni RGB formatda o‘qish
def read_image(path):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Oddiy rasmni o‘lchamini standartlashtirish
def standardize_input(image):
    return cv2.resize(image, (32, 32))  # Barcha rasm o‘lchami bir xil bo‘ladi
