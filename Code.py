import cv2
import numpy as np
from skimage.filters import sobel
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt


def muat_file():
    root = Tk()
    root.withdraw()
    path_file = filedialog.askopenfilename(
        filetypes=[
            ("File gambar", "*.jpg;*.jpeg;*.png"),
            ("File video", "*.mp4;*.avi"),
        ]
    )
    return path_file


def proses_gambar(gambar):
    abu_abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    tepi = sobel(abu_abu)
    return abu_abu, tepi


def deteksi_jerawat(gambar):
    abu_abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(abu_abu, (5, 5), 0)

    # Deteksi tepi menggunakan Sobel
    tepi = sobel(blur)

    # Segmentasi menggunakan thresholding
    _, thresh = cv2.threshold(tepi, 0.02, 1.0, cv2.THRESH_BINARY)

    # Konversi ke uint8
    thresh = (thresh * 255).astype(np.uint8)

    # Morfologi untuk menghilangkan noise dan memperbesar area jerawat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Temukan kontur jerawat
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    jerawat_count = 0
    jerawat_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 10 < w < 50 and 10 < h < 50:  # Filter kontur kecil dan besar
            jerawat_count += 1
            jerawat_area += w * h
            cv2.rectangle(gambar, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                gambar,
                "Jerawat",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

    return gambar, jerawat_count, jerawat_area


def deteksi_kondisi_kulit(jerawat_count, jerawat_area, total_area):
    rasio_jerawat = jerawat_area / total_area

    if jerawat_count == 0:
        return "Kulit Bersih", (0, 255, 0)
    elif jerawat_count <= 1:
        return "Jerawat Ringan", (255, 255, 0)
    # elif jerawat_count <= 20:
    #     return "Jerawat Sedang", (0, 255, 255)
    elif jerawat_count <= 5:
        return "Jerawat Parah", (0, 255, 0)
    else:
        return "Jerawat Sangat Parah", (0, 0, 255)


def deteksi_jerawat_dan_kondisi_kulit(gambar):
    abu_abu, tepi = proses_gambar(gambar)
    total_area = abu_abu.size

    # Deteksi jerawat
    gambar, jerawat_count, jerawat_area = deteksi_jerawat(gambar)

    # Klasifikasi kondisi kulit
    kondisi, warna = deteksi_kondisi_kulit(jerawat_count, jerawat_area, total_area)

    # Tambahkan latar belakang berwarna untuk teks
    font_scale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(kondisi, font, font_scale, 2)
    text_w, text_h = text_size
    cv2.rectangle(gambar, (10, 10), (10 + text_w, 30 + text_h), (0, 0, 0), -1)

    cv2.putText(gambar, kondisi, (10, 30 + text_h), font, font_scale, warna, 2)

    return gambar


def tampilkan_gambar(gambar):
    plt.imshow(cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def analisis_gambar(path_file):
    gambar = cv2.imread(path_file)
    if gambar is None:
        print("Error: Tidak dapat membaca gambar.")
        return

    gambar_hasil = deteksi_jerawat_dan_kondisi_kulit(gambar)
    tampilkan_gambar(gambar_hasil)


def analisis_video(path_file):
    cap = cv2.VideoCapture(path_file)

    if not cap.isOpened():
        print("Error: Tidak dapat membuka file video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_hasil = deteksi_jerawat_dan_kondisi_kulit(frame)
        cv2.imshow("Analisis Deteksi Jerawat", frame_hasil)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    path_file = muat_file()
    if path_file.lower().endswith((".jpg", ".jpeg", ".png")):
        analisis_gambar(path_file)
    elif path_file.lower().endswith((".mp4", ".avi")):
        analisis_video(path_file)
    else:
        print("Format file tidak didukung.")
