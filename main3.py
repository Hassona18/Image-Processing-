import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2

# Algorithms
def halftoning_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return binary

def advanced_halftoning_error_diffusion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for y in range(gray.shape[0] - 1):
        for x in range(1, gray.shape[1] - 1):
            old_pixel = gray[y, x]
            new_pixel = 255 if old_pixel > 128 else 0
            gray[y, x] = new_pixel
            error = old_pixel - new_pixel
            gray[y, x + 1] += error * 7 / 16
            gray[y + 1, x - 1] += error * 3 / 16
            gray[y + 1, x] += error * 5 / 16
            gray[y + 1, x + 1] += error * 1 / 16
    return np.clip(gray, 0, 255).astype(np.uint8)

def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def edge_detection(image, operator="sobel"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if operator == "sobel":
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    elif operator == "prewitt":
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        edges = cv2.filter2D(gray, -1, kernelx) + cv2.filter2D(gray, -1, kernely)
    elif operator == "kirsch":
        kernels = [
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        ]
        edges = np.max([cv2.filter2D(gray, -1, k) for k in kernels], axis=0)
    elif operator == "difference_of_gaussians":
        blur1 = cv2.GaussianBlur(gray, (3, 3), 0)
        blur2 = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.absdiff(blur1, blur2)
    else:
        edges = gray
    return cv2.convertScaleAbs(edges)

# عرض الصورة
def display_image(image):
    img = Image.fromarray(image)
    img = img.resize((400, 400))
    return ImageTk.PhotoImage(img)

# تحميل الصورة
def load_image():
    global img_path, input_image
    img_path = filedialog.askopenfilename()
    if img_path:
        input_image = cv2.imread(img_path)
        input_img_tk = display_image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        panel_original.config(image=input_img_tk)
        panel_original.image = input_img_tk

# تطبيق الخوارزمية
def apply_algorithm():
    global input_image
    algorithm = algo_var.get()
    if input_image is not None and algorithm != "Select Algorithm":
        if algorithm == "Halftoning Threshold":
            result = halftoning_threshold(input_image)
        elif algorithm == "Advanced Halftoning Error Diffusion":
            result = advanced_halftoning_error_diffusion(input_image)
        elif algorithm == "Histogram Equalization":
            result = histogram_equalization(input_image)
        elif algorithm.startswith("Edge Detection"):
            operator = algorithm.split(": ")[1].lower()
            result = edge_detection(input_image, operator)
        else:
            result = input_image
        result_img_tk = display_image(result)
        panel_processed.config(image=result_img_tk)
        panel_processed.image = result_img_tk

# واجهة المستخدم
root = tk.Tk()
root.title("Image Processing Application")

# إطار الصورة الأصلية
frame_original = tk.Frame(root)
frame_original.pack(side="left", padx=10, pady=10)
tk.Label(frame_original, text="Original Image").pack()
panel_original = tk.Label(frame_original)
panel_original.pack()

# إطار الصورة المعالجة
frame_processed = tk.Frame(root)
frame_processed.pack(side="right", padx=10, pady=10)
tk.Label(frame_processed, text="Processed Image").pack()
panel_processed = tk.Label(frame_processed)
panel_processed.pack()

# إطار الأدوات
frame_tools = tk.Frame(root)
frame_tools.pack(side="bottom", pady=10)

btn_load = tk.Button(frame_tools, text="Load Image", command=load_image)
btn_load.grid(row=0, column=0, padx=10)

algo_var = tk.StringVar(value="Select Algorithm")
algo_menu = ttk.Combobox(frame_tools, textvariable=algo_var, state="readonly")
algo_menu["values"] = [
    "Halftoning Threshold",
    "Advanced Halftoning Error Diffusion",
    "Histogram Equalization",
    "Edge Detection: Sobel",
    "Edge Detection: Prewitt",
    "Edge Detection: Kirsch",
    "Edge Detection: Difference of Gaussians",
]
algo_menu.grid(row=0, column=1, padx=10)

btn_apply = tk.Button(frame_tools, text="Apply Algorithm", command=apply_algorithm)
btn_apply.grid(row=0, column=2, padx=10)

root.mainloop()
