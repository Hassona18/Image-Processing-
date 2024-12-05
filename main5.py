import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.ndimage import median_filter


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Application")
        
        self.input_image = None
        self.processed_image = None
        
        self.create_gui()

    def create_gui(self):
        # Original Frame
        frame_original = tk.Frame(self.root)
        frame_original.pack(side="left", padx=10, pady=10)
        tk.Label(frame_original, text="Original Image").pack()
        self.panel_original = tk.Label(frame_original)
        self.panel_original.pack()

        # Processed Frame
        frame_processed = tk.Frame(self.root)
        frame_processed.pack(side="right", padx=10, pady=10)
        tk.Label(frame_processed, text="Processed Image").pack()
        self.panel_processed = tk.Label(frame_processed)
        self.panel_processed.pack()

        # Tools Frame
        frame_tools = tk.Frame(self.root)
        frame_tools.pack(side="bottom", pady=10)

        # Load Image Button
        btn_load = tk.Button(frame_tools, text="Load Image", command=self.load_image)
        btn_load.grid(row=0, column=0, padx=10)

        # Algorithm Selection Dropdown
        self.algo_var = tk.StringVar(value="Select Algorithm")
        self.algo_menu = ttk.Combobox(frame_tools, textvariable=self.algo_var, state="readonly")
        self.algo_menu["values"] = [
            "Convert to Grayscale",
            "Apply Threshold",
            "Simple Halftone",
            "Advanced Halftone",
            "Histogram Equalization",
            "Sobel Edge Detection",
            "Prewitt Edge Detection", 
            "Kirsch Edge Detection",
            "High-pass Filter",
            "Low-pass Filter",
            "Median Filter",
            "Invert Image",
            "Add Image and Copy",
            "Subtract Image and Copy",
            "Manual Histogram Segmentation",
            "Histogram Peak Segmentation",
            "Histogram Valley Segmentation",
            "Adaptive Histogram Segmentation"
        ]
        self.algo_menu.grid(row=0, column=1, padx=10)

        # Apply Algorithm Button
        btn_apply = tk.Button(frame_tools, text="Apply Algorithm", command=self.apply_algorithm)
        btn_apply.grid(row=0, column=2, padx=10)

    def load_image(self):
        img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.bmp")])
        if img_path:
            self.input_image = cv2.imread(img_path)
            self.display_image(self.input_image, self.panel_original)

    def display_image(self, image, panel):
        # Convert to RGB if the image is in BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Create PIL image and resize
        img = Image.fromarray(image_rgb)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        
        # Update panel
        panel.config(image=img_tk)
        panel.image = img_tk

    def apply_algorithm(self):
        if self.input_image is None:
            return

        algorithm = self.algo_var.get()
        img_array = np.array(self.input_image)

        # Convert to grayscale for processing
        if len(img_array.shape) == 3:
            gray_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray_array = img_array

        # Apply selected algorithm
        if algorithm == "Convert to Grayscale":
            processed = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        elif algorithm == "Apply Threshold":
            _, processed = cv2.threshold(gray_array, np.mean(gray_array), 255, cv2.THRESH_BINARY)
        elif algorithm == "Simple Halftone":
            processed = ((gray_array // 128) * 255).astype(np.uint8)
        elif algorithm == "Advanced Halftone":
            processed = self.advanced_halftone(gray_array)
        elif algorithm == "Histogram Equalization":
            processed = cv2.equalizeHist(gray_array)
        elif algorithm == "Sobel Edge Detection":
            processed = self.edge_detection_sobel(gray_array)
        elif algorithm == "Prewitt Edge Detection":
            processed = self.edge_detection_prewitt(gray_array)
        elif algorithm == "Kirsch Edge Detection":
            processed = self.edge_detection_kirsch(gray_array)
        elif algorithm == "High-pass Filter":
            processed = self.high_pass_filter(gray_array)
        elif algorithm == "Low-pass Filter":
            processed = self.low_pass_filter(gray_array)
        elif algorithm == "Median Filter":
            processed = median_filter(gray_array, size=3)
        elif algorithm == "Invert Image":
            processed = 255 - img_array
        elif algorithm == "Add Image and Copy":
            processed = np.clip(img_array + img_array, 0, 255).astype(np.uint8)
        elif algorithm == "Subtract Image and Copy":
            processed = np.clip(img_array - img_array, 0, 255).astype(np.uint8)
        elif algorithm == "Manual Histogram Segmentation":
            hist, _ = np.histogram(gray_array.flatten(), 256, [0, 256])
            threshold = int(np.mean(hist))
            processed = (gray_array > threshold) * 255
        elif algorithm == "Histogram Peak Segmentation":
            hist, bins = np.histogram(gray_array.flatten(), 256, [0, 256])
            peak = np.argmax(hist)
            threshold = bins[peak]
            processed = (gray_array > threshold) * 255
        elif algorithm == "Histogram Valley Segmentation":
            hist, bins = np.histogram(gray_array.flatten(), 256, [0, 256])
            valley = np.argmin(hist)
            threshold = bins[valley]
            processed = (gray_array > threshold) * 255
        elif algorithm == "Adaptive Histogram Segmentation":
            adaptive_threshold = np.mean(gray_array)
            processed = (gray_array > adaptive_threshold) * 255
        else:
            return

        self.display_image(processed, self.panel_processed)

    def apply_filter(self, img_array, kernel):
        return convolve2d(img_array, kernel, mode="same", boundary="wrap")

    def advanced_halftone(self, img_array):
        halftone = img_array.copy().astype(float)
        for i in range(img_array.shape[0] - 1):
            for j in range(img_array.shape[1] - 1):
                old_pixel = halftone[i, j]
                new_pixel = 255 if old_pixel > 128 else 0
                halftone[i, j] = new_pixel
                quant_error = old_pixel - new_pixel
                
                if j + 1 < img_array.shape[1]:
                    halftone[i, j + 1] += quant_error * 7 / 16
                if i + 1 < img_array.shape[0]:
                    halftone[i + 1, j] += quant_error * 5 / 16
                if i + 1 < img_array.shape[0] and j + 1 < img_array.shape[1]:
                    halftone[i + 1, j + 1] += quant_error * 1 / 16
        
        return halftone.astype(np.uint8)

    def edge_detection_sobel(self, img_array):
        sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        gx = self.apply_filter(img_array, sobel_x)
        gy = self.apply_filter(img_array, sobel_y)

        magnitude = np.sqrt(gx**2 + gy**2)
        return ((magnitude / magnitude.max()) * 255).astype(np.uint8)

    def edge_detection_prewitt(self, img_array):
        prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

        gx = self.apply_filter(img_array, prewitt_x)
        gy = self.apply_filter(img_array, prewitt_y)

        magnitude = np.sqrt(gx**2 + gy**2)
        return ((magnitude / magnitude.max()) * 255).astype(np.uint8)

    def edge_detection_kirsch(self, img_array):
        kirsch_masks = [
        np.array([[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]], dtype=np.float32),  # N
        np.array([[ 5,  5, -3], [ 5,  0, -3], [-3, -3, -3]], dtype=np.float32),  # NE
        np.array([[ 5, -3, -3], [ 5,  0, -3], [ 5, -3, -3]], dtype=np.float32),  # E
        np.array([[-3, -3, -3], [ 5,  0, -3], [ 5,  5, -3]], dtype=np.float32),  # SE
        np.array([[-3, -3, -3], [-3,  0, -3], [ 5,  5,  5]], dtype=np.float32),  # S
        np.array([[-3, -3, -3], [-3,  0,  5], [-3,  5,  5]], dtype=np.float32),  # SW
        np.array([[-3, -3,  5], [-3,  0,  5], [-3, -3,  5]], dtype=np.float32),  # W
        np.array([[-3,  5,  5], [-3,  0,  5], [-3, -3, -3]], dtype=np.float32),  # NW
    ]


        edge_magnitudes = []
        for mask in kirsch_masks:
            gx = self.apply_filter(img_array, mask)
            edge_magnitudes.append(np.abs(gx))

        return np.max(edge_magnitudes, axis=0).astype(np.uint8)

    def high_pass_filter(self, img_array):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return np.clip(self.apply_filter(img_array, kernel), 0, 255).astype(np.uint8)

    def low_pass_filter(self, img_array):
        kernel = np.ones((5, 5)) / 25
        return np.clip(self.apply_filter(img_array, kernel), 0, 255).astype(np.uint8)


def main():
    root = tk.Tk()
    root.title("Image Processing Application")
    app = ImageProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()