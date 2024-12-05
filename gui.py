import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import median_filter, gaussian_filter


# Image Processing Application
class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Project")
        self.root.geometry("1200x800")

        self.uploaded_image = None
        self.processed_image = None

        self.create_gui()

    def create_gui(self):
        # Buttons for image operations
        tk.Button(self.root, text="Upload Image", command=self.upload_image).pack()

        self.original_frame = tk.LabelFrame(self.root, text="Original Image")
        self.original_frame.pack(side=tk.LEFT, padx=20, pady=20)
        self.original_label = tk.Label(self.original_frame)
        self.original_label.pack()

        self.processed_frame = tk.LabelFrame(self.root, text="Processed Image")
        self.processed_frame.pack(side=tk.RIGHT, padx=20, pady=20)
        self.processed_label = tk.Label(self.processed_frame)
        self.processed_label.pack()

        operations_frame = tk.LabelFrame(self.root, text="Operations")
        operations_frame.pack(side=tk.TOP, padx=20, pady=20)

        # Buttons for operations
        tk.Button(
            operations_frame,
            text="Convert to Grayscale",
            command=self.convert_grayscale,
        ).pack(pady=5)
        tk.Button(
            operations_frame, text="Apply Threshold", command=self.apply_threshold
        ).pack(pady=5)
        tk.Button(
            operations_frame, text="Simple Halftone", command=self.simple_halftone
        ).pack(pady=5)
        tk.Button(
            operations_frame, text="Advanced Halftone", command=self.advanced_halftone
        ).pack(pady=5)
        tk.Button(
            operations_frame,
            text="Histogram Equalization",
            command=self.histogram_equalization,
        ).pack(pady=5)

        tk.Button(
            operations_frame,
            text="Sobel Edge Detection",
            command=self.edge_detection_sobel,
        ).pack(pady=5)
        tk.Button(
            operations_frame,
            text="Prewitt Edge Detection",
            command=self.edge_detection_prewitt,
        ).pack(pady=5)
        tk.Button(
            operations_frame,
            text="Kirsch Edge Detection",
            command=self.edge_detection_kirsch,
        ).pack(pady=5)

        tk.Button(
            operations_frame,
            text="Apply High-pass Filter",
            command=self.high_pass_filter,
        ).pack(pady=5)
        tk.Button(
            operations_frame, text="Apply Low-pass Filter", command=self.low_pass_filter
        ).pack(pady=5)
        tk.Button(
            operations_frame, text="Apply Median Filter", command=self.median_filter_op
        ).pack(pady=5)

        tk.Button(
            operations_frame, text="Invert Image", command=self.invert_image
        ).pack(pady=5)
        tk.Button(
            operations_frame, text="Add Image and Copy", command=self.add_images
        ).pack(pady=5)
        tk.Button(
            operations_frame,
            text="Subtract Image and Copy",
            command=self.subtract_images,
        ).pack(pady=5)

        # Histogram Based Segmentation
        tk.Button(
            operations_frame,
            text="Manual Histogram Segmentation",
            command=self.manual_histogram_segmentation,
        ).pack(pady=5)
        tk.Button(
            operations_frame,
            text="Histogram Peak Segmentation",
            command=self.histogram_peak_segmentation,
        ).pack(pady=5)
        tk.Button(
            operations_frame,
            text="Histogram Valley Segmentation",
            command=self.histogram_valley_segmentation,
        ).pack(pady=5)
        tk.Button(
            operations_frame,
            text="Adaptive Histogram Segmentation",
            command=self.adaptive_histogram_segmentation,
        ).pack(pady=5)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")] )
        if file_path:
            self.uploaded_image = Image.open(file_path)
            self.display_image(self.uploaded_image, "original")

    def display_image(self, img, frame):
        img_resized = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img_resized)
        if frame == "original":
            self.original_label.config(image=img_tk)
            self.original_label.image = img_tk
        else:
            self.processed_label.config(image=img_tk)
            self.processed_label.image = img_tk

    # 1. Convert Image to Grayscale
    def convert_grayscale(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image)
            grayscale = np.dot(img_array[..., :3], [0.299, 0.587, 0.114]).astype(
                np.uint8
            )
            self.processed_image = Image.fromarray(grayscale)
            self.display_image(self.processed_image, "processed")

    # 2. Apply Threshold (based on the mean value)
    def apply_threshold(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image)
            threshold_value = np.mean(img_array)
            thresholded = (img_array > threshold_value) * 255
            self.processed_image = Image.fromarray(thresholded.astype(np.uint8))
            self.display_image(self.processed_image, "processed")

    # 3. Simple Halftone (Threshold)
    def simple_halftone(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image.convert("L"))
            halftone = ((img_array // 128) * 255).astype(np.uint8)
            self.processed_image = Image.fromarray(halftone)
            self.display_image(self.processed_image, "processed")

    # 4. Advanced Halftone (Error Diffusion)
    def advanced_halftone(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image.convert("L"))
            halftone = img_array.copy()
            for i in range(img_array.shape[0] - 1):
                for j in range(img_array.shape[1] - 1):
                    old_pixel = img_array[i, j]
                    new_pixel = 255 * (old_pixel > 128)
                    halftone[i, j] = new_pixel
                    quant_error = old_pixel - new_pixel
                    if j + 1 < img_array.shape[1]:
                        halftone[i, j + 1] += quant_error * 7 / 16
                    if i + 1 < img_array.shape[0]:
                        halftone[i + 1, j] += quant_error * 5 / 16
                    if i + 1 < img_array.shape[0] and j + 1 < img_array.shape[1]:
                        halftone[i + 1, j + 1] += quant_error * 1 / 16
            self.processed_image = Image.fromarray(halftone)
            self.display_image(self.processed_image, "processed")

    # 5. Histogram Equalization
    def histogram_equalization(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image.convert("L"))
            hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
            equalized = cdf_normalized[img_array]
            self.processed_image = Image.fromarray(equalized.astype(np.uint8))
            self.display_image(self.processed_image, "processed")

    # 6. Edge Detection using Sobel Operator
    def edge_detection_sobel(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image.convert("L"))
            sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

            gx = self.apply_filter(img_array, sobel_x)
            gy = self.apply_filter(img_array, sobel_y)

            magnitude = np.sqrt(gx ** 2 + gy ** 2)
            magnitude = (magnitude / magnitude.max()) * 255
            self.processed_image = Image.fromarray(magnitude.astype(np.uint8))
            self.display_image(self.processed_image, "processed")

    # 7. Edge Detection using Prewitt Operator
    def edge_detection_prewitt(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image.convert("L"))
            prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

            gx = self.apply_filter(img_array, prewitt_x)
            gy = self.apply_filter(img_array, prewitt_y)

            magnitude = np.sqrt(gx ** 2 + gy ** 2)
            magnitude = (magnitude / magnitude.max()) * 255
            self.processed_image = Image.fromarray(magnitude.astype(np.uint8))
            self.display_image(self.processed_image, "processed")

    # 8. Edge Detection using Kirsch Compass Masks
    def edge_detection_kirsch(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image.convert("L"))

        # Kirsch masks for edge detection
            kirsch_masks = [
                np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
                np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
                np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
                np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
                np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
                np.array([[-3, -3, -3], [-3, 0, 5], [5, 5, 5]]),
                np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
                np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
            ]

        # List to hold edge magnitudes for each mask
            edge_magnitudes = []

        # Apply each Kirsch mask to the image
            for mask in kirsch_masks:
                gx = self.apply_filter(img_array, mask)
                edge_magnitudes.append(np.abs(gx))

            # Find the maximum edge magnitude for each pixel
            max_magnitude = np.max(edge_magnitudes, axis=0)

            # Normalize the values to ensure they fall within the valid range (0 to 255)
            max_magnitude = (max_magnitude / max_magnitude.max()) * 255
            max_magnitude = max_magnitude.astype(np.uint8)

            # Convert the result to an image and display
            self.processed_image = Image.fromarray(max_magnitude)
            self.display_image(self.processed_image, "processed")

    def apply_filter(self, img_array, kernel):
        # Perform the 2D convolution with the given kernel
        return convolve2d(img_array, kernel, mode='same', boundary='symm')



    def high_pass_filter(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image.convert("L"))
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            high_pass = self.apply_filter(img_array, kernel)
            self.processed_image = Image.fromarray(np.uint8(high_pass))
            self.display_image(self.processed_image, "processed")

    def low_pass_filter(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image.convert("L"))
            kernel = np.ones((5, 5)) / 25
            low_pass = self.apply_filter(img_array, kernel)
            self.processed_image = Image.fromarray(np.uint8(low_pass))
            self.display_image(self.processed_image, "processed")

    def median_filter_op(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image.convert("L"))
            median_filtered = median_filter(img_array, size=3)
            self.processed_image = Image.fromarray(median_filtered)
            self.display_image(self.processed_image, "processed")

    def invert_image(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image)
            inverted = 255 - img_array
            self.processed_image = Image.fromarray(inverted)
            self.display_image(self.processed_image, "processed")

    def add_images(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image)
            added = img_array + img_array
            self.processed_image = Image.fromarray(
                np.clip(added, 0, 255).astype(np.uint8)
            )
            self.display_image(self.processed_image, "processed")

    def subtract_images(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image)
            subtracted = img_array - img_array
            self.processed_image = Image.fromarray(
                np.clip(subtracted, 0, 255).astype(np.uint8)
            )
            self.display_image(self.processed_image, "processed")

    # Histogram-Based Segmentation Techniques
    def manual_histogram_segmentation(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image.convert("L"))
            hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
            threshold = int(np.mean(bins))
            segmented = (img_array > threshold) * 255
            self.processed_image = Image.fromarray(segmented.astype(np.uint8))
            self.display_image(self.processed_image, "processed")

    def histogram_peak_segmentation(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image.convert("L"))
            hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
            peak = np.argmax(hist)
            threshold = bins[peak]
            segmented = (img_array > threshold) * 255
            self.processed_image = Image.fromarray(segmented.astype(np.uint8))
            self.display_image(self.processed_image, "processed")

    def histogram_valley_segmentation(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image.convert("L"))
            hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
            valley = np.argmin(hist)
            threshold = bins[valley]
            segmented = (img_array > threshold) * 255
            self.processed_image = Image.fromarray(segmented.astype(np.uint8))
            self.display_image(self.processed_image, "processed")

    def adaptive_histogram_segmentation(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image.convert("L"))
            adaptive_threshold = np.mean(img_array)
            segmented = (img_array > adaptive_threshold) * 255
            self.processed_image = Image.fromarray(segmented.astype(np.uint8))
            self.display_image(self.processed_image, "processed")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
