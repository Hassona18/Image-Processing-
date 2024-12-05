import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from scipy.ndimage import gaussian_filter

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")

        # Initialize variables
        self.uploaded_image = None
        self.processed_image = None

        # GUI Layout
        self.setup_gui()

    def setup_gui(self):
        # Frames
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, padx=10, pady=10)

        display_frame = tk.Frame(self.root)
        display_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Buttons
        tk.Button(control_frame, text="Upload Image", command=self.upload_image).pack(pady=5)
        tk.Button(control_frame, text="Apply Gaussian Blur", command=self.gaussian_blur).pack(pady=5)
        tk.Button(control_frame, text="Laplacian Edge Detection", command=self.laplacian_edge_detection).pack(pady=5)
        tk.Button(control_frame, text="Rotate Image", command=self.rotate_image).pack(pady=5)
        tk.Button(control_frame, text="Zoom In/Out", command=lambda: self.zoom_image(scale=1.5)).pack(pady=5)
        tk.Button(control_frame, text="Save Processed Image", command=self.save_image).pack(pady=5)

        # Canvas for displaying images
        self.original_canvas = tk.Canvas(display_frame, width=300, height=300, bg="gray")
        self.original_canvas.pack(side=tk.LEFT, padx=10)

        self.processed_canvas = tk.Canvas(display_frame, width=300, height=300, bg="gray")
        self.processed_canvas.pack(side=tk.RIGHT, padx=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
        if file_path:
            self.uploaded_image = Image.open(file_path)
            self.processed_image = None
            self.display_image(self.uploaded_image, "original")

    def display_image(self, image, canvas_type):
        # Resize image to fit in canvas
        resized_image = image.resize((300, 300))
        tk_image = ImageTk.PhotoImage(resized_image)

        if canvas_type == "original":
            self.original_canvas.image = tk_image
            self.original_canvas.create_image(150, 150, image=tk_image)
        elif canvas_type == "processed":
            self.processed_canvas.image = tk_image
            self.processed_canvas.create_image(150, 150, image=tk_image)

    def gaussian_blur(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image.convert("L"))
            blurred = gaussian_filter(img_array, sigma=2)  # Adjust sigma for intensity
            self.processed_image = Image.fromarray(blurred.astype(np.uint8))
            self.display_image(self.processed_image, "processed")

    def laplacian_edge_detection(self):
        if self.uploaded_image:
            img_array = np.array(self.uploaded_image.convert("L"))
            laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            edges = self.apply_filter(img_array, laplacian_kernel)
            edges = np.clip(edges, 0, 255)
            self.processed_image = Image.fromarray(edges.astype(np.uint8))
            self.display_image(self.processed_image, "processed")

    def apply_filter(self, img_array, kernel):
        from scipy.signal import convolve2d
        return convolve2d(img_array, kernel, mode="same", boundary="symm")

    def rotate_image(self):
        if self.uploaded_image:
            self.processed_image = self.uploaded_image.rotate(90)  # Change angle as needed
            self.display_image(self.processed_image, "processed")

    def zoom_image(self, scale=1.5):
        if self.uploaded_image:
            width, height = self.uploaded_image.size
            new_size = (int(width * scale), int(height * scale))
            zoomed = self.uploaded_image.resize(new_size)
            self.processed_image = zoomed
            self.display_image(self.processed_image, "processed")

    def save_image(self):
        if self.processed_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"),
                                                                ("All files", "*.*")])
            if file_path:
                self.processed_image.save(file_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
