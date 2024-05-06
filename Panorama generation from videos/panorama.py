import cv2
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import numpy as np

# Get current timestamp
timestamp = int(time.time())


class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path

    #remove the black edges for original panorama
    def remove_black_edges(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3,3), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        
        image[eroded == 0] = (0, 0, 0)
        return image
    #Video frames read by the stitcher
    def stitch_frames(self, frames):
        if not frames:
            raise Exception("No frames are available for stitching")
        stitcher = cv2.createStitcher() if cv2.__version__.startswith('3') else cv2.Stitcher_create()
        status, stitched = stitcher.stitch(frames)
        if status != cv2.Stitcher_OK:
            raise Exception(f'Unable to stitch image, error code:{status}')

        stitched = self.remove_black_edges(stitched)
        return stitched
    #Selection of algorithms used for feature matching
    def detect_and_match(self, frame1, frame2):
        algo = self.algo_var.get()  


        if algo == "SIFT":
            detector = cv2.SIFT_create()
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            #SIFT-FlannBasedMathcer
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif algo == "ORB":
            detector = cv2.ORB_create()
            #ORB-BFMatcher
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

   
        keypoints1, descriptors1 = detector.detectAndCompute(frame1, None)
        keypoints2, descriptors2 = detector.detectAndCompute(frame2, None)


        if descriptors1 is None or descriptors2 is None:
            raise Exception("Descriptors cannot be None.")

       
        if algo == "SIFT":
            matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
            # Lowe's ratio test
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        elif algo == "ORB":
            matches = matcher.match(descriptors1, descriptors2)  
            good_matches = sorted(matches, key=lambda x: x.distance)[:30]  


        matched_image = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, good_matches, None, flags=2)
        return matched_image

    def adjust_frame(self, frame):
        # Convert to HSV colour space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Adjust brightness and contrast (V channel)
        v = hsv[:, :, 2]
        v = np.clip(v * 1.2, 0, 255).astype(np.uint8)  # Increase brightness
        hsv[:, :, 2] = v
        
        # Convert back to BGR colour space
        adjusted_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return adjusted_frame
    #Extract frames from read video
    def extract_frames(self, step=30, apply_blur=False, callback=None):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("Unable to open the video file.")
            frames = []
            idx = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    adjusted_frame = self.adjust_frame(frame)
                    if apply_blur:  
                        adjusted_frame = self.apply_gaussian_blur(adjusted_frame)
                    frames.append(adjusted_frame)
                    if callback:
                        callback(len(frames), int(total_frames / step))
                idx += 1
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process the video: {str(e)}")
        finally:
            cap.release()
        return frames
    
    def extract_two_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
        cap.release()
        if not ret:
            raise Exception("Unable to read enough frames for matching.")
        return frame1, frame2


    def crop_panorama(self, pano):
        # Add a border around the panorama to ensure edge cases are handled correctly
        stitched = cv2.copyMakeBorder(pano, 0, 0, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
        # Convert the image to grayscale to simplify processing
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        # Threshold the image to create a binary image where non-black areas are white
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        # Find the left top corner by scanning diagonally for the first non-black pixel
        leftTopPt = None
        for i in range(1, thresh.shape[0]):
            pixel = thresh[i, i]
            if pixel != 0:
                leftTopPt = (i, i)
                break
        # Find the right top corner by scanning from top-right to bottom-left
        rightTopPt = None
        for i in range(1, thresh.shape[0]):
            x = thresh.shape[1] - i
            y = i
            pixel = thresh[y, x]
            if pixel != 0:
                rightTopPt = (x, y)
                break
        # Find the left bottom corner by scanning from bottom-left to top-right
        leftBottomPt = None
        for i in range(1, thresh.shape[0]):
            x = i
            y = thresh.shape[0] - i
            pixel = thresh[y, x]
            if pixel != 0:
                leftBottomPt = (x, y)
                break
        # Find the right bottom corner by scanning from bottom-right to top-left
        rightBottomPt = None
        for i in range(1, thresh.shape[0]):
            x = thresh.shape[1] - i
            y = thresh.shape[0] - i
            pixel = thresh[y, x]
            if pixel != 0:
                rightBottomPt = (x, y)
                break
        # Determine the maximum and minimum coordinates to define the bounding box
        topMaxY = max(leftTopPt[1], rightTopPt[1])
        leftMaxX = max(leftTopPt[0], leftBottomPt[0])
        rightMinX = min(rightTopPt[0], rightBottomPt[0])
        bottomMinY = min(leftBottomPt[1], rightBottomPt[1])
        # Use the calculated points to redefine the corners of the cropped area
        leftTopPt = (leftMaxX, topMaxY)
        rightTopPt = (rightMinX, topMaxY)
        leftBottomPt = (leftMaxX, bottomMinY)
        rightBottomPt = (rightMinX, bottomMinY)
        # Crop the panorama based on the bounding box and return it
        tempMat = pano[leftTopPt[1]:rightBottomPt[1], leftTopPt[0]:rightBottomPt[0]]
        return tempMat



    def sharpen_frame(self, frame):
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(frame, -1, kernel)
        return sharpened

    def apply_gaussian_blur(self, frame, kernel_size=(3, 3), sigmaX=0):
    
        return cv2.GaussianBlur(frame, kernel_size, sigmaX)


def cv2_to_imgtk(img):
    """Convert cv2 images to ImageTk images"""
    img_pil = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    return imgtk

# GUI section
class PanoramaApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_processor = None  # Initialised to None
        self.sharpen_enabled = False  # Initial state is off
        
        # Flag to initialise whether Gaussian blur is enabled or not
        self.blur_enabled = False
        self.setup_gui()

    def setup_gui(self):
        window_width = 1200
        window_height = 800

        style = ttk.Style()
        style.configure('TButton', font=('Arial', 12), padding=6)
        style.configure('TProgressbar', thickness=10, relief='flat', background='#3de23d')

        self.frame = tk.Frame(self.window)
        self.frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        #Setup Buttons
        self.btn_browse = ttk.Button(self.frame, text="Uploading Videos", command=self.open_video, width=20)
        self.btn_browse.grid(row=1, column=0, padx=10, pady=10)

        self.btn_stitch = ttk.Button(self.frame, text="Generating panoramas", command=self.stitch_video_async, width=20, state='disabled')
        self.btn_stitch.grid(row=2, column=0, padx=10, pady=10)

        self.progress = ttk.Progressbar(self.frame, length=100, mode='determinate', style='TProgressbar')
        self.progress.grid(row=3, column=0, padx=10, pady=10, sticky='ew')

        self.btn_preview = ttk.Button(self.frame, text="Feature Matching Preview", command=self.preview_video, width=20)
        self.btn_preview.grid(row=4, column=0, padx=10, pady=10)
        # Add Gaussian Blur checkbox
        self.chk_blur = ttk.Checkbutton(self.frame, text="Enable Gaussian Blur", command=self.toggle_blur)
        self.chk_blur.grid(row=5, column=0, padx=10, pady=10)
        # Add sharpening checkbox
        self.chk_sharpen = ttk.Checkbutton(self.frame, text="Enable sharpening", command=self.toggle_sharpen)
        self.chk_sharpen.grid(row=6, column=0, padx=10, pady=10)
        self.algo_var = tk.StringVar()
        # Add algorithm selection interactions
        self.algo_var.set("SIFT") 
        self.algo_dropdown = ttk.OptionMenu(self.frame, self.algo_var, "SIFT", "SIFT", "ORB")
        self.algo_dropdown.grid(row=7, column=0, padx=10, pady=10)

        self.btn_help = ttk.Button(self.frame, text="HELP", command=self.show_help, width=20)
        self.btn_help.grid(row=8, column=0, padx=10, pady=10)


        self.status_label = ttk.Label(self.frame, text="")
        self.status_label.grid(row=0, column=0, padx=10, pady=10)


        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.window.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.window.mainloop()
    #Switch
    def toggle_sharpen(self):
        self.sharpen_enabled = not self.sharpen_enabled

    def toggle_blur(self):
        self.blur_enabled = not self.blur_enabled
    #Feature matching preview
    def preview_video(self):
        if not hasattr(self, 'video_path') or not self.video_path:
            messagebox.showerror("Error", "Please upload a video file first.")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Unable to open the video file.")
            return

        #Switching based on algorithm selection
        algo = self.algo_var.get()
        if algo == "SIFT":
            detector = cv2.SIFT_create()
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif algo == "ORB":
            detector = cv2.ORB_create()
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        ret, last_frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "There are no frames in the video file to read.")
            return
        last_keypoints, last_descriptors = detector.detectAndCompute(last_frame, None)


        preview_window = tk.Toplevel(self.window)
        preview_window.title("Feature Matching Preview")
        label = tk.Label(preview_window)
        label.pack()

        def update_frame():
            nonlocal last_frame, last_keypoints, last_descriptors
            ret, frame = cap.read()
            if not ret:
                cap.release()
                preview_window.destroy()
                return

            keypoints, descriptors = detector.detectAndCompute(frame, None)
            if algo == "SIFT":
                matches = matcher.knnMatch(last_descriptors, descriptors, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
            elif algo == "ORB":
                matches = matcher.match(last_descriptors, descriptors)
                good_matches = sorted(matches, key=lambda x: x.distance)[:30]

            matched_frame = cv2.drawMatches(last_frame, last_keypoints, frame, keypoints, good_matches, None, flags=2)

            imgtk = cv2_to_imgtk(cv2.cvtColor(matched_frame, cv2.COLOR_BGR2RGB))
            label.imgtk = imgtk
            label.configure(image=imgtk)
            label.after(10, update_frame)  # Cyclic update frames

        update_frame()  # Initiate video frame update



    def show_help(self):
        help_window = tk.Toplevel(self.window)
        help_window.title("HELP")

        # Create a text box and a scroll bar
        text_scroll = tk.Scrollbar(help_window)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        help_text = tk.Text(help_window, height=10, width=50, wrap=tk.WORD, yscrollcommand=text_scroll.set, font=("Arial", 12))
        help_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Insert help content in the text box
        help_content = """
    Uploading Videos: Select the video file to process.
    Generating panoramas: Frames are extracted from the uploaded video and stitched into a panorama.
    Feature Matching Preview: Displays the frames in the video and their feature matches.
    Enable sharpening: Sharpening effects are applied when processing images.
    Enable Gaussian Blur: Apply Gaussian blur to reduce noise and smooth the image.
    Selection algorithm: Choose to use SIFT or ORB algorithm for feature detection and matching.
    """
        help_text.insert(tk.END, help_content)
        help_text.config(state=tk.DISABLED)  # Set the text box to read-only mode
        text_scroll.config(command=help_text.yview)

        # Resize and reposition the help window
        help_window.update_idletasks()  # Update the window state to get the correct size
        window_width = help_window.winfo_reqwidth()
        window_height = help_window.winfo_reqheight()
        position_right = self.window.winfo_x() + (self.window.winfo_width() // 2 - window_width // 2)
        position_down = self.window.winfo_y() + (self.window.winfo_height() // 2 - window_height // 2)

        help_window.geometry(f"{window_width}x{window_height}+{position_right}+{position_down}")


    def stitch_video_async(self):
        # Disable setting modifications and buttons
        self.chk_blur.config(state='disabled')
        self.chk_sharpen.config(state='disabled')
        self.algo_dropdown.config(state='disabled')  # Disable algorithm selection drop-down menu
        self.btn_stitch.config(state='disabled')  # Disable the Generate Panorama button
        self.btn_browse.config(state='disabled')  # Disable the upload video button
        threading.Thread(target=self.stitch_video).start()

    def stitch_video(self):
        if not hasattr(self, 'video_path') or not self.video_path:
            messagebox.showerror("Error", "Please upload a video file first.")
            self.enable_controls()  # If there is no video path, re-enable the control immediately
            return

        processor = VideoProcessor(self.video_path)
        try:
            frames = processor.extract_frames(step=30, apply_blur=self.blur_enabled, callback=self.update_progress)
            panorama = processor.stitch_frames(frames)

            if panorama is not None:
                if self.sharpen_enabled:
                    panorama = processor.sharpen_frame(panorama)
                self.show_panorama(panorama)
            else:
                raise Exception("Generating panoramas failed.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            # Re-enable the control whether it succeeds or fails
            self.enable_controls()
            self.progress['value'] = 0

    def enable_controls(self):
        """Re-enable all user interface controls"""
        self.chk_blur.config(state='normal')
        self.chk_sharpen.config(state='normal')
        self.algo_dropdown.config(state='normal')  # Re-enable the algorithm selection drop-down menu
        self.btn_stitch.config(state='normal')  # Re-enable the Generate Panorama button
        self.btn_browse.config(state='normal')  # Re-enable the upload video button

    def update_progress(self, frame_idx, total_frames):
        progress = int((frame_idx / total_frames) * 100)
        self.progress['value'] = progress
        self.window.update_idletasks()
        # Update status information when progress bar reaches 100%
        if progress == 100:
            self.status_label.config(text="Frame extraction complete, please wait for the stitching process.")

    def open_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            self.video_processor = VideoProcessor(self.video_path)  # Create a VideoProcessor instance
            self.btn_stitch.config(state=tk.NORMAL)
            messagebox.showinfo("Uploading successfully", "The video file has been uploaded successfully and processing can begin.")
        else:
            messagebox.showinfo("Upload failure", "No video file is selected.")


    def show_panorama(self, panorama):
        save_directory = 'run'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Cropping panoramas
        panorama = self.video_processor.crop_panorama(panorama)

        # Generate current timestamp
        timestamp = int(time.time())  # Update timestamp

        # Save the cropped panorama
        file_name = f"panorama_{timestamp}.jpg"
        save_path = os.path.join(save_directory, file_name)
        cv2.imwrite(save_path, panorama)
        print(f"Cropped panorama saved to {save_path}")
        messagebox.showinfo("The panorama was generated successfully", f"The panorama has been successfully saved to：{save_path}")

        # Create and display the cropped panorama window
        self.create_panorama_window(panorama, save_path)

    def create_panorama_window(self, panorama, save_path):
        panorama_window = tk.Toplevel(self.window)
        panorama_window.title("Panorama")
        panorama_window.protocol("WM_DELETE_WINDOW", lambda: self.on_close_panorama(panorama_window))

        # Get screen size
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        # Get image size
        image_height, image_width, _ = panorama.shape

        # Calculate scaling to fit the screen
        scale_width = screen_width / image_width
        scale_height = screen_height / image_height
        scale = min(scale_width, scale_height)

        # Resize images
        if scale < 1:  # Scale only when the image is larger than the screen
            image_width = int(image_width * scale)
            image_height = int(image_height * scale)
            panorama = cv2.resize(panorama, (image_width, image_height), interpolation=cv2.INTER_AREA)

        panorama_photo = cv2_to_imgtk(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        label_panorama = tk.Label(panorama_window, image=panorama_photo)
        label_panorama.image = panorama_photo  # Keep references to images from being reclaimed by the rubbish collector
        label_panorama.pack(fill=tk.BOTH, expand=tk.YES)

        # Set the window size and position so that it is centred
        center_x = int((screen_width - image_width) / 2)
        center_y = int((screen_height - image_height) / 2)
        panorama_window.geometry(f'{image_width}x{image_height}+{center_x}+{center_y}')
    

    def show_feature_matches(self):
        try:
            frame1, frame2 = self.video_processor.extract_two_frames()
            matched_image = self.video_processor.detect_and_match(frame1, frame2)
            self.display_image_in_gui(matched_image)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_image_in_gui(self, img):
        imgtk = cv2_to_imgtk(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.label_image.config(image=imgtk)
        self.label_image.image = imgtk


    def on_close_panorama(self, panorama_window):
        # Destroy panorama window
        panorama_window.destroy()
        # Refocus on the main window
        self.window.focus_set()

# Main programme start
if __name__ == '__main__':
    # Create a window and run it
    root = tk.Tk()
    app = PanoramaApp(root, "20321406")