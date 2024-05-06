import cv2
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import numpy as np

# 获取当前时间戳
timestamp = int(time.time())


class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path

    
    def remove_black_edges(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3,3), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        
        image[eroded == 0] = (0, 0, 0)
        return image
    
    def stitch_frames(self, frames):
        if not frames:
            raise Exception("No frames are available for stitching")
        stitcher = cv2.createStitcher() if cv2.__version__.startswith('3') else cv2.Stitcher_create()
        status, stitched = stitcher.stitch(frames)
        if status != cv2.Stitcher_OK:
            raise Exception(f'Unable to stitch image, error code:{status}')

        stitched = self.remove_black_edges(stitched)
        return stitched

    def detect_and_match(self, frame1, frame2):
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
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 调整亮度和对比度（V通道）
        v = hsv[:, :, 2]
        v = np.clip(v * 1.2, 0, 255).astype(np.uint8)  # 提高亮度
        hsv[:, :, 2] = v
        
        # 转换回BGR色彩空间
        adjusted_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return adjusted_frame

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
        stitched = cv2.copyMakeBorder(pano, 0, 0, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        leftTopPt = None
        for i in range(1, thresh.shape[0]):
            pixel = thresh[i, i]
            if pixel != 0:
                leftTopPt = (i, i)
                break

        rightTopPt = None
        for i in range(1, thresh.shape[0]):
            x = thresh.shape[1] - i
            y = i
            pixel = thresh[y, x]
            if pixel != 0:
                rightTopPt = (x, y)
                break

        leftBottomPt = None
        for i in range(1, thresh.shape[0]):
            x = i
            y = thresh.shape[0] - i
            pixel = thresh[y, x]
            if pixel != 0:
                leftBottomPt = (x, y)
                break

        rightBottomPt = None
        for i in range(1, thresh.shape[0]):
            x = thresh.shape[1] - i
            y = thresh.shape[0] - i
            pixel = thresh[y, x]
            if pixel != 0:
                rightBottomPt = (x, y)
                break

        topMaxY = max(leftTopPt[1], rightTopPt[1])
        leftMaxX = max(leftTopPt[0], leftBottomPt[0])
        rightMinX = min(rightTopPt[0], rightBottomPt[0])
        bottomMinY = min(leftBottomPt[1], rightBottomPt[1])

        leftTopPt = (leftMaxX, topMaxY)
        rightTopPt = (rightMinX, topMaxY)
        leftBottomPt = (leftMaxX, bottomMinY)
        rightBottomPt = (rightMinX, bottomMinY)

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
    """将cv2图像转换为ImageTk图像"""
    img_pil = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    return imgtk

# GUI部分
class PanoramaApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_processor = None  # 初始化为 None
        self.sharpen_enabled = False  # 初始状态为关闭
        
        # 初始化高斯模糊是否启用的标志
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

        self.btn_browse = ttk.Button(self.frame, text="Uploading Videos", command=self.open_video, width=20)
        self.btn_browse.grid(row=1, column=0, padx=10, pady=10)

        self.btn_stitch = ttk.Button(self.frame, text="Generating panoramas", command=self.stitch_video_async, width=20, state='disabled')
        self.btn_stitch.grid(row=2, column=0, padx=10, pady=10)

        self.progress = ttk.Progressbar(self.frame, length=100, mode='determinate', style='TProgressbar')
        self.progress.grid(row=3, column=0, padx=10, pady=10, sticky='ew')

        self.btn_preview = ttk.Button(self.frame, text="Feature Matching Preview", command=self.preview_video, width=20)
        self.btn_preview.grid(row=4, column=0, padx=10, pady=10)
        
        self.chk_blur = ttk.Checkbutton(self.frame, text="Enable Gaussian Blur", command=self.toggle_blur)
        self.chk_blur.grid(row=5, column=0, padx=10, pady=10)
        # 添加锐化复选框
        self.chk_sharpen = ttk.Checkbutton(self.frame, text="Enable sharpening", command=self.toggle_sharpen)
        self.chk_sharpen.grid(row=6, column=0, padx=10, pady=10)
        self.algo_var = tk.StringVar()
        
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
    
    def toggle_sharpen(self):
        self.sharpen_enabled = not self.sharpen_enabled

    def toggle_blur(self):
        self.blur_enabled = not self.blur_enabled
    def preview_video(self):
        if not hasattr(self, 'video_path') or not self.video_path:
            messagebox.showerror("Error", "Please upload a video file first.")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Unable to open the video file.")
            return


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
            label.after(10, update_frame)  # 循环更新帧

        update_frame()  # 启动视频帧更新



    def show_help(self):
        help_window = tk.Toplevel(self.window)
        help_window.title("HELP")

        # 创建一个文本框以及滚动条
        text_scroll = tk.Scrollbar(help_window)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        help_text = tk.Text(help_window, height=10, width=50, wrap=tk.WORD, yscrollcommand=text_scroll.set, font=("Arial", 12))
        help_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # 在文本框中插入帮助内容
        help_content = """
    Uploading Videos: Select the video file to process.
    Generating panoramas: Frames are extracted from the uploaded video and stitched into a panorama.
    Feature Matching Preview: Displays the frames in the video and their feature matches.
    Enable sharpening: Sharpening effects are applied when processing images.
    Enable Gaussian Blur: Apply Gaussian blur to reduce noise and smooth the image.
    Selection algorithm: Choose to use SIFT or ORB algorithm for feature detection and matching.
    """
        help_text.insert(tk.END, help_content)
        help_text.config(state=tk.DISABLED)  # 设置文本框为只读模式
        text_scroll.config(command=help_text.yview)

        # 调整帮助窗口的尺寸和位置
        help_window.update_idletasks()  # 更新窗口状态以获取正确的尺寸
        window_width = help_window.winfo_reqwidth()
        window_height = help_window.winfo_reqheight()
        position_right = self.window.winfo_x() + (self.window.winfo_width() // 2 - window_width // 2)
        position_down = self.window.winfo_y() + (self.window.winfo_height() // 2 - window_height // 2)

        help_window.geometry(f"{window_width}x{window_height}+{position_right}+{position_down}")


    def stitch_video_async(self):
        # 禁用设置修改和按钮
        self.chk_blur.config(state='disabled')
        self.chk_sharpen.config(state='disabled')
        self.algo_dropdown.config(state='disabled')  # 禁用算法选择下拉菜单
        self.btn_stitch.config(state='disabled')  # 禁用生成全景图按钮
        self.btn_browse.config(state='disabled')  # 禁用上传视频按钮
        threading.Thread(target=self.stitch_video).start()

    def stitch_video(self):
        if not hasattr(self, 'video_path') or not self.video_path:
            messagebox.showerror("Error", "Please upload a video file first.")
            self.enable_controls()  # 如果没有视频路径，立即重新启用控件
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
            # 无论成功或失败，重新启用控件
            self.enable_controls()
            self.progress['value'] = 0

    def enable_controls(self):
        """重新启用所有的用户界面控件"""
        self.chk_blur.config(state='normal')
        self.chk_sharpen.config(state='normal')
        self.algo_dropdown.config(state='normal')  # 重新启用算法选择下拉菜单
        self.btn_stitch.config(state='normal')  # 重新启用生成全景图按钮
        self.btn_browse.config(state='normal')  # 重新启用上传视频按钮

    def update_progress(self, frame_idx, total_frames):
        progress = int((frame_idx / total_frames) * 100)
        self.progress['value'] = progress
        self.window.update_idletasks()
        # 当进度条到达100%时，更新状态信息
        if progress == 100:
            self.status_label.config(text="Frame extraction complete, please wait for the stitching process.")

    def open_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            self.video_processor = VideoProcessor(self.video_path)  # 创建VideoProcessor实例
            self.btn_stitch.config(state=tk.NORMAL)
            messagebox.showinfo("Uploading successfully", "The video file has been uploaded successfully and processing can begin.")
        else:
            messagebox.showinfo("Upload failure", "No video file is selected.")


    def show_panorama(self, panorama):
        save_directory = 'run'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # 裁剪全景图
        panorama = self.video_processor.crop_panorama(panorama)

        # 生成当前时间戳
        timestamp = int(time.time())  # 更新时间戳

        # 保存裁剪后的全景图
        file_name = f"panorama_{timestamp}.jpg"
        save_path = os.path.join(save_directory, file_name)
        cv2.imwrite(save_path, panorama)
        print(f"Cropped panorama saved to {save_path}")
        messagebox.showinfo("The panorama was generated successfully", f"The panorama has been successfully saved to：{save_path}")

        # 创建并显示裁剪后的全景图窗口
        self.create_panorama_window(panorama, save_path)

    def create_panorama_window(self, panorama, save_path):
        panorama_window = tk.Toplevel(self.window)
        panorama_window.title("Panorama")
        panorama_window.protocol("WM_DELETE_WINDOW", lambda: self.on_close_panorama(panorama_window))

        # 获取屏幕尺寸
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        # 获取图像尺寸
        image_height, image_width, _ = panorama.shape

        # 计算缩放比例，以适应屏幕
        scale_width = screen_width / image_width
        scale_height = screen_height / image_height
        scale = min(scale_width, scale_height)

        # 调整图像尺寸
        if scale < 1:  # 仅当图像比屏幕大时才进行缩放
            image_width = int(image_width * scale)
            image_height = int(image_height * scale)
            panorama = cv2.resize(panorama, (image_width, image_height), interpolation=cv2.INTER_AREA)

        panorama_photo = cv2_to_imgtk(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        label_panorama = tk.Label(panorama_window, image=panorama_photo)
        label_panorama.image = panorama_photo  # 保持对图像的引用，避免被垃圾收集器回收
        label_panorama.pack(fill=tk.BOTH, expand=tk.YES)

        # 设置窗口尺寸和位置，使其居中
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
        # 销毁全景图窗口
        panorama_window.destroy()
        # 重新聚焦到主窗口
        self.window.focus_set()

# 主程序开始
if __name__ == '__main__':
    # 创建窗口并运行
    root = tk.Tk()
    app = PanoramaApp(root, "20321406")