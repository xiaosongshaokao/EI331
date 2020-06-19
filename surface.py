import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
import carPlateIdentity
import cv2
from PIL import Image, ImageTk
import threading
import time


class Surface(ttk.Frame):
    pic_path = ""
    viewhigh = 600
    viewwide = 600
    update_time = 0
    thread = None
    thread_run = False
    camera = None
    color_transform = {"green": ("绿牌", "#55FF55"), "yello": ("黄牌", "#FFFF00"), "blue": ("蓝牌", "#6666FF")}   # 颜色转换字典

    def __init__(self, win):    # 初始化窗口
        ttk.Frame.__init__(self, win)
        frame_left = ttk.Frame(self)
        frame_right1 = ttk.Frame(self)
        frame_right2 = ttk.Frame(self)
        win.title("车牌识别")   # 窗口标题
        # win.state("zoomed")   # 窗口的默认状态，zoomed表示全屏，缺省为非全屏
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
        frame_left.pack(side=LEFT, expand=1, fill=BOTH)
        frame_right1.pack(side=TOP, expand=1, fill=tk.Y)
        frame_right2.pack(side=RIGHT, expand=0)

        # 初始化Label控件
        ttk.Label(frame_left, text='原图：').pack(anchor="nw")
        ttk.Label(frame_right1, text='车牌位置：').grid(column=0, row=0, sticky=tk.W)

        # 初始化Button控件
        from_pic_ctl = ttk.Button(frame_right2, text="来自图片", width=20, command=self.from_pic)
        # from_vedio_ctl = ttk.Button(frame_right2, text="来自摄像头", width=20, command=self.from_vedio)
        self.image_ctl = ttk.Label(frame_left)
        self.image_ctl.pack(anchor="nw")

        self.roi_ctl = ttk.Label(frame_right1)
        self.roi_ctl.grid(column=0, row=1, sticky=tk.W)
        ttk.Label(frame_right1, text='识别结果：').grid(column=0, row=2, sticky=tk.W)
        self.r_ctl = ttk.Label(frame_right1, text="")
        self.r_ctl.grid(column=0, row=3, sticky=tk.W)
        self.color_ctl = ttk.Label(frame_right1, text="", width="20")
        self.color_ctl.grid(column=0, row=4, sticky=tk.W)
        # from_vedio_ctl.pack(anchor="se", pady="5")
        from_pic_ctl.pack(anchor="se", pady="5")


    def get_imgtk(self, img_bgr):   # 处理读入的图像
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # 将BGR图转换为RGB图
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)

        # 获取读入图像的宽高
        wide = imgtk.width()
        high = imgtk.height()

        # 如果读入的图像大小超过显示范围，则对图像进行按比例的resize
        if wide > self.viewwide or high > self.viewhigh:
            wide_factor = self.viewwide / wide
            high_factor = self.viewhigh / high
            factor = min(wide_factor, high_factor)
            wide = int(wide * factor)
            if wide <= 0:
                wide = 1
            high = int(high * factor)
            if high <= 0:
                high = 1
            im = im.resize((wide, high), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk

    def show_roi(self, r, roi):  # 显示车牌图像
        if r:   # 如果识别出车牌
            roi = Image.fromarray(roi)
            self.imgtk_roi = ImageTk.PhotoImage(image=roi)
            self.roi_ctl.configure(image=self.imgtk_roi, state='enable')
            self.r_ctl.configure(text=str(r))   # 分字符显示车牌号
            self.update_time = time.time()
        elif self.update_time + 8 < time.time():    # 如果未识别出车牌则不显示该部分
            self.roi_ctl.configure(state='disabled')
            self.r_ctl.configure(text="")



    def from_pic(self):     # 从图片识别车牌
        self.thread_run = False
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg")])
        if self.pic_path:
            img_bgr = carPlateIdentity.imreadex(self.pic_path)  #调用车牌检测函数
            self.imgtk = self.get_imgtk(img_bgr)    # 读取和预处理图片
            self.image_ctl.configure(image=self.imgtk)      # 配置窗口控件（Widgets）
            text,car_plate=carPlateIdentity.recognize(img_bgr)     # 识别图像中的车牌信息
            self.show_roi(text, car_plate)    # 在窗口显示识别的车牌信息

    @staticmethod
    def vedio_thread(self):
        self.thread_run = True
        predict_time = time.time()
        while self.thread_run:
            _, img_bgr = self.camera.read()
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            if time.time() - predict_time > 2:
                r, roi, color = self.predictor.predict(img_bgr)
                self.show_roi(r, roi, color)
                predict_time = time.time()
        print("run end")


def close_window():     # 关闭窗口
    print("destroy")
    if surface.thread_run:
        surface.thread_run = False
        surface.thread.join(2.0)
    win.destroy()


if __name__ == '__main__':
    win = tk.Tk()   # 生成初始化窗口

    surface = Surface(win)  # 新建一个自定义类Surface（在本文件内定义），作用有窗口的初始化和功能的定义
    win.protocol('WM_DELETE_WINDOW', close_window)  # 定义窗口关闭事件
    win.mainloop()  # 进入窗口消息循环
