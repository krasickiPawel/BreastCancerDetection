import tkinter as tk
from PIL import Image, ImageTk
import time
from threading import Thread


class ImgFrame(tk.Frame):
    def __init__(self, master, bg_img_path, bg, *args):
        tk.Frame.__init__(self, master, bg=bg, *args)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.image = Image.open(bg_img_path)
        self.img_copy = self.image.copy()
        self.background_image = ImageTk.PhotoImage(self.image)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw", image=self.background_image)
        self.canvas.bind('<Configure>', self._resize_image)

    def _resize_image(self, event):
        new_width = event.width
        new_height = event.height
        self.image = self.img_copy.resize((new_width, new_height))
        self.background_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.background_image)


class GifFrame(tk.Frame):
    def __init__(self, master, gif_path, bg, *args):
        super().__init__(master, bg=bg, *args)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.gif_path = gif_path
        self.gif = Image.open(gif_path)
        self.gif_images = None
        self.image_on_canvas = None
        self.play = False

    def init_gif_images(self):
        if self.gif_images is None:
            self.gif_images = [tk.PhotoImage(file=self.gif_path,
                                             format=f'gif -index {i}') for i in range(self.gif.n_frames)]
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor='nw', image=self.gif_images[0])

    def freeze_gif(self):
        self.play = False

    def animate_gif(self):
        self.play = True
        thread = Thread(target=self.animate_thread)
        thread.daemon = True
        thread.start()

    def animate_thread(self):
        for img in self.gif_images:
            self.canvas.itemconfig(self.image_on_canvas, image=img)
            time.sleep(0.005)
            if not self.play:
                return
        self.animate_thread()
