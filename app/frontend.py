import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from app.settings import Settings
from app.frontend_custom_frames import ImgFrame
from app.backend import CNN, FileManager
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.image as mpimg
import ctypes as ct
import os


settings = Settings()
cnn = CNN(settings.default_model_path)
root = tk.Tk()


def dark_title_bar(window, mode=2):
    """
    setting window title bar to dark mode
    MORE INFO:
    https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute
    """
    window.update()
    set_window_attribute = ct.windll.dwmapi.DwmSetWindowAttribute
    get_parent = ct.windll.user32.GetParent
    hwnd = get_parent(window.winfo_id())
    val = mode
    val = ct.c_int(val)
    set_window_attribute(hwnd, 20, ct.byref(val), 4)


def init_backend():
    cnn.load_model()


def print_text(text, warning="normal"):
    text_results.insert(tk.INSERT, f"\n{text}", warning)
    text_results.see("end")
    root.update()


def single_img_command():
    root.after(10, start_single_image_process)


def many_img_command():
    root.after(10, start_many_images_process)


def start_single_image_process():
    unpack_plot_command()
    print_text("\nSelect an image to detect breast cancer.")
    img_path = filedialog.askopenfilename()
    is_file = os.path.exists(img_path)
    if not is_file:
        print_text("FILE DOES NOT EXIST!!\n", "error")
        return
    set_save_button_disabled()
    set_run_buttons_disabled()
    print_text("Loading trained model...")
    init_backend()
    print_text("Searching cancer...")
    is_cancer, confidence = cnn.check_single(img_path)
    if is_cancer is None:
        print_text("SELECTED FILE IS NOT AN IMAGE!\n", "error")
        set_run_buttons_enabled()
        return
    print_text(f"Decision made:\n\n{img_path}:")
    confidence = "{:.2f}%".format(confidence)
    is_cancer_txt = "CANCER" if is_cancer else "OK"
    info = f"{is_cancer_txt} at confidence {confidence}\n"
    print_text(info, is_cancer_txt.lower())
    info_to_be_saved = f"{img_path}:\n{info}\n"
    settings.result_lines.append(info_to_be_saved)
    set_save_button_enabled()
    pack_plot(img_path, info, is_cancer)
    set_run_buttons_enabled()


def start_many_images_process():
    unpack_plot_command()
    print_text("\nSelect the image folder for breast cancer analysis.")
    dir_path = filedialog.askdirectory(title='Select directory with files to analysis')
    is_dir = os.path.exists(dir_path)
    if not is_dir:
        print_text("DIRECTORY DOES NOT EXIST!!\n", "error")
        return
    set_save_button_disabled()
    set_run_buttons_disabled()
    text_results.delete('1.0', tk.END)
    print_text("Loading trained model...")
    init_backend()
    print_text("Searching cancer...")
    predictions = cnn.check_many(dir_path, print_text, settings.val_batch_size)
    if predictions is None:
        print_text("There are more than 1000 photos in a folder, you cannot analyze as many photos at once!", "error")
        set_save_button_enabled()
        return
    settings.result_lines = []
    any_cancer = 0
    for path, is_cancer in predictions:
        if is_cancer:
            any_cancer += 1
            cancer_txt = "CANCER"
        else:
            cancer_txt = "OK"
        line_txt = f"{path}:\n{cancer_txt}\n"
        settings.result_lines.append(line_txt)
    print_text("Searching cancer process finished!")
    if any_cancer:
        print_text(f"Breast cancer was detected in {any_cancer} images!", "cancer")
    else:
        print_text("None of the images contained breast cancer.", "ok")
    set_save_button_enabled()
    set_run_buttons_enabled()


def save_results():
    if not os.path.exists(settings.default_path_to_save):
        os.makedirs(settings.default_path_to_save)
    save_path = filedialog.asksaveasfilename(
        defaultextension="txt",
        filetypes=[("Text files", "*.txt")],
        initialdir=settings.default_path_to_save,
        title="Enter filename",
    )
    try:
        FileManager.save_file(save_path, settings.result_lines)
    except FileNotFoundError:
        print_text("File name not specified, saving aborted", "error")
        return
    del settings.result_lines[:]
    settings.result_lines = []
    print_text("Results saved!", "save")
    set_save_button_disabled()


def set_save_button_enabled():
    butt = settings.buttons.get("button_save")
    butt.config(state="normal")


def set_save_button_disabled():
    butt = settings.buttons.get("button_save")
    butt.config(state="disable")


def set_run_buttons_enabled():
    butt1 = settings.buttons.get("button_single")
    butt1.config(state="normal")
    butt2 = settings.buttons.get("button_many")
    butt2.config(state="normal")


def set_run_buttons_disabled():
    butt1 = settings.buttons.get("button_single")
    butt1.config(state="disable")
    butt2 = settings.buttons.get("button_many")
    butt2.config(state="disable")


def check_device():
    available = cnn.is_cuda_available
    device_info = "CUDA available!" if available else "CPU only!"
    print_text(f"{device_info}", "device")
    settings.labels.get("label_cuda_cpu").config(text=device_info)


def toggle_fullscreen(event):
    root.attributes("-fullscreen", True)


def end_fullscreen(event):
    root.attributes("-fullscreen", False)


def unpack_plot_command():
    if settings.figs.get("fig_tk"):
        settings.figs["fig_tk"].place_forget()
        del(settings.figs["fig_tk"])
    if settings.toolbars.get("toolbar"):
        settings.toolbars["toolbar"].destroy()
    button_close_canvas.place_forget()
    text_results.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.9)
    root.update()


def pack_plot(img_path, x_label_txt, is_cancer=True):
    plt.rcParams['figure.facecolor'] = settings.general_bg
    plt.rcParams['axes.facecolor'] = settings.general_bg

    x_label_txt = f"{x_label_txt}\n"
    text_results.place_forget()
    fig = plt.figure(figsize=(6, 6), dpi=100)
    figure_canvas = FigureCanvasTkAgg(fig, frame_results)
    fig_tk = figure_canvas.get_tk_widget()

    toolbar = NavigationToolbar2Tk(figure_canvas, frame_results)
    toolbar.config(background=settings.general_bg)
    toolbar._message_label.config(background='grey')
    toolbar.update()

    img = mpimg.imread(img_path)
    red_font_dict = {'family': 'arial', 'color': '#F44336', 'weight': 'normal', 'size': 12}
    green_font_dict = {'family': 'arial', 'color': '#32CD32', 'weight': 'normal', 'size': 12}
    font_dict = red_font_dict if is_cancer else green_font_dict

    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(img_path, color=settings.general_fg)
    ax.spines['bottom'].set_color(settings.general_fg)
    ax.tick_params(axis='x', colors=settings.general_fg)
    ax.spines['left'].set_color(settings.general_fg)
    ax.tick_params(axis='y', colors=settings.general_fg)
    ax.spines['right'].set_color(settings.general_bg)
    ax.spines['top'].set_color(settings.general_bg)
    ax.set_xlabel(x_label_txt, fontdict=font_dict)
    ax.imshow(img)
    root.update()

    fig_tk.place(relx=0.0, rely=0.05, relwidth=0.9, relheight=0.85)
    toolbar.place(relx=0.5, rely=0.95, relwidth=0.5, relheight=0.05, anchor="center")

    button_close_canvas.place(relx=0.91, rely=0.05, relwidth=0.04, relheight=0.02)

    settings.figs["fig_tk"] = fig_tk
    settings.toolbars["toolbar"] = toolbar


def resize(e):
    if button_many.winfo_width() < settings.len_button_many_text:
        button_many.config(text=settings.button_many_text_short, font="helvetica 11")
        button_single.config(text=settings.button_single_text_short, font="helvetica 11")
        button_save.config(text=settings.button_save_text_short, font="helvetica 10")
        label_cuda_cpu.config(font="helvetica 9")
    else:
        button_many.config(text=settings.button_many_text, font="helvetica 13")
        button_single.config(text=settings.button_single_text, font="helvetica 13")
        button_save.config(text=settings.button_save_text, font="helvetica 12")
        label_cuda_cpu.config(font="helvetica 12")

    h = button_save.winfo_height()
    if h < 30:
        button_save.config(image=button_save_img_super_small)
    elif h < 40:
        button_save.config(image=button_save_img_small)
    else:
        button_save.config(image=button_save_img)


def root_update(event):
    root.update()


def temp_switch_theme(e):
    switch_theme()


def run():
    root.mainloop()


def change_theme(theme_picked):
    settings.general_bg = theme_picked["general_bg"]
    settings.general_fg = theme_picked["general_fg"]
    settings.menu_bg = settings.general_bg
    settings.menu_fg = settings.general_fg
    settings.button_bg = settings.general_bg
    settings.button_fg = settings.general_fg
    settings.button_active_bg = theme_picked["button_active_bg"]
    settings.button_active_fg = theme_picked["button_active_fg"]
    settings.entry_bg = theme_picked["entry_bg"]
    settings.entry_fg = theme_picked["entry_fg"]
    settings.entry_active_fg = theme_picked["entry_active_fg"]
    settings.text_bg = theme_picked["text_bg"]
    settings.text_fg = theme_picked["text_fg"]
    settings.signature_fg = theme_picked["signature_fg"]

    for frame in settings.frames.values():
        frame.config(bg=settings.general_bg)
    for button in settings.buttons.values():
        button.config(
            bg=settings.button_bg,
            fg=settings.button_fg,
            activeforeground=settings.button_active_fg,
            activebackground=settings.button_active_bg,
        )
    for txt in settings.texts.values():
        txt.config(
            bg=settings.text_bg,
            fg=settings.text_fg
        )
    for label in settings.labels.values():
        label.config(bg=settings.general_bg)

    settings.frames["frame_control_panel_logo"].canvas.config(bg=settings.general_bg)
    button_save.config(bg=settings.general_bg)
    button_close.config(bg=settings.general_bg)
    button_close_canvas.config(bg=settings.general_bg)


def switch_theme():
    if settings.light_enabled:
        change_theme(settings.dark_theme)
        settings.light_enabled = False
    else:
        change_theme(settings.light_theme)
        settings.light_enabled = True
    root.update()


frame_control_panel = tk.Frame(root, bg=settings.general_bg)
frame_control_panel.place(relx=0.0, rely=0.0, relwidth=0.3, relheight=1)
settings.frames["frame_control_panel"] = frame_control_panel

frame_control_panel_input_buttons = tk.Frame(frame_control_panel, bg=settings.general_bg)
frame_control_panel_input_buttons.place(relx=0.0, rely=0.0, relwidth=1, relheight=0.45)
settings.frames["frame_control_panel_input_buttons"] = frame_control_panel_input_buttons

button_single = tk.Button(
    frame_control_panel_input_buttons,
    text=settings.button_single_text,
    bg=settings.button_bg,
    fg=settings.button_fg,
    font=settings.button_font,
    command=single_img_command,
    activebackground=settings.general_bg
)
button_single.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.25)
settings.buttons["button_single"] = button_single

button_many = tk.Button(
    frame_control_panel_input_buttons,
    text=settings.button_many_text,
    bg=settings.button_bg,
    fg=settings.button_fg,
    font=settings.button_font,
    command=many_img_command,
    activebackground=settings.general_bg,
)
button_many.place(relx=0.1, rely=0.38, relwidth=0.8, relheight=0.25)
settings.buttons["button_many"] = button_many
label_cuda_cpu = tk.Label(
    frame_control_panel_input_buttons,
    fg="#039BE5",
    bg=settings.general_bg,
    text="",
    font="helvetica 12"
)
label_cuda_cpu.place(relx=0.5, rely=0.72, anchor="center")
settings.labels["label_cuda_cpu"] = label_cuda_cpu

button_save_img_small = Image.open(settings.button_save_img_path)
button_save_img_small = button_save_img_small.resize((35, 35), Image.Resampling.LANCZOS)
button_save_img_super_small = button_save_img_small.resize((25, 25), Image.Resampling.LANCZOS)
button_save_img_small = ImageTk.PhotoImage(button_save_img_small)
button_save_img_super_small = ImageTk.PhotoImage(button_save_img_super_small)

button_save_img = tk.PhotoImage(file=settings.button_save_img_path)
button_save = tk.Button(
    frame_control_panel_input_buttons,
    image=button_save_img,
    text=settings.button_save_text,
    bg=settings.button_bg,
    fg=settings.button_fg,
    font="helvetica 12",
    command=save_results,
    borderwidth=0,
    activebackground=settings.general_bg,
)
button_save.place(relx=0.5, anchor="center", rely=0.9, relheight=0.15, relwidth=0.5)
settings.buttons["button_save"] = button_save


frame_control_panel_logo = ImgFrame(
    frame_control_panel,
    bg_img_path=settings.logo_path,
    bg=settings.general_bg
)
frame_control_panel_logo.place(relx=0.2, rely=0.5, relwidth=0.6, relheight=0.49)
settings.frames["frame_control_panel_logo"] = frame_control_panel_logo


frame_results = tk.Frame(root, bg=settings.general_bg)
frame_results.place(relx=0.3, rely=0.0, relwidth=0.7, relheight=0.92)
settings.frames["frame_results"] = frame_results
text_results = tk.Text(
    frame_results,
    bg=settings.text_bg,
    fg=settings.text_fg,
    font=settings.text_font
)
text_results.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.9)
settings.texts["text_results"] = text_results

frame_info = tk.Frame(root, bg=settings.general_bg)
frame_info.place(relx=0.3, rely=0.92, relwidth=0.7, relheight=0.15)
settings.frames["frame_info"] = frame_info
label_signature = tk.Label(
    frame_info,
    fg=settings.signature_fg,
    bg=settings.general_bg,
    text=settings.signature
)
label_signature.place(relx=0.5, rely=0.25, anchor="center")
settings.labels["label_signature"] = label_signature


button_close_canvas = tk.Canvas(frame_results, bg=settings.general_bg, highlightthickness=0)
button_close_img_to_be_resized = Image.open(settings.button_close_path)
resized_button_img = button_close_img_to_be_resized.resize((30, 30), Image.Resampling.LANCZOS)
new_button_img = ImageTk.PhotoImage(resized_button_img)
button_close = tk.Button(
    frame_results,
    image=new_button_img,
    bg=settings.general_bg, borderwidth=0,
    activebackground=settings.general_bg,
    command=unpack_plot_command
)
button_close_window = button_close_canvas.create_window(0, 0, anchor="nw", window=button_close)


text_results.insert(tk.INSERT, "Cancer searching process has not started yet.")
check_device()
text_results.bind("<Key>", lambda e: "break")
text_results.tag_config("cancer", foreground="#F44336")
text_results.tag_config("ok", foreground="lime green")
text_results.tag_config("save", foreground="#C88E05")
text_results.tag_config("device", foreground="#039BE5")
text_results.tag_config("normal")
text_results.tag_config("error", foreground="violet red")
set_save_button_disabled()


root.title(settings.title)
root.state('zoomed')

root.bind("<F11>", toggle_fullscreen)
root.bind("<Escape>", end_fullscreen)
root.bind("<b>", temp_switch_theme)


root.geometry(settings.geometry_main_window)

root.bind("<Configure>", resize)
root.bind("<<root_update>>", root_update)


root.minsize(480, 300)


