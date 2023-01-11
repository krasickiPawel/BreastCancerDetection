from dataclasses import dataclass, field


@dataclass
class Settings:
    dark_theme = {
        'general_bg': "#081120",
        'general_fg': "white",
        'menu_bg': "#535a77",
        'menu_fg': "white",
        'button_bg': "#081120",
        'button_fg': "white",
        'button_active_bg': "#535a77",
        'button_active_fg': "lightblue",
        'entry_bg': "black",
        'entry_fg': "white",
        'entry_active_fg': "lightblue",
        'text_bg': "black",
        'text_fg': "white",
        'signature_fg': "grey",
    }

    light_theme = {
        'general_bg': "#CBE6F5",
        'general_fg': "#081120",
        'menu_bg': "#CBE6F5",
        'menu_fg': "#081120",
        'button_bg': "#CBE6F5",
        'button_fg': "#081120",
        'button_active_bg': "white",
        'button_active_fg': "lightblue",
        'entry_bg': "white",
        'entry_fg': "black",
        'entry_active_fg': "lightblue",
        'text_bg': "white",
        'text_fg': "black",
        'signature_fg': "grey",
    }

    light_enabled = 1
    theme_picked = light_theme

    # colors
    general_bg: str = theme_picked["general_bg"]
    general_fg: str = theme_picked["general_fg"]
    menu_bg: str = general_bg
    menu_fg: str = general_fg
    button_bg: str = general_bg
    button_fg: str = general_fg
    button_active_bg: str = theme_picked["button_active_bg"]
    button_active_fg: str = theme_picked["button_active_fg"]
    entry_bg: str = theme_picked["entry_bg"]
    entry_fg: str = theme_picked["entry_fg"]
    entry_active_fg: str = theme_picked["entry_active_fg"]
    text_bg: str = theme_picked["text_bg"]
    text_fg: str = theme_picked["text_fg"]
    signature_fg: str = theme_picked["signature_fg"]

    # fonts
    text_font = ('Consolas', 14)
    welcome_text_font = ('Calibri', 8)
    description_font = ('Calibri', 7)
    signature_font = ('Arial', 8)
    button_font = "helvetica 14"

    # window sizes
    geometry_main_window: str = "960x540"
    # geometry_top_level: str = "400x400"

    # defined texts
    title: str = "Breast Cancer Detection - Praca Inżynierska Paweł Krasicki"
    description: str = f'{title} is an application to detect cancer on images using Convolutional Neural Network (' \
                       'CNN). For the correct operation of the application, the existence of a .pth file containing ' \
                       'the trained model is required. By default, the application loads the trained model from the ' \
                       '"models" folder. Please note that the application only supports the work of the doctor and ' \
                       'does not replace him.'
    welcome_text: str = 'Welcome to the breast cancer detection app! The author of the application is Paweł Krasicki. '\
                        'All rights reserved.\nHave a nice use!\n'
    signature: str = "Paweł Krasicki, 2022. All rights reserved."
    button_single_text_short = "Single"
    button_single_text = "Detect cancer in an single image"
    len_button_single_text = len(button_single_text)*10
    button_many_text_short = "Many"
    button_many_text = "Detect cancer on all images in the directory"
    len_button_many_text = len(button_many_text)*8
    button_save_text_short = "Save"
    button_save_text = "Save results"
    len_button_save = len(button_save_text)*10

    # paths
    default_model_path: str = "models/cnn_breast_cancer_full_64_sgd_01_20e.pth"
    # default_model_path: str = "models/cnn_breast_cancer_full_64_sgd_01_20e_wts.pth"
    # default_model_path: str = "models/cnn_breast_cancer_small_64_sgd_01_20_drop.pth"
    logo_path: str = "images/logo2.png"
    # button_single_img_path: str = "images/icons8-play-button-circled-100.png"
    # button_many_img_path: str = "images/icons8-many-50.png"
    button_close_path = "images/xbuttwhite.png"
    # button_close_path = "images/xbutt.png"
    default_path_to_save = "results/"
    # loading_gif_path = "images/loading200.gif"
    button_save_img_path = "images/icons8-save-50.png"
    # button_save_img_path_saved = "images/icons8-save-close-80.png"

    # sizes
    val_batch_size = 4

    # gui elements memory
    frames: dict = field(default_factory=dict)
    buttons: dict = field(default_factory=dict)
    scrolls: dict = field(default_factory=dict)
    texts: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    toolbars: dict = field(default_factory=dict)
    axs: dict = field(default_factory=dict)
    figs: dict = field(default_factory=dict)
    canvas: dict = field(default_factory=dict)
    button_canvas: dict = field(default_factory=dict)

    # result memory
    result_lines: list[str] = field(default_factory=list)
