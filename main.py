from app.frontend import run
import os
import sys


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("app"), relative_path)


def main():
    run()


if __name__ == '__main__':
    resource_path("models/cnn_breast_cancer_full_64_sgd_01_20e.pth")
    resource_path("images/logo2.png")
    resource_path("images/xbuttwhite.png")
    resource_path("images/icons8-save-50.png")

    main()
