from app.backend_init_model import create_resnet_model
from app.backend_train import train as universal_train
from app.backend_loaders import load_train_test_dataset
from app.backend_transformations import transform_singe_image, train_test_trans
import torch
import matplotlib.pyplot as plt


class DeepLearning:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.model = None
        self.dls = None
        self.optimizer = None
        self.acc_hist = None
        self.best_acc = None
        self.test_size = None
        self.batch_size = None
        self.train_test_dir = None
        self.validate_dir = None
        self.num_epochs = None
        self.transformations = None

    def validate(self):
        pass

    def load_valid_data(self):
        pass

    def load_single_valid_data(self, path):
        return self.transformations.transform_singe_image(path)


class TrainCNN(DeepLearning):
    def __init__(self, default):
        super().__init__()
        self.num_classes = None
        self.pretrained = None
        self.fine_tuning = None
        self.input_required_size = None
        self.default = default
        self.check_default()
        self.init_train_data_params()

    def check_default(self):
        if self.default:
            self.init_model()

    def init_model(self, num_classes=2, pretrained=True, fine_tuning=True, input_required_size=224):
        self.model = create_resnet_model(num_classes, pretrained, fine_tuning)
        self.input_required_size = input_required_size
        self.transformations = train_test_trans(input_required_size)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def init_train_data_params(self, num_epochs=10, test_size=.2, batch_size=16):
        self.num_epochs = num_epochs
        self.test_size = test_size
        self.batch_size = batch_size

    def set_optim(self, optimizer):
        self.optimizer = optimizer

    def set_loss_func(self, loss_func):
        self.loss_func = loss_func

    def set_train_dir(self, train_test_dir):
        self.train_test_dir = train_test_dir

    def set_valid_dir(self, path):
        self.validate_dir = path

    def load_train_data(self):
        self.dls = load_train_test_dataset(
            self.train_test_dir,
            ts=self.test_size,
            batch_size=self.batch_size,
            input_required_size=self.input_required_size
        )

    def train(self):
        if self.model is None or self.dls is None or self.loss_func is None or self.optimizer is None:
            return False

        self.model, self.acc_hist, self.best_acc = universal_train(
            self.device,
            self.model,
            self.dls,
            self.loss_func,
            self.optimizer,
            self.num_epochs
        )
        return self.model, self.acc_hist, self.best_acc

    def save_state_dict(self, save_model_path):
        torch.save(self.model.state_dict(), save_model_path)

    def save_model(self, save_model_path):
        torch.save(self.model, save_model_path)

    def load_state_dict(self, load_path):
        if self.model is None:
            self.model = create_resnet_model(2, pretrained=False, fine_tuning=False)
        self.model.load_state_dict(torch.load(load_path))
        self.model.eval()

    def load_model(self, load_path):
        self.model = torch.load(load_path)
        self.model.eval()


def main():
    # train_dir_path = r"C:\IDC_regular_ps50_idx5\8955"
    train_dir_path = "../train_test_balanced"
    # train_dir_path = "../train_test_small"
    num_epochs = 20
    # num_epochs = 20
    fine_tuning = True
    batch_size = 64
    learning_rate = 0.01

    cnn = TrainCNN(default=False)
    cnn.init_model(fine_tuning=fine_tuning)
    cnn.init_train_data_params(num_epochs, batch_size=batch_size)
    cnn.set_train_dir(train_dir_path)

    # optimizer = torch.optim.Adam(cnn.model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(cnn.model.parameters(), lr=learning_rate, momentum=0.9)

    cnn.set_optim(optimizer)
    cnn.load_train_data()

    result = cnn.train()

    cnn.save_model("cnn_breast_cancer_full_64_sgd_01_20e.pth")
    cnn.save_state_dict("cnn_breast_cancer_full_64_sgd_01_20e_wts.pth")
    # print(result)


def check_file():
    entire_model_path = "../models/cnn_breast_cancer_full_64_sgd_01_20e.pth"
    model = torch.load(entire_model_path)
    model.eval()

    valid_path = "../validate/8867_idx5_x1051_y1051_class1.png"
    tensor = transform_singe_image(valid_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tensor = tensor.to(device)
    model = model.to(device)
    output = model.forward(tensor)

    probs = torch.nn.functional.softmax(output, dim=1)
    conf, classes = torch.max(probs, 1)

    conf = conf.item()
    conf *= 100
    prediction = classes.item()

    x_label = "{0} at confidence {1:.2f}%".format(prediction, conf)

    image = plt.imread(valid_path)
    plt.xlabel(x_label)
    plt.imshow(image)
    print(x_label)
    plt.show()


if __name__ == '__main__':
    check_file()
    # main()

