import torch
from app.backend_transformations import transform_singe_image
from app.backend_loaders import load_val_dir


class FileManager:
    @staticmethod
    def save_file(file_path: str, result_list: list[str]):
        with open(file_path, 'w') as file:
            file.writelines(result_list)


class CNN:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.is_cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.is_cuda_available else torch.device('cpu')

    def load_model(self) -> None:
        if self.model is None:
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
            self.model = self.model.to(self.device)

    def check_single(self, img_path: str) -> tuple:
        img_tensor = transform_singe_image(img_path)
        if img_tensor is None:
            return None, None

        img_tensor = img_tensor.to(self.device)
        result = self.model.forward(img_tensor)

        decision = torch.nn.functional.softmax(result, dim=1)

        confidence, is_cancer = torch.max(decision, 1)
        confidence = confidence.item()
        confidence *= 100
        is_cancer = is_cancer.item()
        return is_cancer, confidence

    def check_many_experimental(self, dir_path, print_text, val_batch_size):
        val_dataloader = load_val_dir(dir_path, batch_size=val_batch_size)
        if val_dataloader is None:
            return None
        val_dataloader_len = len(val_dataloader.dataset)
        pred_list = []
        for idx, (images, paths) in enumerate(val_dataloader):
            # images = images.to(self.device)

            for j, (img, path) in enumerate(zip(images, paths)):
                img = img.to(self.device)

                activations = self.model.forward(img)
                decision = torch.nn.functional.softmax(activations, dim=1)
                confidence, is_cancer = torch.max(decision, 1)
                confidence = confidence.item()
                pred = is_cancer.item()
                confidence *= 100

                pred_list.append((path, pred))
                image_num = idx * val_batch_size + j + 1
                print_text(f"\n{image_num}/{val_dataloader_len}")
                print_text(f"{path}:")
                is_cancer_txt = "CANCER" if pred else "OK"
                is_cancer_txt += f' at confidence {confidence}'
                print_text(is_cancer_txt, is_cancer_txt.lower())

        return pred_list

    def check_many(self, dir_path, print_text, val_batch_size):
        val_dataloader = load_val_dir(dir_path, batch_size=val_batch_size)
        if val_dataloader is None:
            return None
        val_dataloader_len = len(val_dataloader.dataset)
        pred_list = []
        for idx, (images, paths) in enumerate(val_dataloader):
            images = images.to(self.device)

            outputs = self.model(images)
            _, predictions = torch.max(outputs, 1)

            predictions = list(predictions.cpu().detach().numpy())
            for idx_inner, (path, pred) in enumerate(zip(paths, predictions)):
                pred_list.append((path, pred))
                image_num = idx*val_batch_size + idx_inner + 1
                print_text(f"\n{image_num}/{val_dataloader_len}")
                print_text(f"{path}:")
                is_cancer_txt = "CANCER" if pred else "OK"
                print_text(is_cancer_txt, is_cancer_txt.lower())

        return pred_list
