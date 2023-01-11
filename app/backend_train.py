import copy
import torch
import time


def train(device, model, data_loaders_phases, loss_func, optimizer, num_epochs=1):
    model = model.to(device)
    test_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    torch.cuda.empty_cache()

    start = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in data_loaders_phases:
            model.train() if phase == 'train' else model.eval()
            current_loss = 0.0
            current_corrects = 0

            for imgs, labels in data_loaders_phases[phase]:
                imgs = imgs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(imgs)
                    loss = loss_func(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                current_loss += loss.item() * imgs.size(0)
                current_corrects += torch.sum(preds == labels.data)

            epoch_loss = current_loss / len(data_loaders_phases[phase].dataset)
            epoch_acc = current_corrects.double() / len(data_loaders_phases[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "test":
                test_acc_history.append(epoch_acc)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))

    for i, h in enumerate(test_acc_history):
        print(f"Acc history {i}:", h)
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, test_acc_history, best_acc
