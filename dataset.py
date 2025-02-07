from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Compose

class MyDataset(Dataset):
    def __init__(self, data_path, csv_file, is_train, transform=None):  # data_path="data"

        self.diagnosis = pd.read_csv(csv_file)
        self.categories = [0, 1, 2, 3, 4]
        self.transform = transform
        self.labels = []
        self.image_paths = []
        for index in range(len(self.diagnosis)):
            if os.path.exists(os.path.join(data_path, "{}.png".format(self.diagnosis.iloc[index, 0]))):
                self.image_paths.append(os.path.join(data_path, "{}.png".format(self.diagnosis.iloc[index, 0])))
                self.labels.append(self.diagnosis.iloc[index, 1])

        # print(len(self.diagnosis))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
        # image.show()
        # print(self.image_paths[item])
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)
        return image, label

    # def __getitem__(self, item):
    #     image = cv2.imread(self.image_paths[item])
    #     label = self.labels[item]
    #     if self.transform:
    #         image = self.transform(image)
    #     return image, label


if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
    dataset = MyDataset(data_path="data/train_images", csv_file="train-2.csv", is_train=False, transform=transform)
    print(len(dataset))
    image, label = dataset[85]
    print(image.shape, label)

    # dataset = MyDataset(data_path="data", csv_file="train.csv", is_train=True, transform=None)
    # image, label = dataset[500]
    # # print(image.shape, label)
    # # image = np.reshape(image, (3, 1958, 2588))
    # # print(image.shape, label)
    # # image = np.transpose(image, (1, 2, 0))
    # # print(image.shape, label)
    # image = cv2.resize(image, (1294, 979))
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
