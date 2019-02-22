import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from Datacollection import result1
from Textpreprocessing import list_tokenized_train

class FlickrDataset(data.Dataset):
    def __init__(self, root, result,list_tokenized_train, transform=None):
        self.root = root
        self.result = result
        self.transform = transform
        self.tokens = np.asarray(list_tokenized_train)

    def __getitem__(self, index):
        path = os.path.join(self.root,self.result[index,0])
        image = cv2.imread(path)
        image = cv2.resize(image,(224,224))

        if self.transform:
          image = self.transform(image)
        caption = np.asarray(self.tokens[index])
        caption = torch.Tensor(caption)
        return image, caption
    def __len__(self):
        return 31783
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
dataset = FlickrDataset(dir,result2,list_tokenized_train,transform = transform)
def collate_fn(data):

    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=32,collate_fn = collate_fn)
