# load and format data
from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


# (cr: https://github.com/leftthomas/SRGAN/blob/master/data_utils.py)
def check_format(name):
    return any(name.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def valid_crop_size(crop_size, upscale_factor):
    out_crop_size = crop_size - (crop_size % upscale_factor)
    return out_crop_size


def hr_transform_train(crop_size):
    return Compose([RandomCrop(crop_size), ToTensor(), ])


def lr_transform_train(crop_size, upscale_factor):
    return Compose([ToPILImage(), Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC), ToTensor()])


def display_transform(crop_size, upscale_factor):
    return Compose([ToPILImage(), Resize(400), CenterCrop(400), ToTensor()])


class Train_Dataset(Dataset):
    def __init__(self, data_dir, crop_size, upscale_factor):
        super(Train_Dataset, self).__init__()
        self.image_names = [join(data_dir, x) for x in listdir(data_dir) if check_format(x)]
        crop_size = valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = hr_transform_train(crop_size)
        self.lr_transform = lr_transform_train(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_names[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_names)


class Val_Dataset(Dataset):
    def __init__(self, data_dir, upscale_factor):
        super(Val_Dataset, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_names = [join(data_dir, x) for x in listdir(data_dir) if check_format(x)]

    # (cr: https://github.com/leftthomas/SRGAN/blob/master/data_utils.py)
    def __getitem__(self, index):
        hr_image = Image.open(self.image_names[index])
        w, h = hr_image.size
        crop_size = valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_names)


class Test_Dataset(Dataset):
    def __init__(self, data_dir, upscale_factor):
        super(Test_Dataset, self).__init__()
        self.lr_path = data_dir + '/data/'
        self.hr_path = data_dir + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_names = [join(self.lr_path, x) for x in listdir(self.lr_path) if check_format(x)]
        self.hr_names = [join(self.hr_path, x) for x in listdir(self.lr_path) if check_format(x)]

    def __getitem__(self, index):
        image_name = self.lr_names[index].split('/')[-1]
        lr_image = Image.open(self.lr_names[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_names[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_image = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_image), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_names)