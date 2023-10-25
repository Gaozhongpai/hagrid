import argparse
import logging
from typing import Optional, Tuple

import torch.optim
import torch.utils
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from classifier.dataset import GestureDataset
from classifier.preprocess import get_transform
from classifier.train import TrainClassifier
from classifier.utils import build_model, collate_fn, set_random_state

from glob import glob
import os, cv2, math
import numpy as np

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)


def flip_img(img):
    """Flip rgb images or masks.
    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img

def rotate_image(image, angle, image_center, scale, xy_coords=None, dsize=224):
    angle = math.degrees(angle)
    # image_size = int(224 / scale) 
    ## image_center [col, row]
    inter_flag = cv2.INTER_LINEAR # cv2.INTER_CUBIC 
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
    xy_news = xy_coords
    if xy_coords is not None:
        xy_news = np.transpose(np.dot(rot_mat, np.transpose(xy_coords)))
        xy_news = (xy_news - np.array(image_center)) + dsize//2 #### important
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=inter_flag, borderMode=cv2.BORDER_CONSTANT)
    img_pad = cv2.copyMakeBorder(result, dsize//2, dsize//2, dsize//2, dsize//2, cv2.BORDER_CONSTANT)
    image_center = (int(image_center[0]), int(image_center[1]))
    image_crop = img_pad[image_center[1]:image_center[1]+dsize, image_center[0]:image_center[0]+dsize]
    return image_crop, xy_news

class HandGestureDataset(object):
    def __init__(self, is_train=True):
        torch.manual_seed(0)
        hanco_path = "../HanCo"
        # train = "train" if is_train else "val"
        lists_positive = glob(os.path.join(hanco_path, "uppcrop/*/*/", "*.jpeg"))
        lists_positive = [file for file in lists_positive if 'Others' not in file 
                                    and "_10/" not in file and "_11/" not in file]
        lists_negative = glob(os.path.join(hanco_path, "uppcrop/Others/*", "*.jpeg"))
    
        train_negative_list = torch.randperm(len(lists_negative))[:int(len(lists_negative)*0.9)]
        test__negative_list = torch.randperm(len(lists_negative))[int(len(lists_negative)*0.9):]
        if is_train:
            lists_positive = [name for name in lists_positive if "/ZW/" not in name]
            lists_negative = [lists_negative[idx] for idx in train_negative_list]
        else:
            lists_positive = [name for name in lists_positive if "/ZW/" in name]
            lists_negative = [lists_negative[idx] for idx in test__negative_list]
        self.lists = lists_positive + lists_negative
        self.targets = torch.zeros(len(self.lists)).long()
        for i, name in enumerate(self.lists):
            if "_00/" in name: ## Move
                self.targets[i] = 1 
            elif "_01/" in name or "_02/" in name: ## Zoom
                self.targets[i] = 2
            elif "_03/" in name or "_04/" in name or "_05/" in name \
                or "_06/" in name or "_07/" in name or "_08/" in name:  ## RotateX
                self.targets[i] = 3
            # elif "_06/" in name or "_07/" in name or "_08/" in name:  ## RotateY
            #     self.targets[i] = 4
            elif "_09/" in name: # or "_10/" in name or "_11/" in name: ## RotateZ
                self.targets[i] = 4
            elif "_12/" in name:   ## Release
                self.targets[i] = 5
            else:
                self.targets[i] = 0
                
        if is_train:
            list_1 = [name for name in self.lists if "_00/" in name]
            list_2 = [name for name in self.lists if "_01/" in name or "_02/" in name]
            list_3 = [name for name in self.lists if "_03/" in name or "_04/" in name or "_05/" in name \
                        or "_06/" in name or "_07/" in name or "_08/" in name]
            # list_4 = [name for name in self.lists if "_06/" in name or "_07/" in name or "_08/" in name]
            list_4 = [name for name in self.lists if "_09/" in name] #  or "_10/" in name or "_11/" in name
            list_5 = [name for name in self.lists if "_12/" in name]
            self.class_sample_counts = [len(lists_negative), len(list_1), len(list_2), 
                               len(list_3), len(list_4),  len(list_5)]
            print(self.class_sample_counts)
        
        self.args = args
        self.is_train = is_train
        self.noise_factor = 0.3
        self.rot_factor = 8 # Random rotation in the range [-rot_factor, rot_factor]
        self.scale_factor = 0.15 # rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]
        self.trans_factor = 10 # pixel

    def __len__(self):
        return len(self.lists)

    def augm_params(self, cls, ori_scale=1.0, ori_rotation=0.0, ori_image_center=(112, 112)):
        """Get augmentation parameters."""
        # Each channel is multiplied with a number 
        flip = -1 if np.random.rand() > 0.5 else 1            # flipping
        # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
        # pn = np.random.uniform(1-self.noise_factor, 1+self.noise_factor, 3)
    
        # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
        rot_factor = 90 if cls == 0 else self.rot_factor 
        rot = math.radians(min(2*rot_factor,
                max(-2*rot_factor, np.random.randn()*rot_factor))) + ori_rotation

        # The scale is multiplied with a number
        # in the area [1-scaleFactor,1+scaleFactor]
        sc = min(1+self.scale_factor,
                max(1-self.scale_factor, np.random.randn()*self.scale_factor+1))*ori_scale
        # but it is zero with probability 3/5
        img_center = (np.random.uniform(-1, 1, 2)*self.trans_factor + np.array(ori_image_center)).astype(np.int32).tolist()
        if np.random.uniform() <= 0.5:
            rot = ori_rotation

        return flip, rot, sc, img_center

    def get_image(self, idx): 
        cv2_im = cv2.imread(self.lists[idx])
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        return cv2_im

    def __getitem__(self, idx):

        img = self.get_image(idx)
        cls = self.targets[idx]
        flip, rot, sc, img_center = self.augm_params(cls)
        img, _ = rotate_image(img, rot, img_center, sc)  
        ### random downsampling and adding noise
        if flip == -1:
            img = flip_img(img)
        # Process image
        img = np.transpose(img.astype('float32'),(2,0,1))/255.0
        img = torch.from_numpy(img).float()
        # meta_data = {}
        # meta_data['cls'] = self.targets[idx]
        # meta_data['name'] = self.lists[idx]
        label = {"gesture": cls}
        return img, label

def _initialize_model(conf: DictConfig):
    set_random_state(conf.random_state)

    num_classes = len(conf.dataset.targets)
    conf.num_classes = {"gesture": num_classes, "leading_hand": 2}

    model = build_model(
        model_name=conf.model.name,
        num_classes=num_classes,
        checkpoint=conf.model.get("checkpoint", None),
        device=conf.device,
        pretrained=conf.model.pretrained,
        freezed=conf.model.freezed,
    )

    return model


def _run_test(path_to_config: str):
    """
    Run training pipeline

    Parameters
    ----------
    path_to_config : str
        Path to config
    """
    conf = OmegaConf.load(path_to_config)
    model = _initialize_model(conf)

    experimnt_pth = f"experiments/{conf.experiment_name}"
    writer = SummaryWriter(log_dir=f"{experimnt_pth}/logs")
    writer.add_text("model/name", conf.model.name)

    test_dataset = GestureDataset(is_train=False, conf=conf, transform=get_transform(), is_test=True)

    logging.info(f"Current device: {conf.device}")

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=conf.train_params.test_batch_size,
        num_workers=conf.train_params.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=conf.train_params.prefetch_factor,
    )

    TrainClassifier.eval(model, conf, 0, test_dataloader, writer, "test")


def _run_train(path_to_config: str) -> None:
    """
    Run training pipeline

    Parameters
    ----------
    path_to_config : str
        Path to config
    """

    conf = OmegaConf.load(path_to_config)
    model = _initialize_model(conf)

    train_dataset = HandGestureDataset(True)  # GestureDataset(is_train=True, conf=conf, transform=get_transform())
    test_dataset = HandGestureDataset(False) # GestureDataset(is_train=False, conf=conf, transform=get_transform())

    logging.info(f"Current device: {conf.device}")
    TrainClassifier.train(model, conf, train_dataset, test_dataset)


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Gesture classifier...")

    parser.add_argument(
        "-c", "--command", default="train", type=str, help="Training or test pipeline", choices=("train", "test")
    )

    parser.add_argument("-p", "--path_to_config", default="./classifier/config/default.yaml", type=str, help="Path to config")

    known_args, _ = parser.parse_known_args(params)
    return known_args


if __name__ == "__main__":
    args = parse_arguments()
    if args.command == "train":
        _run_train(args.path_to_config)
    elif args.command == "test":
        _run_test(args.path_to_config)
