import os
import cv2
import numpy as np
import torch
import torchvision
import argparse

import matplotlib.pyplot as plt

import pandas as pd

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from model import load_optimizer, save_model
from modules import SimCLR, NT_Xent, get_resnet
from modules.transformations import TransformsSimCLR
from modules.sync_batchnorm import convert_model
from prep import process_image_file
from utils import yaml_config_hook
from torch.utils.data import Dataset, DataLoader

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):          

        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()
        
        if prob < 0.5:            

            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class TransformsSimCLR_covid:
    """
    A stochastic data augmentation module that transforms any given data example randomly 
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        degrees = [-5, 5]
        self.train_transform = torchvision.transforms.Compose(
            [   torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(), 
                torchvision.transforms.RandomRotation(degrees, 
                                                    resample=False,
                                                    expand=False, 
                                                    center=None),
                torchvision.transforms.ColorJitter(brightness=0.2),
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.001 * size)),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [   torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((size,size)),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


class covidDataset(DataLoader):
    
    def __init__(self, transform, size, datatype, purpose):

        category_dict = {'normal' : 0, 'pneumonia' : 1, 'COVID-19' : 2}
        data_path = '/home/sdsra/Documents/COVID-Net/data' 

        if datatype == 'train':
            tmppath = os.path.join(data_path, 'train_split_org.txt')
            self.imgpath = os.path.join(data_path, 'train')

        else:            
            tmppath = os.path.join(data_path, 'test_split.txt')   
            self.imgpath = os.path.join(data_path, 'test')

        self.data_info = pd.read_csv(tmppath, header=None, delim_whitespace=True)

        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        self.labels = np.asarray(self.data_info.iloc[:, 2])
        self.labels = [int(category_dict[name]) for name in self.labels]

        self.purpose = purpose
        self.transform = transform(size)
        self.datatype = datatype

        
    def __len__(self):
        return len(self.data_info.index)
    
    def __getitem__(self, index):
        
        single_image_name = self.image_arr[index]
        single_image_label = self.labels[index]
        single_img = cv2.imread(os.path.join(self.imgpath, single_image_name ))
        single_img = process_image_file(single_img, top_percent= 0.15)

        if self.purpose=='train':
            single_img = self.transform(single_img)
        else:
            single_img = self.transform.test_transform(single_img)
    
        return single_img, single_image_label


def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    step = 0
    #for step, ((x_i, x_j), _) in enumerate(train_loader):
    for data in train_loader:

        optimizer.zero_grad()

        x_i = data[0][0].cuda(non_blocking=True)
        x_j = data[0][1].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        # f1  f2 are zi & zj
        features = torch.cat([z_i.unsqueeze(1), z_j.unsqueeze(1)], dim=1)
        loss = criterion(features, labels)

        # loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
        step = step + 1
    return loss_epoch


def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    
    else:
        # covid        
        train_dataset = covidDataset(transform=TransformsSimCLR_covid, 
                                     size=args.image_size,
                                     datatype='train',
                                     purpose='train')

    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=(train_sampler is None),
                                                drop_last=True,
                                                num_workers=args.workers,
                                                sampler=train_sampler, )
    # initialize ResNet
    # encoder = get_resnet(args.resnet, pretrained=False)
    encoder = get_resnet(args.resnet, pretrained=args.pretrain)

    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    model = SimCLR(args, encoder, n_features)
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
        print('....... loading from ', model_fp)
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    # criterion = NT_Xent(args.batch_size, args.temperature, args.device, args.world_size)
    criterion = NT_Xent( args.temperature) #, contrast_mode='all', base_temperature=0.07)

    # DDP / DP
    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
    else:
        if args.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()

    args.global_step = 0
    args.current_epoch = 0
    res_epoch = []

    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 10 == 0:
            save_model(args, model, optimizer)

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            args.current_epoch += 1
        res_epoch.append(loss_epoch / len(train_loader))

        if epoch % 10 == 0:
            plt.plot(res_epoch, 'ro')
            name_save = 'fig_' + str(epoch)+ '.png'
            plt.savefig(name_save)
            plt.close()



    ## end training
    save_model(args, model, optimizer)
    import pdb; pdb.set_trace()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    if args.nodes > 1:
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)


# python -m testing.logistic_regression --dataset=STL10 --model_path=. --epoch_num=100 --resnet resnet18