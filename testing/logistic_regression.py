import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import cv2
import pandas as pd 

from utils import yaml_config_hook

from modules import SimCLR, LogisticRegression, get_resnet
from modules.transformations import TransformsSimCLR

from main import TransformsSimCLR_covid, process_image_file
from sklearn.metrics import confusion_matrix

def mkprc(data_info, labels, prc):
    
    ids_normal = [s for s in range(len(labels)) if labels[s]==0]
    ids_pneumonia = [s for s in range(len(labels)) if labels[s]==1]
    ids_covid = [s for s in range(len(labels)) if labels[s]==2]
    
    num_normal = int (prc * len(ids_normal))
    ids_normal_prc = np.random.choice(ids_normal, size=num_normal, replace=False)
    
    num_pneumonia = int (prc * len(ids_pneumonia))
    ids_pneumonia_prc = np.random.choice(ids_pneumonia, size=num_pneumonia, replace=False)
    
    num_covid = int (prc * len(ids_covid))
    ids_covid_prc = np.random.choice(ids_covid, size=num_covid, replace=False)
    
    total_ids = np.hstack((ids_normal_prc, ids_pneumonia_prc))
    total_ids = np.hstack((total_ids, ids_covid_prc))

    tmp = (ids_normal_prc.shape[0]+ ids_pneumonia_prc.shape[0]+ ids_covid_prc.shape[0])
    if total_ids.shape[0] != tmp:
        import pdb; pdb.set_trace()
        
    df_prc = data_info.loc[total_ids,:]    
    print(prc * data_info.shape[0], df_prc.shape[0])
    labels_prc = [labels[s] for s in total_ids]

    if(df_prc.shape[0]!= len(labels_prc)):
        import pdb; pdb.set_trace()
    
    return df_prc, labels_prc


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

        self.labels = np.asarray(self.data_info.iloc[:, 2])
        self.labels = [int(category_dict[name]) for name in self.labels]
        
        # if we want to use % train data 
        if datatype == 'train':
            self.data_info, self.labels = mkprc(self.data_info, self.labels, prc=1.0)
       
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
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


def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    step = 0 

    # for step, (x, y) in enumerate(loader):
    for data in loader:
        
        # import pdb; pdb.set_trace()

        x = data[0].to(device)
        y = data[1].to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.cpu().detach().numpy())

        # if step % 20 == 0:
        #     print(f"Step [{step}/{len(loader)}]\t Computing features...")
        # step = step + 1

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    step = 0
    #for step, (x, y) in enumerate(loader):
    for data in loader:
        optimizer.zero_grad()

        x = data[0].to(args.device)
        y = data[1].to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        step = step + 1
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch


def test(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    # for step, (x, y) in enumerate(loader):
    ytotal    = []
    predtotal = []

    for data in loader:
    
        model.zero_grad()

        x = data[0].to(args.device)
        y = data[1].to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

        ytotal.extend(y)
        predtotal.extend(predicted)
    
    ytotal    = [s.cpu() for s in ytotal]
    predtotal =[s.cpu() for s in predtotal]
    cmf = confusion_matrix(ytotal, predtotal)

    return cmf, loss_epoch, accuracy_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="train",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="test",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    else:
        # covid       
        train_dataset = covidDataset(transform=TransformsSimCLR_covid, 
                                    size=args.image_size,
                                    datatype='train',
                                    purpose='test')
        
        test_dataset = covidDataset(transform=TransformsSimCLR_covid, 
                                    size=args.image_size,
                                    datatype='test',
                                    purpose='test')

    # train sampler
    targets = train_dataset.labels
    class_count = np.unique(targets, return_counts=True)[1]
    print(class_count)

    weight = 1. / class_count
    samples_weight = weight[targets]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    weighted=True

    if weighted==True:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.logistic_batch_size,
                                                    shuffle=False,
                                                    drop_last=True,
                                                    num_workers=args.workers,
                                                    sampler=sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.logistic_batch_size,
                                                    shuffle=True,
                                                    drop_last=True,
                                                    num_workers=args.workers,)
                                                
    
    # test sampler
    # targets = test_dataset.labels
    # class_count = np.unique(targets, return_counts=True)[1]
    # print(class_count)

    # weight = 1. / class_count
    # samples_weight = weight[targets]
    # samples_weight = torch.from_numpy(samples_weight)
    # sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.logistic_batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=args.workers,)
                                                #sampler=sampler)

    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(args, encoder, n_features)
    simclr_model.eval()
    model_fp = os.path.join( args.model_path, "checkpoint_{}.tar".format(args.epoch_num))

    simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    simclr_model = simclr_model.to(args.device)
    
    print('.....loading', model_fp)
    

    ## Logistic Regression
    n_classes = 3  # CIFAR-10 / STL-10
    model = LogisticRegression(simclr_model.n_features, n_classes)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y) = get_features( simclr_model, train_loader, test_loader, args.device)

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape, np.unique(train_y), np.unique(test_y))
    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays( train_X, train_y, test_X, test_y, args.logistic_batch_size)

    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = train( args, arr_train_loader, simclr_model, model, criterion, optimizer)
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t Accuracy: {accuracy_epoch / len(train_loader)}"
        )

    # final testing
    cmf, loss_epoch, accuracy_epoch = test( args, arr_test_loader, simclr_model, model, criterion, optimizer)
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t Accuracy: {accuracy_epoch / len(test_loader)}"
    )

    import pdb; pdb.set_trace()
