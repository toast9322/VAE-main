import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import argparse
import torch
import torch.nn as nn

import numpy as np
from torchvision.utils import save_image, make_grid

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))
        x_hat = torch.sigmoid(self.FC_output(h))
        
        x_hat = x_hat.view(x.size(0), 3, 28, 28)
        return x_hat

        
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        
        z = mean + var*epsilon            
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) 
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', help='cuda device, 0 or cpu')
    parser.add_argument('--weights', type=str, default='models/VAE_model_final.pth', help='填入模型 副檔名為.pth')
    parser.add_argument('--batch-size', type=int, default=10, help='設定批次大小')
    parser.add_argument('--source', type=str, default='test', help='測試用圖片') 
    parser.add_argument('--img-size', nargs=2, type=int, default=[28, 28], help='圖片大小 手寫資料集為28x28')
    parser.add_argument('--rounds', type=int, default=1, help='生成圖片輪數')
    opt = parser.parse_args()
    print(opt)
    dataset_path = opt.source 
    width, height = opt.img_size
    weights=opt.weights
    device=opt.device

    img_transform = transforms.Compose([
        transforms.Resize((width, height)),
        #transforms.CenterCrop(28),
        transforms.ToTensor(),
    ])
    train_dataset = ImageFolder(root=dataset_path, transform=img_transform)

    if device=='cpu':
        DEVICE = torch.device("cpu")
    else:
        cuda = True
        DEVICE = torch.device("cuda" if cuda else "cpu")


    batch_size = opt.batch_size
    x_dim = 3*width*height 
    hidden_dim = 400
    latent_dim = 200
    kwargs = {'num_workers': 1, 'pin_memory': True} 
    test_loader  = DataLoader(dataset=train_dataset,  batch_size=batch_size, shuffle=False, **kwargs)
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)
    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
    model.load_state_dict(torch.load(weights))
    model.eval()
    print('已載入模型')
    count=0
    rounds=opt.rounds
    print('預計生成',rounds,'輪圖片')
    while True:

            with torch.no_grad():
                for batch_idx, (x, _) in enumerate(test_loader):
                    x = x.to(DEVICE)
                    
                    x_hat, _, _ = model(x)


                    break

            def show_image(x, idx):
                x = x.view(batch_size, 3, width, height)
                x = x.permute(0, 2, 3, 1)

                fig = plt.figure()
                plt.imshow(x[idx].cpu().numpy())
            #print(x.shape)
            save_image(x.view(batch_size, 3, width, height), 'output/original.png')
            save_image(x_hat.view(batch_size, 3, width, height), 'output/generated_image.png')

            
            count+=1
            print('正在生成第',count,'輪圖片')
            for idx in range(batch_size):
                save_image(x_hat[idx], f'output/generate_{count}_{idx}.jpg')
                print('已儲存圖片',idx)
                
            if count>=rounds:
                break
    print('已儲存所有圖片至 ./output/')