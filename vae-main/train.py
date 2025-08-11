
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import argparse
import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
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
    parser.add_argument('--batch-size', type=int, default=10, help='設定批次大小')
    parser.add_argument('--epochs', type=int, default=1000, help='設定訓練步數')
    parser.add_argument('--source', type=str, default='dataset', help='訓練用圖片資料夾') 
    parser.add_argument('--img-size', nargs=2, type=int, default=[28, 28], help='圖片大小 手寫資料集為28x28')
    parser.add_argument('--lr', type=float, default=1e-4, help='設定學習率')
    parser.add_argument('--save_period', type=int, default=20, help='設定儲存間隔')
    opt = parser.parse_args()
    print(opt)
    batch_size = opt.batch_size
    dataset_path = opt.source 
    width, height = opt.img_size
    x_dim =  3*width*height 
    lr = opt.lr
    epochs = opt.epochs
    save_period=opt.save_period
    img_transform = transforms.Compose([
        transforms.Resize((width, height)),  
        #transforms.CenterCrop(28),
        transforms.ToTensor(), 
    ])
    train_dataset = ImageFolder(root=dataset_path, transform=img_transform)
    device=opt.device
    if device=='cpu':
        DEVICE = torch.device("cpu")
    else:
        cuda = True
        DEVICE = torch.device("cuda" if cuda else "cpu")
    hidden_dim = 400
    latent_dim = 200
    kwargs = {'num_workers': 1, 'pin_memory': True} 

    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = DataLoader(dataset=train_dataset,  batch_size=batch_size, shuffle=False, **kwargs)
    print('load')

    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)



    from torch.optim import Adam

    BCE_loss = nn.BCELoss()

    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD


    optimizer = Adam(model.parameters(), lr=lr)
    save_path = './models/'
    os.makedirs(save_path, exist_ok=True)


    print("Start training VAE...")
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(DEVICE)

            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} complete! Average Loss: {overall_loss / len(train_loader.dataset)}")
        if epoch %save_period==0:
            torch.save(model.state_dict(), f'{save_path}VAE_model_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), f'{save_path}VAE_model_final.pth')




    print("Train Finish!!")



    import matplotlib.pyplot as plt
    from tqdm import tqdm
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            #x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)
            
            x_hat, _, _ = model(x)


            break

    def show_image(x, idx):
        x = x.view(batch_size, 3, width, height)
        x = x.permute(0, 2, 3, 1)

        fig = plt.figure()
        plt.imshow(x[idx].cpu().numpy())
    #show_image(x, idx=0)
    #show_image(x_hat, idx=0)
    print(x.shape)

    save_image(x.view(batch_size, 3, width, height), 'original_img.png')
    save_image(x_hat.view(batch_size, 3, width, height), 'generated_img.png')


    '''for idx in range(batch_size):
        save_image(x_hat[idx], f'generated_sample_{idx}.png')'''





