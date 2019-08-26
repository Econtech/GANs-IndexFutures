import numpy as np
from pandas import DataFrame
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data_loader import load_data
from laplotter import LossAccPlotter
from visualization import plot

class Generator(nn.Module):
    def __init__(self, latent_dim, ngf=32):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, ngf*8, 3, 1, 1, bias=False),
            nn.BatchNorm1d(ngf*8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4
            nn.ConvTranspose1d(ngf*8, ngf*4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(ngf*4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4
            nn.ConvTranspose1d(ngf*4, ngf*2, 3, 1, 1, bias=False),
            nn.BatchNorm1d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 4
            nn.ConvTranspose1d(ngf*2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 4
            nn.ConvTranspose1d(ngf, 239, 4, 1, 0, bias=False),
            nn.Tanh()
            # state size. 240 x 4
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, ndf=32):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # input is 240 * 4
            nn.Conv1d(239, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf) x 4
            nn.Conv1d(ndf, ndf*2, 3, 1, 1, bias=False),
            nn.BatchNorm1d(ndf*2),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*2) x 4
            nn.Conv1d(ndf*2, ndf*4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(ndf*4),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*4) x 4
            nn.Conv1d(ndf*4, ndf*8, 3, 1, 1, bias=False),
            nn.BatchNorm1d(ndf*8),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*8) x 4
            nn.Conv1d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def visualize(instrument, tensor_data, epoch, num_batch):
    for i in range(tensor_data.shape[0]):
        dimage = tensor_data.data[i].numpy()
        df = DataFrame(dimage, columns=['high','low','open','close'])
        plot(df, epoch, num_batch, i, instrument=instrument, generate=True, v=False)

def train(instrument, batch_size, latent_dim, epochs, mode=None):
    data_loader = load_data(instrument, batch_size)

    generator = Generator(latent_dim)
    generator.apply(weights_init)
    discriminator = Discriminator()
    discriminator.apply(weights_init)

    # print("Generator's state_dict:")
    # for param_tensor in generator.state_dict():
    #     print(param_tensor, "\t", generator.state_dict()[param_tensor].size())

    # print("Discriminator's state_dict:")
    # for param_tensor in discriminator.state_dict():
    #     print(param_tensor, "\t", discriminator.state_dict()[param_tensor].size())

    g_optimizer = optim.Adam(generator.parameters(), lr=0.002, betas=(0.5,0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.002, betas=(0.5,0.999))

    loss = nn.BCELoss()

    if not os.path.exists("./plots/"+instrument):
        os.makedirs("./plots/"+instrument)

    plotter = LossAccPlotter(
                        save_to_filepath="./loss/"+instrument+"/loss.png",
                        show_regressions=False,
                        show_acc_plot=False,
                        show_averages=False,
                        show_plot_window=True,
                        x_label="Epoch")

    if not os.path.exists("./model/"+instrument):
        os.makedirs("./model/"+instrument)
    if not os.path.exists("./loss/"+instrument):
        os.makedirs("./loss/"+instrument)
        
    epoch = 0
    d_loss_list = []
    g_loss_list = []
    while True:
        for num_batch, real_data in enumerate(data_loader):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            for i in range(1):
                real_data_d = next(iter(data_loader))
                size = real_data_d.size(0)
                real_data_t = torch.ones(size,239,4)
                for i in range(size):
                    for j in range(239):
                        real_data_t[i][j] = torch.log(real_data_d[i,j+1,:]/real_data_d[i,j,3])

                y_real = Variable(torch.ones(size, 1, 1))
                y_fake = Variable(torch.zeros(size, 1, 1))
                real_data = Variable(real_data_t.float())
                fake_data = Variable(torch.from_numpy(
                    np.random.normal(0,0.2,(size, latent_dim, 1))).float()
                )
                fake_gen = generator(fake_data).detach()

                prediction_real = discriminator(real_data)
                loss_real = loss(prediction_real, y_real)
                prediction_fake = discriminator(fake_gen)
                loss_fake = loss(prediction_fake, y_fake)
                d_loss = loss_real + loss_fake

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            size = real_data.size(0)
            y_real = Variable(torch.ones(size, 1, 1))
            fake_data = Variable(torch.from_numpy(
                np.random.normal(0,0.2,(size, latent_dim, 1))).float()
            )            
            fake_gen = generator(fake_data)
            prediction = discriminator(fake_gen)
            g_loss = loss(prediction, y_real)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if num_batch % 20 == 0:
                tmp = torch.ones(size,239,4)
                data = real_data if num_batch == 0 else fake_gen
                for batch in range(size):
                    for t in range(238,0,-1):
                        data[batch,t,0]=torch.sum(data[batch,0:t,3])+data[batch,t,0]
                        data[batch,t,1]=torch.sum(data[batch,0:t,3])+data[batch,t,1]
                        data[batch,t,2]=torch.sum(data[batch,0:t,3])+data[batch,t,2]
                        data[batch,t,3]=torch.sum(data[batch,0:t+1,3])
                data=torch.exp(data)
                tmp=tmp*data
                print("epoch: %d, num_batch: %d, d-loss: %.4f, g-loss: %.4f" 
                            % (epoch, num_batch, d_loss.data.numpy(), g_loss.data.numpy()))
                visualize(instrument, tmp, epoch, num_batch)

            plotter.add_values(epoch,
                loss_train=g_loss.item(),
                loss_val=d_loss.item())
            d_loss_list.append(d_loss.item())
            g_loss_list.append(g_loss.item())

        if epoch % 10 == 0:
            torch.save(generator, "./model/"+instrument+"/generator_epoch_"+str(epoch)+".model")
            torch.save(discriminator, "./model/"+instrument+"/discriminator_epoch_"+str(epoch)+".model")

            d_loss_np = np.array(d_loss_list)
            np.save("./loss/"+instrument+"/d_loss_epoch_"+str(epoch)+".npy", d_loss_np)
            g_loss_np = np.array(g_loss_list)
            np.save("./loss/"+instrument+"/g_loss_epoch_"+str(epoch)+".npy", g_loss_np)
        
        epoch += 1
        if mode == "test" and epoch == epochs:
            break
        
    

if __name__ == "__main__":
    instrument = 'IF'
    batch_size = 64
    epochs = 10
    latent_dim = 100
    mode = "train"
    train(instrument, batch_size, latent_dim, epochs, mode=mode)