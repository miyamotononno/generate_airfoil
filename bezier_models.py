import torch
import torch.nn as nn
import torch.nn.functional as F


n_points = 64
bezier_degree=31
depth_cpw = 32*8
EPSILON = 1e-7
coords_size = (248, 2) 


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Generator(nn.Module):
    def __init__(self, latent_dim):
        
        super(Generator, self).__init__()
        self.dim_cpw = (bezier_degree+1)//8
        self.kernel_size = (4,3)
        padding = (1, 1)
        # 1. fully_connected
        self.linear1 = nn.Linear(latent_dim+1, 1024)
        self.batch1 = nn.BatchNorm1d(1024, 0.8)

        # 2.1. deconvolutional layers
        self.linear2_1_1 = nn.Linear(1024, self.dim_cpw*3*depth_cpw)
        self.batch2_1_1 = nn.BatchNorm1d(self.dim_cpw*3*depth_cpw, 0.8)
        
        self.conv_tra_2_1_2 = nn.ConvTranspose2d(depth_cpw,
                                                 depth_cpw//2,
                                                 kernel_size=self.kernel_size,
                                                 stride=(2,1),
                                                 padding=padding)
        self.batch2_1_2 = nn.BatchNorm2d(depth_cpw//2, 0.8)
        self.conv_tra_2_1_3 = nn.ConvTranspose2d(depth_cpw//2,
                                                 depth_cpw//4,
                                                 kernel_size=self.kernel_size,
                                                 stride=(2,1),
                                                 padding=padding)
        self.batch2_1_3 = nn.BatchNorm2d(depth_cpw//4, 0.8)
        self.conv_tra_2_1_4 = nn.ConvTranspose2d(depth_cpw//4,
                                                 depth_cpw//8,
                                                 kernel_size=self.kernel_size,
                                                 stride=(2,1),
                                                 padding=padding)
        self.batch2_1_4 = nn.BatchNorm2d(depth_cpw//8, 0.8)
        
        # Control points
        self.conv2_1_5_1 = nn.Conv2d(depth_cpw//8, 1, self.kernel_size, stride=(1,2)) # padding='valid'あとで実装しないといけない?
        self.tanh2_1 = nn.Tanh()
            
        # Weights
        self.conv2_1_5_2 = nn.Conv2d(depth_cpw//8, 1, self.kernel_size, stride=(1,3)) # padding='valid'あとで実装しないといけない?
        self.sigmoid2_1 = nn.Sigmoid()

        # 2.2 fully_connected & softmax
        self.softmax2_2 = nn.Softmax()
        self.linear2_2_1 = nn.Linear(1024, 256)
        self.batch2_2_1 = nn.BatchNorm1d(256, 0.8)
        self.linear2_2_2 = nn.Linear(256, coords_size[0]-1)
        self.pad2_2 = nn.ZeroPad2d((1,0,0,0))

    # 3 Bazier Layer
    def bazier_layer(self, cp, w, ub):
        num_control_points = bezier_degree + 1
        lbs = ub.repeat(1,1,num_control_points) # batch_size x n_data_points x n_control_points
        pw1 = torch.range(start=0, end=bezier_degree, dtype=torch.float32)
        pw1 = torch.reshape(pw1, (1,1,-1)) # 1 x 1 x n_control_points
        pw2 = torch.fliplr(pw1)
        lbs = torch.add(torch.multiply(pw1, torch.log(lbs+EPSILON)), torch.multiply(pw2, torch.log(1-lbs+EPSILON))) # batch_size x n_data_points x n_control_points
        lc = torch.add(torch.lgamma(pw1+1), torch.lgamma(pw2+1))
        num_control_points = torch.FloatTensor(num_control_points)
        lc = torch.subtract(torch.lgamma(num_control_points), lc) # 1 x 1 x n_control_points
        lbs = torch.add(lbs, lc) # batch_size x n_data_points x n_control_points
        bs = torch.exp(lbs)

        # Compute data points
        cp_w = torch.multiply(cp, w)
        dp = torch.matmul(bs, cp_w) # batch_size x n_data_points x 2
        bs_w = torch.matmul(bs, w) # batch_size x n_data_points x 1
        dp = torch.div(dp, bs_w) # batch_size x n_data_points x 2
        dp = torch.unsqueeze(dp, -1) # batch_size x n_data_points x 2 x 1
        return dp

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, noise, labels):
        # 1. fully_connected
        gen_input = torch.cat((labels, noise), -1)
        x = F.leaky_relu(self.batch1(self.linear1(gen_input)), 0.2)
        
        # 2.1. deconvolutional layers
        cpw = self.batch2_1_1(self.linear2_1_1(x))
        cpw = torch.reshape(cpw, (-1, depth_cpw, self.dim_cpw, 3)) # batch, in_channel, height, width
        cpw = F.leaky_relu(self.batch2_1_2(self.conv_tra_2_1_2(cpw)), 0.2)
        cpw = F.leaky_relu(self.batch2_1_3(self.conv_tra_2_1_3(cpw)), 0.2)
        cpw = F.leaky_relu(self.batch2_1_4(self.conv_tra_2_1_4(cpw)), 0.2)
        # Control points
        cp = self.tanh2_1(self.conv2_1_5_1(cpw)) # batch_size x (bezier_degree+1) x 2 x 1
        cp = torch.squeeze(cp) # batch_size x (bezier_degree+1) x 2
        # Weights
        w = self.sigmoid2_1(self.conv2_1_5_2(cpw))
        w = torch.squeeze(w)

        # 2.2. softmax
        db = self.softmax2_2(self.linear2_2_2(self.batch2_2_1(self.linear2_2_1(x))))
        ub = torch.cumsum(self.pad2_2(db), dim=1)
        ub = torch.unsqueeze(torch.minimum(ub, torch.Tensor([1.0])), -1) # 1 x n_data_points x 1
        
        # 3. bezier layer
        dp = self.bazier_layer(cp,w, ub)

        return dp, cp, w, ub, db

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.depth = 64
        self.kernel_size = (4,2)

        def block(in_feat, out_feat, training=True): # 入力が192でなく248なので変更の必要あり
            layers = [nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         self.kernel_size,
                                         stride=(2,1),
                                         padding=[self.kernel_size[1]/2, self.kernel_size[0]/2]
                                         )]
            layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if training:
              layers.append(nn.Dropout(0.4))
            return layers

        self.model = nn.Sequential(
          *block(1+496, self.depth*1),
          *block(self.depth*1, self.depth*2),
          *block(self.depth*2, self.depth*4),
          *block(self.depth*4, self.depth*8),
          *block(self.depth*8, self.depth*16),
          *block(self.depth*16, self.depth*32),
          nn.Flatten(),
          nn.Linear(self.depth*32, self.depth*32),
          nn.BatchNorm2d(self.depth*32, 0.8),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(self.depth*16, 1)
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def forward(self, coords, labels):
        d_in = torch.cat((coords.view(coords.size(0), -1), labels), -1)
        validity = self.model(d_in)
        return validity
