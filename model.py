from models import networks
import torch
input_c = 3
output_c = 3
ngf = 64
netG = 'resnet_9blocks'
normG = 'instance'
no_dropout = True
init_type = 'xavier'
init_gain = 0.02
no_antialias = False
no_antialias_up = False
gpu_ids =[]
opt = None

model = networks.define_G(input_c, output_c, ngf, netG, normG, not no_dropout, init_type, init_gain, no_antialias, no_antialias_up, gpu_ids, opt)
model.load_state_dict(torch.load('17_net_G.pth'))