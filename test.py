import time

import numpy as np
import torch
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from psnr import psnr
from steg_net import StegNet
from change_size import scale_down


# 展示模型效果
def addToTensorboard(host, guest):
    guest = guest.convert("1")
    # 图像原始的大小
    host_H = host.size[1]
    host_W = host.size[0]
    guest_H = guest.size[0]

    host_tensor = transforms.ToTensor()(host)
    guest_tensor = transforms.ToTensor()(guest)
    # print(guest_tensor.shape)

    # 秘密图像取灰度图
    # if (guest_tensor.shape[0] != 1):
    #     guest_tensor = transforms.Grayscale(num_output_channels=1)(guest_tensor)


    # host_resize = transforms.Resize((H, W))(host_tensor)
    guest_resize = transforms.Resize((host_H, host_W))(guest_tensor)


    writer.add_image("host/carrier" + f"(H{host_H}xW{host_W})", host_tensor, 0)
    writer.add_image("guest/output" + f"(H{host_H}xW{host_W})", guest_tensor, 0)

    host_resize = torch.reshape(host_tensor, [1, 3, host_H, host_W])
    guest_resize = torch.reshape(guest_resize, [1, 1, host_H, host_W])

    carrier, output = model(host_resize, guest_resize)
    carrier = torch.reshape(carrier, [3, host_H, host_W])
    output = torch.squeeze(transforms.Resize((guest_H, guest_H))(output), dim=0)

    writer.add_image("host/carrier" + f"(H{host_H}xW{host_W})", carrier, 1)
    writer.add_image("guest/output" + f"(H{host_H}xW{host_W})", output, 1)

    # host = torch.squeeze(host_resize, 0).view(-1, W)
    # guest = torch.squeeze(guest_resize, 0).view(-1, W)
    # carrier = torch.squeeze(carrier, 0).view(-1, W)
    # output = torch.squeeze(output, 0).view(-1, W)
    # print(host.shape)
    # print(guest.shape)
    # print(carrier.shape)
    # print(output.shape)
    # psnr_host = psnr(host, carrier)
    # psnr_guest = psnr(guest, output)
    # writer.add_scalar("psnr_host", psnr_host)
    # writer.add_scalar("psnr_guest", psnr_guest)


configs = {
    'train_rate': 0.8,  # 训练数据占数据总量的比例
    'host_channels': 3,
    'guest_channels': 1,
    'img_width': 32,
    'img_height': 32,
    'epoch_num': 50,
    'train_batch_size': 32,
    'val_batch_size': 64,
    'encoder_weight': 1.2,
    'decoder_weight': 1,
    'model_path': 'modules',
    'learning_rate': 1e-4
}


writer = SummaryWriter("logs5")

model = StegNet()
model.load_model(configs['model_path'], file_name=f"steg_net"
                                                  # f"_datasetcifar10"
                                                  f"_host{configs['host_channels']}"
                                                  f"_guest{configs['guest_channels']}"
                                                  f"_epochNum{configs['epoch_num']}"
                                                  f"_batchSize{configs['train_batch_size']}"
                                                  f"_lr{configs['learning_rate']}"
                                                  f"_encoderWeight{configs['encoder_weight']}"
                                                  f"_decoderWeight{configs['decoder_weight']}"
                                                  # f"_imgSize{configs['img_width']}x{configs['img_height']}"
                                                  f".pth")


host_src = "pics/DSC07730.jpg"
guest_src = "pics/QR1000.png"
cache_src = "cache/temp.jpg"
host = Image.open(host_src)
guest = Image.open(guest_src)


# 判断图片是否过大
if (max(host.size[0], host.size[1]) > 3000):
    # 压缩图片
    scale_down(host_src, cache_src, 250)
    time.sleep(5)
    host = Image.open(cache_src)

# 转换四通道图为三通道图
if (transforms.ToTensor()(host).shape[0] == 4):
    r, g, b, a = host.split()
    host = Image.merge("RGB", (r, g, b))
if (transforms.ToTensor()(guest).shape[0] == 4):
    r, g, b, a = guest.split()
    guest = Image.merge("RGB", (r, g, b))
    # guest.save("cache/t.png")

# 变为单通道黑白图
guest = guest.convert("1")
# guest.save("cache/bw.jpg")

addToTensorboard(host, guest)

writer.close()
# tensorboard --logdir=logs --port=10000
