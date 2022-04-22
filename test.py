import numpy as np
import torch
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from steg_net import StegNet

def addToTensorboard(host, guest, H, W):

    host_tensor = transforms.ToTensor()(host)
    guest_tensor = transforms.ToTensor()(guest)
    guest_tensor = transforms.Grayscale(num_output_channels=1)(guest_tensor)

    host_resize = transforms.Resize((H, W))(host_tensor)
    guest_resize = transforms.Resize((H, W))(guest_tensor)

    writer.add_image("host/carrier", host_resize, 0)
    writer.add_image("guest/output", guest_resize, 0)

    host_resize = torch.reshape(host_resize, [1, 3, H, W])
    guest_resize = torch.reshape(guest_resize, [1, 1, H, W])

    carrier, output = model(host_resize, guest_resize)
    # print(output.shape)
    carrier = torch.reshape(carrier, [3, H, W])
    output = torch.reshape(output, [1, H, W])
    writer.add_image("host/carrier", carrier, 1)
    writer.add_image("guest/output", output, 1)

    writer.close()


configs = {
    'train_rate': 0.8,  # 训练数据占数据总量的比例
    'host_channels': 3,
    'guest_channels': 1,
    'img_width': 32,
    'img_height': 32,
    'epoch_num': 50,
    'train_batch_size': 64,
    'val_batch_size': 64,
    'encoder_weight': 1,
    'decoder_weight': 1,
    'model_path': '',
    'learning_rate': 1e-4
}




writer = SummaryWriter("logs")

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



host = Image.open("pics/01.jpg")
guest = Image.open("pics/QR1000.png")
addToTensorboard(host, guest, 300, 300)

# tensorboard --logdir=logs --port=10000




