import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from dataloaders.utils import decode_seg_map_sequence
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import matplotlib.pyplot as plt


import time

# Deeplabv3+
from modeling.deeplab import DeepLab
# Deeplabv3
from modeling.my_deeplab import My_DeepLab
# Unet
from modeling.test_net_my_attention import UNet
from modeling.patch_cnn import PatchNet

model_paths = {
    'person': {
        'deeplabv3+': "./bestmodels/deep_person/checkpoint.pth.tar",
        'deeplabv3': "./bestmodels/my_deep_person/checkpoint.pth.tar",
        'unet': "./bestmodels/unet_person/checkpoint.pth.tar"

    },
    'clothes': {
        'deeplabv3+': "./bestmodels/deep_clothes/checkpoint.pth.tar",
        'deeplabv3': "./bestmodels/my_deep_clothes/checkpoint.pth.tar",
        'unet': "./bestmodels/unet_clothes/checkpoint.pth.tar"
    },
    'electrode': {
        'deeplabv3+': "./bestmodels/deep_electrode/checkpoint.pth.tar",
        'deeplabv3': "./bestmodels/my_deep_electrode/checkpoint.pth.tar",
        'unet': "./bestmodels/unet_electrode/checkpoint.pth.tar",
        'patchnet': "./bestmodels/patchnet_electrode/checkpoint.pth.tar"
    }
}


def transform_val(sample):
    composed_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    return composed_transforms(sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSC420 Segmentation")

    parser.add_argument('--model', '-m', default='unet',
                        metavar='[deeplabv3+, deeplabv3, unet, patchnet]',
                        help="Specify Which Model"
                             "(default : DeepLabV3+)",
                        choices=['deeplabv3+', 'deeplabv3', 'unet', 'patchnet']
                        )

    parser.add_argument('--task', '-t', default='electrode', metavar='[person, fashion]',
                        help="Specify Task [person, fashion]",
                        choices=['person', 'clothes', 'electrode'], required=False
                        )

    ############  U-NET  test  ###########
    parser.add_argument('--path', '-p',
                        default='./run/ele/unet/experiment/checkpoint.pth.tar',
                        metavar='model_path', help="Specify Model Path")

    parser.add_argument('--input', '-i', default='../da/ele/predict_img_UNet/input/', metavar='input_path',
                        help='Input image ', required=False)

    parser.add_argument('--output', '-o', default='../data/ele/predict_img_UNet/output/', metavar='output_path',
                        help='Output image', required=False)




    args = parser.parse_args()

    path = args.path

    if args.task == 'person':
        dataset = "fashion_person"
        path = model_paths['person'][args.model]
        nclass = 2
    elif args.task == 'clothes':
        dataset = "fashion_clothes"
        path = model_paths['clothes'][args.model]
        nclass = 7
    elif args.task == 'electrode':
        dataset = "electrode"
        path = model_paths['electrode'][args.model]
        nclass = 1

    if (args.path):
        path = args.path

    print("Model Path is {}".format(path))

    if args.model == "deeplabv3+":
        # Suggested in paper, output stide is set to 8
        # to get better evaluation performance
        model = DeepLab(num_classes=nclass, output_stride=8)
    elif args.model == 'deeplabv3':

        model = My_DeepLab(num_classes=nclass, in_channels=3)
    elif args.model == 'unet':
        model = UNet(num_filters=32, num_categories=nclass, num_in_channels=3)
    elif args.model == 'patchnet':
        model = PatchNet(num_categories=nclass, num_in_channels=3)

    if torch.cuda.is_available():
        print("Moving model to GPU")
        model.cuda()
    else:
        print("CUDA not available, run model on CPU")
        model.cpu()
        torch.set_num_threads(8)

    if not os.path.isfile(path):
        raise RuntimeError("no model found at'{}'".format(path))

    # if not os.path.isfile(args.input):
    #     raise RuntimeError("no image found at'{}'".format(input))
    #
    # if os.path.exists(args.output):
    #     raise RuntimeError("Existed file or dir found at'{}'".format(args.output))

    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')

    print("Model loaded")
    model.load_state_dict(checkpoint['state_dict'])

    print("Image Loaded")
    for name in os.listdir(args.input):



        image = Image.open(args.in_path).convert('L')
        target = Image.open(args.input + "/" + name).convert('L')
        sample = {'image': image, 'label': target}
        image = transform_val(image)



        model.eval()
        t = time.time()



        if torch.cuda.is_available():
            image = image.cuda()
        with torch.no_grad():
            image = torch.unsqueeze(image, 0)
            #output = model(image)
            d1, d2, output = model(image)

        prediction = decode_seg_map_sequence(torch.max(output[:], 1)[1].detach().cpu().numpy(),
                                                 dataset=dataset)

        prediction = prediction.squeeze(0)
        #prediction = prediction.resize((3,256, 256), Image.ANTIALIAS)
        maskpath=args.output + "/" + "{}_mask.png".format(name[0:-4])
        save_image(prediction, maskpath, normalize=False)
        #save_image(prediction, args.output, normalize=False)

        #mask= Image.open(maskpath).resize((256, 256), Image.ANTIALIAS)
        #mask.save(maskpath)

        #print("Time spend {}s".format(time.time() - t))
        print("image:{} time: {} ".format(name, time.time() - t))

        "#################################################################################################################"
        prediction_d1 = decode_seg_map_sequence(torch.max(d1[:], 1)[1].detach().cpu().numpy(),
                                             dataset=dataset)
        prediction_d1 = prediction_d1.squeeze(0)
        maskpath_d1 = args.output + "/" + "{}_mask_d1.png".format(name[0:-4])
        save_image(prediction_d1, maskpath_d1, normalize=False)
        print("image:{} time: {} ".format(name, time.time() - t))

        "#################################################################################################################"
        prediction_d2 = decode_seg_map_sequence(torch.max(d2[:], 1)[1].detach().cpu().numpy(),
                                                dataset=dataset)
        prediction_d2 = prediction_d2.squeeze(0)
        maskpath_d2 = args.output + "/" + "{}_mask_d2.png".format(name[0:-4])
        save_image(prediction_d2, maskpath_d2, normalize=False)
        print("image:{} time: {} ".format(name, time.time() - t))







        "#################################################################################################################"# # # 可视化 输出
        "#####################                中间特征图32通道可视化                                  #################"# # # 可视化 输出
        # feature_unet = model.feature1
        # feature_msff = model.feature2
        # feature_mlpa = model.feature3
        # im_unet = np.squeeze(feature_msff.detach().cpu().numpy())
        # im_msff = np.squeeze(feature_unet.detach().cpu().numpy())
        # # o=nn.Sigmoid()
        # # output=o(output)
        # im_mlpa = np.squeeze(feature_mlpa.detach().cpu().numpy())
        #
        #
        # #im = decode_seg_map_sequence(torch.max(last_conv_layer[:], 1)[1].detach().cpu().numpy(), dataset=dataset)
        # #print(im.size())
        # #im = im.squeeze(0)
        # #print(im.size())
        # # [C, H, W] -> [H, W, C]
        # #im = np.transpose(im, [1, 2, 0])
        #
        #
        #
        #
        # # show top 12 feature maps
        # plt.figure()
        # for i in range(32):
        #     ax_unet = plt.subplot(4, 8, i + 1)
        #
        #     # [H, W, C]
        #     # 我们特征矩阵每一个 channel 所对应的是一个二维的特征矩阵，就像灰度图一样，channel = 1
        #     # 如果不指定 cmap='gray' 默认是以蓝色和绿色替换黑色和白色
        #     image1 = ax_unet.imshow(im_unet[i,:, :],cmap="jet")
        #
        #     # 去除坐标轴
        #     plt.xticks([])
        #     plt.yticks([])
        #     # 去除黑框
        #     ax_unet.spines['top'].set_visible(False)
        #     ax_unet.spines['right'].set_visible(False)
        #     ax_unet.spines['bottom'].set_visible(False)
        #     ax_unet.spines['left'].set_visible(False)
        # plt.savefig(args.output + "/" + "{}_unet.png".format(name[0:-4]), bbox_inches='tight')
        # plt.close()
        #
        #
        # plt.figure()
        # for i in range(32):
        #     ax_msff = plt.subplot(4, 8, i + 1)
        #     # [H, W, C]
        #     # 我们特征矩阵每一个 channel 所对应的是一个二维的特征矩阵，就像灰度图一样，channel = 1
        #     # 如果不指定 cmap='gray' 默认是以蓝色和绿色替换黑色和白色
        #     image2 = ax_msff.imshow(im_msff[i,:, :],cmap="jet")
        #
        #     # 去除坐标轴
        #     plt.xticks([])
        #     plt.yticks([])
        #     # 去除黑框
        #     ax_msff.spines['top'].set_visible(False)
        #     ax_msff.spines['right'].set_visible(False)
        #     ax_msff.spines['bottom'].set_visible(False)
        #     ax_msff.spines['left'].set_visible(False)
        # plt.savefig(args.output + "/" + "{}_msff.png".format(name[0:-4]), bbox_inches='tight')
        # plt.close()
        #
        # plt.figure()
        # for i in range(32):
        #     ax_mlpa = plt.subplot(4, 8, i + 1)
        #     # [H, W, C]
        #     # 我们特征矩阵每一个 channel 所对应的是一个二维的特征矩阵，就像灰度图一样，channel = 1
        #     # 如果不指定 cmap='gray' 默认是以蓝色和绿色替换黑色和白色
        #     image3 = ax_mlpa.imshow(im_mlpa[i,:, :],cmap="jet")
        #
        #     # 去除坐标轴
        #     plt.xticks([])
        #     plt.yticks([])
        #     # 去除黑框
        #     ax_mlpa.spines['top'].set_visible(False)
        #     ax_mlpa.spines['right'].set_visible(False)
        #     ax_mlpa.spines['bottom'].set_visible(False)
        #     ax_mlpa.spines['left'].set_visible(False)
        # plt.savefig(args.output + "/" + "{}_mlpa.png".format(name[0:-4]), bbox_inches='tight')
        # plt.close()
        "#################################################################################################################"  # # # 可视化 输出















