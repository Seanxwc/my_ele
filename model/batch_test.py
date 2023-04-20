import torch
import argparse
import os
from dataloaders.utils import decode_seg_map_sequence
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import time
# Unet
from modeling.unet import UNet
from modeling.segnet import SegNet




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
        'unet': "./bestmodels/unet_electrode/checkpoint.pth.tar",
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
                        metavar='[unet]',
                        help="Specify Which Model"
                             "(default : DeepLabV3+)",
                        choices=[ 'unet', 'segnet']
                        )

    parser.add_argument('--task', '-t', default='electrode', metavar='[person, fashion]',
                        help="Specify Task [person, fashion]",
                        choices=['person', 'clothes', 'electrode'], required=False
                        )
    ############  DeepLabV3+ electrode test  ###########
    # parser.add_argument('--path', '-p',
    #                     default='./run/electrode/electrode-deeplabv3p-resnet-ft-all/experiment_10/checkpoint.pth.tar',
    #                     metavar='model_path', help="Specify Model Path")
    #
    # parser.add_argument('--input', '-i', default='../data/electrode/predict_img_UNet/input/', metavar='input_path',
    #                     help='Input image ', required=False)
    #
    # parser.add_argument('--output', '-o', default='../data/electrode/predict_img_UNet/output/', metavar='output_path',
    #                     help='Output image', required=False)

    ############  U-NET electrode test  ###########
    parser.add_argument('--path', '-p',
                        default='./run/electrode/electrode-unet/experiment_6/checkpoint.pth.tar',
                        metavar='model_path', help="Specify Model Path")

    parser.add_argument('--input', '-i', default='../data/electrode/predict_img_UNet/input/', metavar='input_path',
                        help='Input image ', required=False)

    parser.add_argument('--output', '-o', default='../data/electrode/predict_img_UNet/output/', metavar='output_path',
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
        nclass = 3

    if (args.path):
        path = args.path

    print("Model Path is {}".format(path))



    if args.model == 'unet':
        model = UNet(num_filters=32, num_categories=nclass, num_in_channels=1)

    elif args.model == 'segnet':
        model = SegNet(num_categories=nclass)


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

        image = Image.open(args.input + "/" + name).convert('RGB').resize((160, 160), Image.BILINEAR)

        # image = Image.open(args.in_path).convert('RGB')
        # target = Image.open(args.input + "/" + name).convert('L')
        # sample = {'image': image, 'label': target}
        image = transform_val(image)



        model.eval()
        t = time.time()



        if torch.cuda.is_available():
            image = image.cuda()
        with torch.no_grad():
            image = torch.unsqueeze(image, 0)
            output = model(image)

        prediction = decode_seg_map_sequence(torch.max(output[:], 1)[1].detach().cpu().numpy(),
                                                 dataset=dataset)

        prediction = prediction.squeeze(0)
        save_image(prediction, args.output + "/" + "{}_mask.png".format(name[0:-4]), normalize=False)
        #save_image(prediction, args.output, normalize=False)

        #print("Time spend {}s".format(time.time() - t))
        print("image:{} time: {} ".format(name, time.time() - t))


