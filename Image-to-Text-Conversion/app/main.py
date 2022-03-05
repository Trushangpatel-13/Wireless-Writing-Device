from flask import Flask,request,jsonify
import sys

import torch.backends.cudnn as cudnn

import string
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from model import Model
import torch
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, Subset
import torchvision.transforms as transforms
import base64
import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png'}
class Object(object):
    pass
class RawDataset(Dataset):

    def __init__(self, root, opt,fileObj):
        self.opt = opt
        self.image_path_list = []
        self.fileObj = fileObj
        self.image_path_list.append(root)
        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = 1

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        img = Image.open(self.fileObj).convert('L')
        print(img)

        #print(img)

        return (img, self.image_path_list[index])

class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)

        images, labels = zip(*batch)

        transform = ResizeNormalize((self.imgW, self.imgH))
        image_tensors = [transform(image) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        print(image_tensors)
        return image_tensors, labels

def demo(opt,file):
    dict={}
    print("Entered in Demo")

    """ model configuration """
    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    if opt.sensitive:
        opt.saved_model = "./TPS-ResNet-BiLSTM-Attn-case-sensitive.pth"
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=False)
    demo_data = RawDataset(root=opt.image_path, opt=opt, fileObj=file)  # use RawDataset
    print(demo_data)
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(1),
        collate_fn=AlignCollate_demo, pin_memory=True)
    print(demo_loader)
    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:

            print(image_tensors)
            batch_size = image_tensors.size(0)
            # print(batch_size)
            image = image_tensors
            # print(image)

            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            # dashed_line = '-' * 80
            # head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

            # print(f'{dashed_line}\n{head}\n{dashed_line}')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                dict['text'] = pred
                dict['score'] = "{:.4f}".format(float(confidence_score))

                # print(f'{pred:25s}\t{confidence_score:0.4f}')
        return dict
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/ocr',methods=['POST'])
def hello_world():
    if request.method == 'POST':
        file = request.form.get('file')
        print(file)
        """
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})
        """
        imgdata = base64.b64decode(file)
        #imagePath = ('/content/drive/MyDrive/image Text Recognition/upload/test.png')

        img = Image.open(io.BytesIO(imgdata))
        print(img)
        img.save('./test.png', 'png')
        print(img)


        opt = Object()
        opt.image_path = "./test/test"
        opt.batch_size = 1
        opt.saved_model = "./TPS-ResNet-BiLSTM-Attn-case-sensitive.pth"
        opt.batch_max_length = 10
        opt.imgH = 32
        opt.imgW = 100
        opt.character = '0123456789abcdefghijklmnopqrstuvwxyz'
        opt.sensitive = True
        opt.Transformation = 'TPS'
        opt.FeatureExtraction = 'ResNet'
        opt.SequenceModeling = 'BiLSTM'
        opt.Prediction = 'Attn'
        opt.num_fiducial = 20
        opt.input_channel = 1
        opt.output_channel = 512
        opt.hidden_size = 256
        """ vocab / character number configuration """
        if opt.sensitive:
            opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        cudnn.benchmark = True
        cudnn.deterministic = True
        opt.num_gpu = torch.cuda.device_count()

            #img = Image.open(file).convert('L')
            #print(img)
        file = './test.png'
        result = demo(opt,file)
        print(result)

        return jsonify({"Result":"1"})
        #except:
        #    return jsonify({'error': 'error during prediction'})



if __name__ == '__main__':
    app.run()
