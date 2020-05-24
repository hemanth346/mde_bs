## Objective :

Train a model to predict,
1. mask for the fg object **(Background Subtraction)**
2. Depth map for the image
 **(Monocular Depth Estimation)**

  given 2 images
   1. a background(bg) image and
   1. Object inside the above background
   > i.e. foreground*(fg)* overlaid on background (fg_bg)


## Usecase
Can be used mostly for security checks and in CC TV footage videos

1. the mask can be used to check if a person is in the frame and the movement of the person
1. the depth map can be used to determine if the person has entered any restricted zone

To be made real-time the inference time has to be very less and be able to run on the high fps.

The objective of this project is to get a working model with less number of parameters (and less size).


## Dataset Overview
  Own dataset has been curated/created for the task at hand.

In total, dataset contains
  - 100 background images

![bg_images_thumbnail](https://github.com/hemanth346/mde_bs/blob/master/images/bg.png)

  - 100 foreground(fg) images

![fg_images_thumbnail](https://github.com/hemanth346/mde_bs/blob/master/images/fg.png)

  - 100 respective masks for fg

  ![fg_masks_thumbnail](https://github.com/hemanth346/mde_bs/blob/master/images/fg_masks.png)

  - 400k foreground with background(fg_bg) images

  ![fg_bg_thumbnail](https://github.com/hemanth346/mde_bs/blob/master/images/fg_bg.png)

  - 400k respective depth images

  ![depth_img_thumbnail](https://github.com/hemanth346/mde_bs/blob/master/images/depth_maps.png)

  - 400k respective masks for fg_bg images

  ![fg_bg_masks_thumbnail](https://github.com/hemanth346/mde_bs/blob/master/images/fg_bg_masks.png)


Calculated mean, std for each of the image set...

Complete details can be found at this link [dataset creation][bc7410ba]


[bc7410ba]: github.com/hemanth346 "Dataset Creation"

## Iterations

#### 1.Get the setup right (for 2 problems)

Worked with minimal data of 12k images to test the waters before diving in.

- [x] ##### Dataloader

Each of the folders in fg_bg, depth_maps, fg_bg_masks -  has separate zip file for every bg file. Images are read directly from the zip file to save on disk space and extraction time.

List of all paths are stored obtained from fg_bg and when dataset is called by dataloader, corresponding images from other folders/zip files are obtained
```
  for file in os.listdir(fg_bg_dir):
     fname = os.path.join(fg_bg_dir, file)
     if zipfile.is_zipfile(fname):
         self.fg_bg+=[x.filename for x in zipfile.ZipFile(fname).infolist()]

def __getitem__(self, index):
   bg = self.fg_bg[index].split('_')[0]
   bg_file = Path(self.data_root).joinpath('bg' ,bg+'.jpg')

   bg_img = np.array(Image.open(str(bg_file)))
   fg_bg_img = self.read_img_from_zip(f'{self.data_root}/fg_bg/{bg}.zip', self.fg_bg[index])
   mask_img = self.read_img_from_zip(f'{self.data_root}/fg_bg_masks/{bg}.zip', self.fg_bg[index])
   depth_img = self.read_img_from_zip(f'{self.data_root}/depth_maps/{bg}.zip', self.fg_bg[index])

```

_A helper function is added to the dataset class to convert the zipfile into PIL file for regular transforms and numpy array for albumentation transformations_
```
def read_img_from_zip(self, zip_name, file_name, array=True):
    imgdata = zipfile.ZipFile(zip_name).read(file_name)
    img = Image.open(io.BytesIO(imgdata))
    # img = img.convert("RGB")
    if array:
        img = np.array(img)
        return img
    # PIL image
    return img

```

- [x] ##### Basic transforms is set for each of the image set.

- Transformations used are : _RandomCrop, HorizontalFlip, Resize(64x64), Normalization_

Used [albumentation library][07ffb173] for transformations. Read about its advantages [here][8a687b30] and [here][3dd4fa4a]

Albumentations doesn't support loading PIL images directly and works with numpy arrays. _Hence we have to modify the dataset class accordingly_. I found this [notebook][33e23d49] very useful as a guide.

Albumentations also have support to pass masks/weights along with original image and get the same transforms applied on them [example here][49e22e45] or we can create our own [custom targets][f98e539c] to be sent for the compose, which are useful if we have multiple images not linked with each other but need same transforms.

Examples of some of the transforms applied for segmentation problem can be found in this [notebook][8a740691]

  [07ffb173]: https://github.com/albumentations-team/albumentations/ "GitHub link"
  [8a687b30]: https://arxiv.org/pdf/1809.06839 "arxiv paper"
  [3dd4fa4a]: https://medium.com/@ArjunThoughts/albumentations-package-is-a-fast-and-%EF%AC%82exible-library-for-image-augmentations-with-many-various-207422f55a24 "Medium article"

  [33e23d49]: https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/migrating_from_torchvision_to_albumentations.ipynb "albumentation transform notebook"

  [49e22e45]: http://www.andrewjanowczyk.com/employing-the-albumentation-library-in-pytorch-workflows-bonus-helper-for-selecting-appropriate-values "transforms for mask article"

  [f98e539c]: https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example_multi_target.ipynb "Multi Target Notebook"

  [8a740691]: https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example_kaggle_salt.ipynb "examples for segmentation - Notebook"

- [x] ##### Setup basic model

> Model is expected to give us a mask and depth map for foreground given 2 images.

```
 Can we think of using a single fg_bg image and predict it as well.?

   Will that change the problem scope, from trying to identify foreground object in given background image the model will be trying to find the mask and depth in any general setting..?

   Will experiment with that as well if time permits
 ```

 - using inputs as
    - [ ] only fg_bg
    - [x] both bg and fg_bg
  - Predicting
    - [x] Only mask
    - [x] Only depth map
    - [x] both mask and depth_maps  

After looking at the 64x64 images, the images became pixelted and share edges are not available. RF for gradient/first layer was set at 5 pixels.

> Since the output is also to be as the same size of input, we have to either do a Transpose conv or maintiain the size without doing any maxpool/stride 2 and no padding. Without any stride/maxpool - to get the receptive field of image size, in final layer, requires lot of convolutional layers.


Used Group convolutions with 2 groups of 3 channels each for first few convolutions. The idea being for the network to be able to learn for low level features from both the images intially.

Initial network is created without using  transpose conv/de-conv. And accounting for RF, below is the brief network summary. Heavy model with more params

```
Total params: 3,769,664
Trainable params: 3,769,664
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.09
Forward/backward pass size (MB): 462.03
Params size (MB): 14.38
Estimated Total Size (MB): 476.51
```
- Training took around 40 mins per epoch on 16G Tesla P100


<!-- Model such that the ground truth image size output is achieved in final layer. We can use **view** on tensor in final layer to get required output size or do tranpose/deConv convolvution.

As a baseline, we are using a ResNet 18 model with final layer. **Training Notebook** -->

<!-- If transpose conv/upsampling is not used, no. of parameters required to reach the  receptive field of image is very high.
If the image sizes are reduced drastically it'll not serve as a baseline model. But since the objective is to get the setup correct, images have been resized to 64x64 using  transforms

Architecture diagram :
**Todo add image**

 -->

- [x] ##### Setup tensorboard on colab

Used below extension and magic function to access TensorBoard. This will open the tensorboard in the cell output
```
#Load extension
%load_ext tensorboard

logs_base_dir = 'logs'
%tensorboard --logdir {logs_base_dir}
```
To write to tensorboard we can use summary writer from torch.utils
```
# TensorBoard support
from torch.utils.tensorboard import SummaryWriter
```

- [ ] ##### Make sure model is training

Model was run for 2 epochs without any issue..


<!-- ## Preprocessing and DataPipeline

Images are read from zip files and resized to 112x112. Enough spatial information is retained for good enough details.  



## ToDo:

  Since the model is trying to predict dense ouput, the architecture should have de-Convolvution(Transpose Convolvution). -->

#### 2. Get the basic skeleton

Different model architectures were tried out **involving transpose convolutions**.

Architecture search was done and found that  similar problems involve one form or another of U-Net. Taken below two architectures as starting point

  - U-Net [arxiv pdf;][f010ac45] [Github Code;][2922dbe6] [article][da501ee2]


  - Deep Residual U-Net [arxiv pdf;][3aad73d3] [Github Code(Pytorch);][714b6f76]
  [Notebook(Keras)][6dbd0345]


Architecutures experimented are

1. [x] transpose convolutions using width of 2
> ConvTranspose is convolution and has trainable kernels while Upsample is a simple interpolation (bilinear, nearest etc.).. Transpose has learning parameter while Up-sampling has no-learning parameters. Using Upsampling can make inference or training faster as it does not require to update weight or compute gradient, but since the input image is already pixelated, using transpose conv with cost of additional parameters and model size
  ```
  ================================================================
  Total params: 1,190,272
  Trainable params: 1,190,272
  Non-trainable params: 0
  ----------------------------------------------------------------
  Input size (MB): 0.09
  Forward/backward pass size (MB): 139.16
  Params size (MB): 4.54
  Estimated Total Size (MB): 143.80
  ```
  - Training took around 15 mins per epoch on 16G Tesla P100

1. [ ] Modified DeepResUNet architecuture.
> Have to revisit the architecture, to reduce training memory.

```
================================================================
Total params: 1,765,421
Trainable params: 1,765,421
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.09
Forward/backward pass size (MB): 43093.52
Params size (MB): 6.73
Estimated Total Size (MB): 43100.34
----------------------------------------------------------------

```

Proceeding with ConvTranspose as the results seems to be good with limited training

#### 3. Loss functions

Above models were trained with Binary Cross entropy loss, which does a pretty good job. But for the problem at hand bce misses out on some additional aspects which are covered below

Below are the loss functions used
- Dice coefficient based Soft Dice Loss for Mask.
    > Read my post about the loss [here]
- Multi-Scale Structural Similarity  loss for Depth Maps - [implementation][909de322]

**Dice loss implementation**
```
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        num = targets.size(0)  # Number of batches

        score = dice_coeff(probs, targets)
        score = 1 - score.sum() / num
        return score
```

  [909de322]: https://github.com/jorge-pessoa/pytorch-msssim "Github - PyTorch differentiable Multi-Scale Structural Similarity (MS-SSIM) loss"

  [f010ac45]: https://arxiv.org/abs/1505.04597 "Paper pdf"
  [2922dbe6]: https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py "Models class"

  [3aad73d3]: https://arxiv.org/pdf/1711.10684.pdf "Paper pdf"
  [714b6f76]: https://github.com/galprz/brain-tumor-segmentation/blob/master/src/models.py "Model"
  [6dbd0345]: https://github.com/nikhilroxtomar/Deep-Residual-Unet/blob/master/Deep%20Residual%20UNet.ipynb "Keras notebook"
  [da501ee2]: https://towardsdatascience.com/u-net-b229b32b4a71 "Medium article"



#### 4. Augmentations and Hyperparam tuning
 - todo



<!-- # Read
https://www.sicara.ai/blog/2019-28-10-deep-learning-memory-usage-and-pytorch-optimization-tricks

https://towardsdatascience.com/getting-started-with-pytorch-part-1-understanding-how-automatic-differentiation-works-5008282073ec

https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255

 -->
  <!-- - [ ] **Copy Code from video and understand**


  1. [ ] **Dataset**
    - [ ] Collect own data
    - [ ] Prepare dataset for the task
    - [ ] find dataset statistics

  1. [ ] **Make API for data to be consumed by model**
    - [ ] Dataset class
    - [ ] Dataloader

  1. [ ] **Design Network architecture and model the data**

    - [ ] model for background subtraction only
      - [ ] Loss functions
        - [ ] L1
        - [ ] L1
        - [ ] L1
      - [ ] Metrics
        - [ ] M1
        - [ ] M1
        - [ ] M1
      - [ ] Hyper param tuning
    - [ ] model for Monocular Depth Estimation only
      - [ ] Loss functions
      - [ ] Metrics
      - [ ] Hyper param tuning

    - [ ] Model to do both at once
      - [ ] Loss functions
      - [ ] Metrics
      - [ ] Hyper param tuning

 -->
