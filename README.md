# Show, Attend and Tell: Neural Image Caption Generation with Visual Attention

## A PyTorch implementation

For a trained model to load into the decoder, use

- [VGG19](https://www.dropbox.com/s/eybo7wvsfrvfgx3/model_10.pth?dl=0)
- [ResNet152](https://www.dropbox.com/s/0fptqsw3ym9fx2w/model_resnet152_10.pth?dl=0)
- [ResNet152 No Teacher Forcing](https://www.dropbox.com/s/wq0g2oo6eautv2s/model_nt_resnet152_10.pth?dl=0)
- [VGG19 No Gating Scalar](https://www.dropbox.com/s/li4390nmqihv4rz/model_no_b_vgg19_5.pth?dl=0)

### Some training statistics

BLEU scores for VGG19 (Orange) and ResNet152 (Red) Trained With Teacher Forcing.

| BLEU Score | Graph                        | Top-K Accuracy   | Graph                              |
|------------|------------------------------|------------------|------------------------------------|
| BLEU-1     | ![BLEU-1](/assets/bleu1.png) | Training Top-1   | ![Train TOP-1](/assets/top1.png)   |
| BLEU-2     | ![BLEU-2](/assets/bleu2.png) | Training Top-5   | ![Train TOP-5](/assets/top5.png)   |
| BLEU-3     | ![BLEU-3](/assets/bleu3.png) | Validation Top-1 | ![Val TOP-1](/assets/val_top1.png) |
| BLEU-4     | ![BLEU-4](/assets/bleu4.png) | Validation Top-5 | ![Val TOP-5](/assets/val_top5.png) |

## To Train

This was written in python3 so may not work for python2. Download the COCO dataset training and validation
images. Put them in `data/coco/imgs/train2014` and `data/coco/imgs/val2014` respectively. Put the COCO
dataset split JSON file from [Deep Visual-Semantic Alignments](https://cs.stanford.edu/people/karpathy/deepimagesent/)
in `data/coco/`. It should be named `dataset.json`.

Run the preprocessing to create the needed JSON files:

```bash
python generate_json_data.py
```

Start the training by running:

```bash
python train.py
```

The models will be saved in `model/` and the training statistics will be saved in `runs/`. To see the
training statistics, use:

```bash
tensorboard --logdir runs
```

## To Generate Captions

```bash
python generate_caption.py --img-path <PATH_TO_IMG> --model <PATH_TO_MODEL_PARAMETERS>
```

## Todo

- [x] Create image encoder class
- [x] Create decoder class
- [x] Create dataset loader
- [x] Write main function for training and validation
- [x] Implement attention model
- [x] Implement decoder feed forward function
- [x] Write training function
- [x] Write validation function
- [x] Add BLEU evaluation
- [ ] Update code to use GPU only when available, otherwise use CPU
- [x] Add performance statistics
- [x] Allow encoder to use resnet-152 and densenet-161

## Captioned Examples

### Correctly Captioned Images

![Correctly Captioned Image 1](/assets/tennis.png)

![Correctly Captioned Image 2](/assets/right_cap.png)

### Incorrectly Captioned Images

![Incorrectly Captioned Image 1](/assets/bad_cap.png)

![Incorrectly Captioned Image 2](/assets/wrong_cap.png)

## References

[Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf)

[Original Theano Implementation](https://github.com/kelvinxu/arctic-captions)

[Neural Machine Translation By Jointly Learning to Align And Translate](https://arxiv.org/pdf/1409.0473.pdf)

[Karpathy's Data splits](https://cs.stanford.edu/people/karpathy/deepimagesent/)
