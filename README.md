# Show, Attend and Tell: Neural Image Caption Generation with Visual Attention

## A PyTorch implementation

## To Train

This was written in python3 so may not work for python2. Download the COOC dataset training and validation
images. Put them in `data/coco/imgs/train2014` and `data/coco/imgs/val2014` respectively.

Run the preprocessing to create the needed JSON files:

```python3
python generate_json_data.py
```

Start the training by running:

```python3
python train.py
```

The models will be saved in `model/` and the training statistics will be saved in `runs/`. To see the
training statistics, use:

```python3
tensorboard --logdir runs
```

## To Generate Captions

```python3
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
- [ ] Add performance statistics
- [x] Allow encoder to use resnet-152 and densenet-201

## References

[Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf)

[Original Theano Implementation](https://github.com/kelvinxu/arctic-captions)

[Neural Machine Translation By Jointly Learning to Align And Translate](https://arxiv.org/pdf/1409.0473.pdf)

[Karpathy's Data splits](https://cs.stanford.edu/people/karpathy/deepimagesent/)
