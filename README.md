## MLPKANSegFormer

## Project Information
MLPKANSegFormer is a lightweight semantic segmentation model based on MLP and KAN (Kernel Attention Network), combined with the SegFormer framework, for semantic segmentation tasks on UAVid datasets.

## Catalog structure
``
📂 MLPKANSegFormer
├── 📂 models
│ ├── EncoderBlock.py
│ ├── EncoderBlock1.py
│ ├── MLPKANDecoder.py
│ ├── MLPKANSegFormer.py
│ ├── LayerNorm2d.py
│ ├── MixFFN.py
│ ├── MultiHeadAttention.py
│ ├── OverlapPatchMerging.py
│ ├── KAN.py
├─ 📂 data
│ ├── train_dataset.py
├── 📜 README.md
├── 📜 requirements.txt
├── 📜 train.py
├── 📜 inference.py

```

## Dependencies
Before you start using this project, make sure the following dependencies are installed on your environment.

```bash
pip install -r requirements.txt
```

``requirements.txt`` Example:
``txt
torch
numpy
opencv-python
albumentations
torchvision
torchvision
matplotlib
tqdm
``

## Data preparation
This project uses the UAVid dataset and the data is organized as follows:
```
📂 uavid_v1.5_official_release_image
├── 📂 uavid_train
│ ├── seq1
│ ├── images/*.png
│ │ ├── labels/*.png
├── 📂 uavid_val
│ ├── seq16
│ ├── images/*.png
│ │ ├── labels/*.png
├── 📂 uavid_test
│ ├── seq21
│ ├── images/*.png
```

## Train the model
Run the following command to train:
```bash
python train.py --data_path . /uavid_v1.5_official_release_image --batch_size 64 --epochs 2000
```
Default parameters:
- `batch_size`: 64
- `epochs`: 500
- `learning_rate`: 3e-4

## Evaluate the model
After training, you can evaluate the model on the validation set using ``inference.py``:
``bash
python inference.py --model_path best_model.pth --data_path . /uavid_v1.5_official_release_image
``

## Results
The current experimental results show that on the UAVid dataset, MLPKANSegFormer achieves **mIoU improvement of 2.6%**, which is a better performance than the standard SegFormer.

## Contribute
If you would like to contribute code or suggest improvements, please submit a PR or issue.

## License
MIT License
