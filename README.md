# Datasets
The datasets divided by 7:1:2 can be obtained from the following：
link：https://pan.baidu.com/s/1q6zHD6rgkmmLhGJSghk3uw 
extraction code：v79n
├── DEPDet
     └── dataset_SSDD
     └── dataset_HRSID

# Key module code corresponding to my thesis
 ├── DEPDet
     └── main
         └── ultralytics
             └── nn
                 └── DEPDet_modules
                     └── attention  # efficient multi-scale attention
                     └── block      # deformable convolution, cross-spatial multi-scale convolution', 'CSMSC2F','partial convolution'
                     └── head       # PCHead

# Training order
Go into the main/ directory and use the following commands, other parameters are adjusted in train.py:

   python train.py --yaml ultralytics/models/v8/DEPDet.yaml

# Verification and Test Commands
Go to the main/ directory and use the following commands, the other parameters are adjusted in val.py:

   python val.py --weight /root/my-tmp/project/DEPDet/main/runs/train/exp/weights/best.pt --split val
   python val.py --weight /root/my-tmp/project/DEPDet/main/runs/train/exp/weights/best.pt --split test

# Model prediction
Adjustments are made in mypre.py, which can be run straight away.

