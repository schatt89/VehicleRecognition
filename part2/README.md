# Experiments comments


### SqueezeNet (weighted) CUDA:2
- Best val (at 7 epoch): acc: 0.775435; loss: 0.713944

### resnext101_32x32d_wsl (weighted, lr 1e-4, adam)
- Best val (at 16 epoch): acc: 0.859219; loss: 0.434397 (191207113036.csv)

----------------------------------------------------------------------------------

### DenseNet (weighted, rest is the same as below)
- (bug) Best val (at 9 epoch): acc: 0.896552; loss: 0.325 (191206181024.csv)

### resnext101_32x32d_wsl (weighted, lr 1e-4, adam)
- (bug) Best val (at 36 epoch): acc: 0.942291; loss: 0.190224 (191206180532.csv)

### SqueezeNet (weighted, lr 1e-3)
- (bug) Best val (at 8 epoch): acc: 0.845255; loss: 0.499326

----------------------------------------------------------------------------------

### resnext101_32x48d_wsl (lr 1e-4) AdamW
- memory error at 2nd

### resnext101_32x32d_wsl (lr 1e-4) AdamW
- Best val (at 15 epoch): acc: 0.903580; loss: 0.309840 (171m 56s) 191204205914.csv

### resnext101_32x32d_wsl (lr 1e-4)
- Best val (at 15 epoch): acc: 0.903580; loss: 0.309837 (167m 34s) 191204205825.csv

### resnext101_32x32d_wsl
- Best val (at 3 epoch): acc: 0.893738; loss: 0.360059 (83m 45s) 191204192309.csv

### resnext101_32x8d
- Best val (at 7 epoch): acc: 0.883754; loss: 0.362519

### resnext50_32x4d
- Best val (at 9 epoch): acc: 0.867351; loss: 0.414604

### ResNet
- Best val (at 3 epoch): acc: 0.841249; loss: 0.487053

### Vgg
- Best val (at 7 epoch): acc: 0.853659; loss: 0.463699

### Inception
- Best val (at 7 epoch): acc: 0.855798; loss: 0.453688 (191204114443.csv)

### Densenet 
- Best val: acc: 0.864499; (191204113332.csv)

### SqueezeNet
- Best val (at 5 epoch): acc: 0.830980; loss: 0.538799 (replicated: 191206175028.???)


## TODO:
- [ ] weighted loss
- [ ] weighted dataset
