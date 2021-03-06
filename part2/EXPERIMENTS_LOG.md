# Experiments comments

### resnext101_32x16d_wsl (two-stage, 1e-4 and 1e-7, Resize(in) + RandomCrop() + RandomHorizontalFlip + Many other augs+others (p=0.25), B 32 and 8)
- Best val (@ 14/15 epoch): acc: 0.918210; loss: 0.291043

----------------------------------------------------------------------------------

### densenet (finetune, 1e-6, Resize(in) + RandomCrop() + RandomHorizontalFlip, B 16)
- Best val (@ 38/40 epoch): acc: 0.888572; loss: 0.369511 191213204643

### densenet (Adam, finetune, 1e-5, Resize(in) + RandomCrop() + RandomHorizontalFlip + Many other augs+others (p=0.25), B 32)
- Best val (@ 26/50 epoch): acc: 0.914506; loss: 0.284608 (191214074310)

### resnext101_32x16d_wsl (wegihted loss, finetune, 1e-6, Resize(in) + CenterCrop() + RandomHorizontalFlip, B 8)
- Best val (@ 9/10 epoch): acc: 0.906811; loss: 0.558235 (191210134227)

----------------------------------------------------------------------------------

### resnext101_32x16d_wsl (SGD, finetune, 1e-5, Resize(in) + RandomCrop() + RandomHorizontalFlip + Many other augs+others (p=0.25), B 8)
- Best val (@ 36/40 epoch): acc: 0.922628; loss: 0.286203 191213211456

### resnext101_32x16d_wsl (finetune, 1e-7, Resize(in) + RandomCrop() + RandomHorizontalFlip + Many other augs+others (p=0.25), B 8)
- Best val (@ 20/20 epoch): acc: 0.923482; loss: 0.268667 (191213142715)
- Best val (@ 36/40 epoch): acc: 0.926475; loss: 0.256813 (191213204155)

### resnext101_32x16d_wsl (finetune, 1e-6, Resize(in) + RandomCrop() + RandomHorizontalFlip + Many other augs+others (p=0.25), B 8)
- Best val (@ 7/40 epoch): acc: 0.926902; loss: 0.259548 (191214074855)

### resnext101_32x16d_wsl (more data (0.9), finetune, 5e-7, Resize(in) + RandomCrop() + RandomHorizontalFlip + Many other augs+others (p=0.25), B 8)
- Best val (@ 14/20 epoch): acc: 0.927531; loss: 0.256636 (191214140044)

### resnext101_32x16d_wsl (wegihted sampler, finetune, 1e-7, Resize(in) + CenterCrop() + RandomHorizontalFlip + Many other augs+others (p=0.25), B 8)
- Best val (@ 15/20 epoch): acc: 0.848817; loss: 0.456121 (191213150219)

### resnext101_32x16d_wsl (finetune, 1e-6, Resize(in) + CenterCrop() + RandomHorizontalFlip + Many other augs (p=0.10), B 8)
- Best val (@ 3/20 epoch): acc: 0.920918; loss: 0.272968 (191210173917)

### resnext101_32x16d_wsl (wegihted sampler, finetune, 1e-6, Resize(in) + CenterCrop() + RandomHorizontalFlip + Many other augs (p=0.10), B 8)
- Best val (@ 15/20 epoch): acc: 0.918210; loss: 0.341146 (191210195835)

### resnext101_32x16d_wsl (finetune, 1e-6, Resize(in) + CenterCrop() + RandomHorizontalFlip, B 8)
- Best val (@ 3/5 epoch): acc: 0.922628; loss: 0.275009 (191210095047) lb 0.91023
- Best val (@ 1/5 epoch): acc: 0.921203; loss: 0.297496 (191210112736.csv) (different seed 14)

### SqueezeNet (finetune, 1e-4, Resize(in) + CenterCrop() + RandomHorizontalFlip)
- Best val (@ 29/30 epoch): acc: 0.851382; loss: 0.571721 (191209141649)

### SqueezeNet (finetune, Resize(in) + CenterCrop() + RandomHorizontalFlip)
- Best val (@ 27/30 epoch): acc: 0.800370; loss: 0.677375 (191209134718.csv)

----------------------------------------------------------------------------------

### resnext101_32x32d_wsl (Resize(in) + CenterCrop() + RandomHorizontalFlip, lr 1e-4, B 32)
- Best val (@ 24/30 epoch): acc: 0.897264; loss: 0.322359 (191209125959)

### SqueezeNet (1e-4, Resize(in) + CenterCrop() + RandomHorizontalFlip)
- Best val (@ 30/30 epoch): acc: 0.830863; loss: 0.517605 (191209151833)

### SqueezeNet (RandomResizedCrop)
- Best val (@ 15/20 epoch): acc: 0.821032; loss: 0.559046 (191209104117.csv)

### SqueezeNet (RandomResizedCrop + RandomHorizontalFlip)
- Best val (@ 8/20 epoch): acc: 0.821887; loss: 0.558620 (191209104715.csv)

### SqueezeNet (Resize(256) + RandomResizedCrop(224) + RandomHorizontalFlip)
- Best val (@ 11/20 epoch): acc: 0.825164; loss: 0.560400 (191209111305.csv)

### SqueezeNet (Resize(in) + CenterCrop() + RandomHorizontalFlip)
- Best val (@ 15/20 epoch): acc: 0.830436; loss: 0.544325 (191209111817.csv)


### SqueezeNet (Resize(in) + RandomResizedCrop(in) + RandomHorizontalFlip)
- Best val (@ 15/20 epoch): acc: 0.820604; loss: 0.564532 (191209113729.csv)

----------------------------------------------------------------------------------

### SqueezeNet (weighted loss)
- Best val (at 6 epoch): acc: 0.769165; loss: 0.892056 (191207183429.csv)

### resnext101_32x32d_wsl (weighted loss, lr 1e-4)
- Best val (at 31 epoch): acc: 0.861072; loss: 0.460283 (191207184712.csv)

----------------------------------------------------------------------------------

### SqueezeNet (weighted sampler)
- Best val (at 7 epoch): acc: 0.775435; loss: 0.713944

### resnext101_32x32d_wsl (weighted sampler, lr 1e-4, adam)
- Best val (at 16 epoch): acc: 0.859219; loss: 0.434397 (191207113036.csv)

----------------------------------------------------------------------------------

### DenseNet (weighted sampler, rest is the same as below)
- (bug) Best val (at 9 epoch): acc: 0.896552; loss: 0.325 (191206181024.csv)

### resnext101_32x32d_wsl (weighted sampler, lr 1e-4, adam)
- (bug) Best val (at 36 epoch): acc: 0.942291; loss: 0.190224 (191206180532.csv)

### SqueezeNet (weighted sampler, lr 1e-3)
- (bug) Best val (at 8 epoch): acc: 0.845255; loss: 0.499326

----------------------------------------------------------------------------------

### resnext101_32x48d_wsl (lr 1e-4) AdamW
- memory error at 2nd epoch

### resnext101_32x32d_wsl (lr 1e-4) AdamW
- Best val (at 15 epoch): acc: 0.903580; loss: 0.309840 (171m 56s) 191204205914.csv

### resnext101_32x32d_wsl (lr 1e-4)
- Best val (at 15 epoch): acc: 0.903580; loss: 0.309837 (167m 34s) 191204205825.csv
- Accuracy: 89.4; (191207181758.csv) accidentally deleted; lb: 0.903...

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
- Best val (at 5 epoch): acc: 0.830980; loss: 0.538799 (replicated: 191206175028.csv)
- Best val (at 6 epoch): acc: 0.832121; loss: 0.532076 (replicated: 191207162603.csv)
- Best val (at 5 epoch): acc: 0.825306; loss: 0.552011 (replicated: 191208113608.csv)

## TODO:
- [x] augmentations
- [x] finetuning whole model -> SqueezeNet (w 1e-4 (lower)) is better; ResNeXt is also better
- [x] weighted loss -> less worse but bad
- [x] weighted dataset -> significantly worse
