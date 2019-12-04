import matplotlib.pyplot as plt
import os

from typing import Tuple, Dict, Union

def save_predictions(
    save_path: str,
    predictions: Tuple[str, int],
    idx_to_class: Dict[int, str]
) -> None:
    '''
        Format:
        Id,Category
        0,Car
        1,Catepillar
    '''
    with open(save_path, 'w') as outf:
        # header
        outf.write('Id,Category\n')

        # other lines
        for (pred_path, pred_idx) in predictions:
            # extract Id from the filename
            Id = int(os.path.split(pred_path)[1].strip('.jpg'))
            outf.write(f'{Id},{idx_to_class[pred_idx]}\n')

    print(f'Wrote preds to {save_path}')


def plot_images(images, data_dir, cls_true, cls_pred=None):
    """
    Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    """
    label_names = sorted(os.listdir(data_dir))
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot img
        ax.imshow(images[i, :, :, :] * 0.25 + 0.45, interpolation='spline16')

        # show true & predicted classes
        cls_true_name = label_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name
            )
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
