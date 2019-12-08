from tqdm import tqdm
import torch
import torch.nn as nn
from utils import save_predictions
from typing import Dict, Union

def test_model(
        model: nn.Module,
        dataloaders: Dict[str, torch.utils.data.dataloader.DataLoader],
        device: torch.device,
        save_pred_path: Union[None, str] = None,
        is_inception: bool = False
    ) -> None:

    # Set model to evaluate mode
    model.eval()

    # Initialize predictions list
    predictions = []

    # class_to_idx dict (class: idx) and reverse
    class_to_idx = dataloaders['train'].dataset.class_to_idx
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    # Iterate over data.
    progress_bar = tqdm(dataloaders['test'], desc=f'Test: ')

    for i, (inputs, paths) in enumerate(progress_bar):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
        # accumulate the preductions after each batch
        assert len(preds.shape) == 1
        predictions.extend(list(zip(paths, preds.tolist())))

    # save predictions
    if save_pred_path is not None:
        save_predictions(save_pred_path, predictions, idx_to_class)
