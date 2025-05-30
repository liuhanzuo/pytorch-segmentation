import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision import transforms
import dataloaders
import models
from utils import transforms as local_transforms
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics
from collections import OrderedDict

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

class Evaluator:
    def __init__(self, model, val_loader, device, config):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.num_classes = val_loader.dataset.num_classes
        self.ignore_index = config.get('ignore_index', 255)
        
        # TRANSFORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(val_loader.MEAN, val_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])
        
        # Reset metrics
        self._reset_metrics()

    def evaluate(self):
        self.model.eval()
        val_visual = []
        
        with torch.no_grad():
            tbar = tqdm(self.val_loader, ncols=100)
            for batch_idx, (data, target) in enumerate(tbar):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Calculate metrics
                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)
                
                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])
                
                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                tbar.set_description('EVAL | PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format(
                    pixAcc, mIoU))
            
            # Visualize results
            self._visualize_results(val_visual)
            
            # Get final metrics
            metrics = self._get_seg_metrics()
            
            # Print final results
            print("\nEvaluation Results:")
            print(f"Pixel Accuracy: {metrics['Pixel_Accuracy']:.3f}")
            print(f"Mean IoU: {metrics['Mean_IoU']:.3f}")
            print("Class IoU:")
            for class_id, iou in metrics['Class_IoU'].items():
                print(f"  Class {class_id}: {iou:.3f}")
            
            return metrics

    def _visualize_results(self, val_visual):
        """Visualize inputs, targets and predictions"""
        val_img = []
        palette = self.val_loader.dataset.palette
        for d, t, o in val_visual:
            d = self.restore_transform(d)
            t, o = colorize_mask(t, palette), colorize_mask(o, palette)
            d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
            [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
            val_img.extend([d, t, o])
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
        
        # Save visualization
        if not os.path.exists('eval_results'):
            os.makedirs('eval_results')
        transforms.ToPILImage()(val_img).save('eval_results/val_samples.png')

    def _reset_metrics(self):
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }

def main(config, model_path):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    loader = get_instance(dataloaders, 'val_loader', config)
    model = get_instance(models, 'arch', config, loader.dataset.num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    
    # If during training, we used data parallel
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        # for gpu inference, use data parallel
        if "cuda" in device.type:
            model = torch.nn.DataParallel(model)
        else:
            # for cpu inference, remove module
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k.replace('module.', '') 
                print(name)
                new_state_dict[name] = v
            checkpoint = new_state_dict
    
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Get data loader
    val_loader = get_instance(dataloaders, 'val_loader', config)
    
    # Evaluate
    evaluator = Evaluator(model, val_loader, device, config)
    evaluator.evaluate()

if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Evaluation')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-m', '--model', required=True, type=str,
                        help='Path to the .pth model checkpoint to evaluate')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    config = json.load(open(args.config))
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args.model)
