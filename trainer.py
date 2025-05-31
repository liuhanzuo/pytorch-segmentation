import torch
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from base import BaseTrainer, DataPrefetcher
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
from utils.lr_scheduler import CosineWithMinLR
class Trainer(BaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None, prefetch=True):
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, train_logger)
        
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])
        
        if self.device ==  torch.device('cpu'): prefetch = False
        if prefetch:
            self.train_loader = DataPrefetcher(train_loader, device=self.device)
            self.val_loader = DataPrefetcher(val_loader, device=self.device)
        # self.lr_scheduler = CosineWithMinLR(
        #     self.optimizer,
        #     num_epochs=config['trainer'].get('epochs', 80),
        #     iters_per_epoch=len(train_loader),
        #     warmup_epochs=config['lr_scheduler']['args'].get('warmup_epochs', 5),
        #     min_lr=1e-4
        # )

        torch.backends.cudnn.benchmark = True

    def _train_epoch(self, epoch):
        self.logger.info('\n')
            
        self.model.train()
        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel): self.model.module.freeze_bn()
            else: self.model.freeze_bn()
        self.wrt_mode = 'train'
        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (data, target) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            # data, target = data.to(self.device), target.to(self.device)

            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            output = self.model(data)
            if self.config['arch']['type'][:3] in ['PSP', 'SCT']:
                assert output[0].size()[2:] == target.size()[1:]
                assert output[0].size()[1] == self.num_classes 
                loss = self.loss(output[0], target)
                loss += self.loss(output[1], target) * 0.4
                output = output[0]
            else:
                assert output.size()[2:] == target.size()[1:]
                assert output.size()[1] == self.num_classes 
                loss = self.loss(output, target)

            if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            self.total_loss.update(loss.item())
            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar(f'{self.wrt_mode}/loss', loss.item(), self.wrt_step)

            # FOR EVAL
            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, _ = self._get_seg_metrics().values()
            
            # PRINT INFO
            tbar.set_description('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} | lr {:.3f}'.format(
                                                epoch, self.total_loss.average, 
                                                pixAcc, mIoU,
                                                self.batch_time.average, self.data_time.average,
                                                self.lr_scheduler.get_last_lr()[0]))
        # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics()
        for k, v in list(seg_metrics.items())[:-1]: 
            self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'{self.wrt_mode}/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
            #self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)

        # RETURN LOSS & METRICS
        log = {'loss': self.total_loss.average,
                **seg_metrics}

        if self.lr_scheduler is not None: 
            self.lr_scheduler.step()
            print(f'Learning rate: {self.lr_scheduler.get_last_lr()[0]}')
        return log

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target) in enumerate(tbar):
                #data, target = data.to(self.device), target.to(self.device)
                # LOSS
                output = self.model(data)
                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format( epoch,
                                                self.total_loss.average,
                                                pixAcc, mIoU))

            # WRTING & VISUALIZING THE MASKS
            val_img = []
            palette = self.train_loader.dataset.palette
            for d, t, o in val_visual:
                d = self.restore_transform(d)
                t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                val_img.extend([d, t, o])
            val_img = torch.stack(val_img, 0)
            val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
            self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-1]: 
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }

        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
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
        
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import torch
import time
from torchvision.utils import make_grid
from torchvision import transforms
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter

class CrossValidationTrainer:
    def __init__(self, model, loss, config, dataset_args, num_folds=5, device=None):
        self.model_class = type(model)
        self.model_args = config['arch']['args']
        self.loss = loss
        self.config = config
        self.dataset_args = dataset_args
        self.num_folds = num_folds
        self.results = []
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize visualization transforms
        # self._init_transforms()
        
    def _init_transforms(self):
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(
                self.dataset_args.get('MEAN', [0.28689529, 0.32513294, 0.28389176]),
                self.dataset_args.get('STD', [0.17613647, 0.18099176, 0.17772235])
            ),
            transforms.ToPILImage()
        ])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()
        ])
    
    def run(self):
        for fold_idx in range(self.num_folds):
            print(f"\n=== Running Fold {fold_idx + 1}/{self.num_folds} ===")
            
            # Initialize fresh model for each fold
            model = self.model_class(**self.model_args).to(self.device)
            optimizer = self._create_optimizer(model)
            lr_scheduler = self._create_lr_scheduler(optimizer)
            
            # Create data loaders
            train_loader, val_loader = self._create_data_loaders(fold_idx)
            
            # Train the model
            fold_result = self._train_fold(model, optimizer, lr_scheduler, train_loader, val_loader)
            self.results.append(fold_result)
        
        self._report_final_metrics()
        return self.results
    
    def _create_optimizer(self, model):
        optimizer_config = self.config.get('optimizer', {'type': 'SGD', 'args': {'lr': 0.01, 'weight_decay': 5e-4}})
        optimizer_type = getattr(torch.optim, optimizer_config['type'])
        return optimizer_type(model.parameters(), **optimizer_config['args'])
    
    def _create_lr_scheduler(self, optimizer):
        if 'lr_scheduler' in self.config:
            scheduler_config = self.config['lr_scheduler']
            scheduler_type = getattr(torch.optim.lr_scheduler, scheduler_config['type'])
            return scheduler_type(optimizer, **scheduler_config['args'])
        return None
    
    def _create_data_loaders(self, fold_idx):
        # Create base dataset
        dataset = CityScapesDataset(
            mode=self.dataset_args.get('mode', 'fine'),
            split='train',  # We'll use the same split for all folds
            root=self.dataset_args['data_dir'],
            mean=self.dataset_args.get('MEAN'),
            std=self.dataset_args.get('STD'),
            augment=self.dataset_args.get('augment', False),
            crop_size=self.dataset_args.get('crop_size'),
            base_size=self.dataset_args.get('base_size'),
            scale=self.dataset_args.get('scale', True)
        )
        
        # Setup KFold splits
        kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        all_indices = list(range(len(dataset)))
        train_indices, val_indices = list(kfold.split(all_indices))[fold_idx]
        
        # Create samplers
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.dataset_args['batch_size'],
            sampler=train_sampler,
            num_workers=self.dataset_args.get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.dataset_args['batch_size'],
            sampler=val_sampler,
            num_workers=self.dataset_args.get('num_workers', 4),
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _train_fold(self, model, optimizer, lr_scheduler, train_loader, val_loader):
        best_metrics = None
        num_epochs = self.config['trainer'].get('epochs', 50)
        log_step = self.config['trainer'].get('log_per_iter', 10)
        
        for epoch in range(1, num_epochs + 1):
            # Train epoch
            model.train()
            if self.config['arch']['args'].get('freeze_bn', False):
                if isinstance(model, torch.nn.DataParallel):
                    model.module.freeze_bn()
                else:
                    model.freeze_bn()
            
            train_metrics = self._run_epoch(
                model, optimizer, train_loader, 
                epoch, num_epochs, is_train=True, 
                log_step=log_step
            )
            
            # Validate epoch
            model.eval()
            val_metrics = self._run_epoch(
                model, None, val_loader, 
                epoch, num_epochs, is_train=False
            )
            
            # Update learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            # Track best model
            if best_metrics is None or val_metrics['Mean_IoU'] > best_metrics['Mean_IoU']:
                best_metrics = val_metrics
        
        return {
            'train_metrics': train_metrics,
            'val_metrics': best_metrics
        }
    
    def _run_epoch(self, model, optimizer, data_loader, epoch, num_epochs, is_train, log_step=10):
        metrics = {
            'batch_time': AverageMeter(),
            'data_time': AverageMeter(),
            'loss': AverageMeter(),
            'total_inter': 0,
            'total_union': 0,
            'total_correct': 0,
            'total_label': 0
        }
        
        tic = time.time()
        tbar = tqdm(data_loader, ncols=130)
        
        for batch_idx, (data, target) in enumerate(tbar):
            metrics['data_time'].update(time.time() - tic)
            
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            with torch.set_grad_enabled(is_train):
                output = model(data)
                
                # Handle auxiliary outputs (for PSPNet, SCTNet, etc.)
                if isinstance(output, tuple):
                    assert output[0].size()[2:] == target.size()[1:]
                    assert output[0].size()[1] == data_loader.dataset.num_classes
                    loss = self.loss(output[0], target)
                    if len(output) > 1:  # Auxiliary loss
                        loss += self.loss(output[1], target) * 0.4
                    output = output[0]
                else:
                    assert output.size()[2:] == target.size()[1:]
                    assert output.size()[1] == data_loader.dataset.num_classes
                    loss = self.loss(output, target)
                
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
            
            # Backward and optimize
            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update metrics
            metrics['loss'].update(loss.item())
            seg_metrics = eval_metrics(output, target, data_loader.dataset.num_classes)
            self._update_seg_metrics(metrics, *seg_metrics)
            
            # Measure elapsed time
            metrics['batch_time'].update(time.time() - tic)
            tic = time.time()
            
            # Logging
            if is_train and batch_idx % log_step == 0:
                self._log_batch(metrics, epoch, batch_idx, len(data_loader), is_train)
            
            # Update progress bar
            pixAcc, mIoU = self._compute_seg_metrics(metrics)
            tbar.set_description(
                '{} ({}/{}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f}'.format(
                    'TRAIN' if is_train else 'EVAL',
                    epoch, num_epochs,
                    metrics['loss'].average,
                    pixAcc, mIoU,
                    metrics['batch_time'].average,
                    metrics['data_time'].average
                )
            )
        
        return {
            'loss': metrics['loss'].average,
            **self._get_seg_metrics(metrics)
        }
    
    def _update_seg_metrics(self, metrics, correct, labeled, inter, union):
        metrics['total_correct'] += correct
        metrics['total_label'] += labeled
        metrics['total_inter'] += inter
        metrics['total_union'] += union
    
    def _compute_seg_metrics(self, metrics):
        pixAcc = 1.0 * metrics['total_correct'] / (np.spacing(1) + metrics['total_label'])
        IoU = 1.0 * metrics['total_inter'] / (np.spacing(1) + metrics['total_union'])
        mIoU = IoU.mean()
        return pixAcc, mIoU
    
    def _get_seg_metrics(self, metrics):
        pixAcc, mIoU = self._compute_seg_metrics(metrics)
        IoU = 1.0 * metrics['total_inter'] / (np.spacing(1) + metrics['total_union'])
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(len(IoU)), np.round(IoU, 3)))
        }
    
    def _log_batch(self, metrics, epoch, batch_idx, num_batches, is_train):
        # Implement your logging logic here (TensorBoard, WandB, etc.)
        pass
    
    def _report_final_metrics(self):
        print("\n=== Cross-Validation Results ===")
        val_mIoUs = [r['val_metrics']['Mean_IoU'] for r in self.results]
        val_losses = [r['val_metrics']['loss'] for r in self.results]
        
        print(f"Mean Val mIoU: {np.mean(val_mIoUs):.4f} (±{np.std(val_mIoUs):.4f})")
        print(f"Mean Val Loss: {np.mean(val_losses):.4f} (±{np.std(val_losses):.4f})")
        print("\nPer-fold results:")
        for i, result in enumerate(self.results):
            print(f"Fold {i+1}: mIoU={result['val_metrics']['Mean_IoU']:.4f}, Loss={result['val_metrics']['loss']:.4f}")
