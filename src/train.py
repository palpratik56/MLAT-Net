import torch
from torchvision import transforms
from models.mlat import MLATNet
from utils.dataset import CVCDataset, KVASIRDataset, ISICDataset
from utils.preprocess import split_data
from tqdm import tqdm
import time
from torch.amp import autocast, GradScaler

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion,
                epochs, step_size):

    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    training_metrics = {
        'loss': [], 'precision': [], 'accuracy': [],'recall': [], 'dice': [], 'fsc': []  }
    validation_metrics = {
        'loss': [], 'precision': [], 'accuracy': [], 'recall': [], 'dice': [], 'fsc': [] }
    
    start_time = time.time()
    
    for epoch in range(epochs):
    
        # TRAINING
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
    
        train_running = {
            'loss': 0.0, 'precision': 0.0, 'accuracy': 0.0, 'recall': 0.0, 'dice': 0.0, 'fsc': 0.0 }
    
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
    
            optimizer.zero_grad(set_to_none=True)
    
            with autocast(device_type=device.type):
                outputs,_ = model(images)
                # outputs = model(images)
                loss = criterion(outputs, masks)
    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            train_running['loss'] += loss.item() * images.size(0)
    
            precision, recall, fsc, acc, dice = Performance_metrics(outputs, masks)
            train_running['precision'] += precision
            train_running['recall'] += recall
            train_running['accuracy'] += acc
            train_running['dice'] +=  dice
            train_running['fsc'] +=  fsc
    
        # Peak VRAM (training)
        # train_peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    
        epoch_metrics = {
            'loss': train_running['loss'] / len(train_loader.dataset),
            'precision': train_running['precision'] / len(train_loader),
            'recall': train_running['recall'] / len(train_loader),
            'accuracy': train_running['accuracy'] / len(train_loader),
            'dice': train_running['dice'] / len(train_loader),
            'fsc': train_running['fsc'] / len(train_loader)
        }
    
        for key in training_metrics:
            training_metrics[key].append(epoch_metrics[key])

    
        print(f'\nEpoch {epoch+1} [TRAIN]')
        print(f'Acc: {epoch_metrics["accuracy"]:.3f}, '
              f'Prec: {epoch_metrics["precision"]:.3f}, '
              f'Rec: {epoch_metrics["recall"]:.3f}, '
              f'Dice: {epoch_metrics["dice"]:.3f}, '
              f'Fsc: {epoch_metrics["fsc"]:.3f}, '
              f'Loss: {epoch_metrics["loss"]:.3f}')
        # print(f'Peak VRAM (Train): {train_peak_vram:.2f} GB')

        # VALIDATION
        model.eval()
        torch.cuda.reset_peak_memory_stats(device)
    
        val_running = {
            'loss': 0.0, 'precision': 0.0, 'accuracy': 0.0, 'recall': 0.0, 'dice': 0.0, 'fsc': 0.0 }
    
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
    
                with autocast(device_type=device.type):
                    outputs,_ = model(images)
                    # outputs = model(images)
                    loss = criterion(outputs, masks)
    
                val_running['loss'] += loss.item() * images.size(0)
    
                precision, recall, fsc, acc, dice = Performance_metrics(outputs, masks)
                val_running['precision'] += precision
                val_running['recall'] += recall
                val_running['accuracy'] += acc
                val_running['dice'] += dice
                val_running['fsc'] += fsc
    
        # Peak VRAM (validation)
        # val_peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    
        val_epoch_metrics = {
            'loss': val_running['loss'] / len(val_loader.dataset),
            'precision': val_running['precision'] / len(val_loader),
            'recall': val_running['recall'] / len(val_loader),
            'accuracy': val_running['accuracy'] / len(val_loader),
            'dice': val_running['dice'] / len(val_loader),
            'fsc': val_running['fsc'] / len(val_loader)
        }
    
        for key in validation_metrics:
            validation_metrics[key].append(val_epoch_metrics[key])
    
        print(f'Epoch {epoch+1} [VAL]')
        print(f'Acc: {val_epoch_metrics["accuracy"]:.3f}, '
              f'Prec: {val_epoch_metrics["precision"]:.3f}, '
              f'Rec: {val_epoch_metrics["recall"]:.3f}, '
              f'Dice: {val_epoch_metrics["dice"]:.3f}, '
              f'Fsc: {val_epoch_metrics["fsc"]:.3f}, '
              f'Loss: {val_epoch_metrics["loss"]:.3f}')
        # print(f'Peak VRAM (Val): {val_peak_vram:.2f} GB')
    
        scheduler.step()
    
        if (epoch + 1) % step_size == 0:
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
    
    training_time = (time.time() - start_time) / 60
    print(f'\nTotal training time: {training_time:.2f} minutes')

    return {'train': training_metrics, 'validation': validation_metrics, 
            'total_time_min': training_time }

def build_run_model(model, train_loader, val_loader, opti='Adam'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Wrap the model for multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
            
    model.to(device)
    
    if torch.__version__ >= "2.0":
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        
    torch.cuda.empty_cache

    if opti=='RMS':
        optimizer = optim.RMSprop(model.parameters(), lr=0.0001, momentum=0.95, 
                              alpha=0.99, weight_decay=0.0001)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0.0001)
    
    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
    criterion = nn.BCEWithLogitsLoss()
        
    # Ensure criterion and optimizer are also on the correct device
    criterion = criterion.to(device)
    
    metrics = train_model(model,train_loader, val_loader, optimizer, scheduler, 
                            criterion, epochs=40, step_size=20)
    return metrics

if __name__ == '__main__':
  data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),transforms.ToTensor(),
    ])
  # Initialize datasets
  # cv_dataset = CVCDataset(cv_image_folder, cv_mask_folder,transform=data_transforms)
  # kv_dataset = KVASDataset(kv_image_folder, kv_mask_folder,transform=data_transforms)
  dataset = ISICDataset(image_folder, mask_folder,transform=data_transforms)

  # cv_train_loader, cv_val_loader, cv_test_loader = split_data(cv_dataset)
  # kv_train_loader, kv_val_loader, kv_test_loader = split_data(kv_dataset)
  is_train_loader, is_val_loader, is_test_loader = split_data(dataset)

  #training
  mlat = MLAT(n_channels=3, n_classes=1)
  print("\n===== Running: MLAT =====")
  kv_metrics = build_run_model(mlat, kv_train_loader, kv_val_loader)
  # print("\n===== Running: CVC =====")
  # cv_metrics = build_run_model(model, cv_train_loader, cv_val_loader)
  # print("\n===== Running: ISIC - Adam =====")
  # is_metrics = build_run_model(model, is_train_loader, is_val_loader, opti='Adam')
