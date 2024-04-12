import numpy as np
import copy
import torch
import os
import pandas as pd
import ray
import datetime
import tempfile

from sklearn.model_selection import StratifiedKFold
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import wraps
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
from filelock import FileLock


CHECKPOINTS_DIR = os.path.abspath('./checkpoints')
LOGS_DIR = os.path.abspath('./logs')

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def load_data(train_set, test_set, batch_size):
    with FileLock(os.path.expanduser("~/.data.lock")):
        # Load datasets
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                                collate_fn=collate_fn, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                                collate_fn=collate_fn, pin_memory=True)
    return train_loader, test_loader

def save_torch_checkpoint(base_dir, file_name, info_dict):
    checkpoint_path = os.path.join(base_dir, file_name)
    torch.save(info_dict, checkpoint_path)

def train_step(model, train_set, test_set, epochs=10, 
                batch_size=32, criterion=CrossEntropyLoss(), 
                optimizer=None, scheduler=None,
                checkpoints_dir=CHECKPOINTS_DIR, logs_dir=LOGS_DIR,
                log_dir_suffix=None, 
                checkpoint_each=('batch', 1),
                **kwargs):
    
    ## DEFAULTS ---------------------------------------------------------------------
    # If a configuration is available, extract the hyperparameters, else get defaults
    config = kwargs.get('config', {})
    lr = config.get("lr", 1e-3)
    batch_size = config.get("batch_size", batch_size)
    epochs = config.get("epochs", epochs)
    checkpoints_dir = config.get("checkpoints_dir", checkpoints_dir)
    logs_dir = config.get("logs_dir", logs_dir)
    log_dir_suffix = config.get("log_dir_suffix", log_dir_suffix)
    checkpoints_dir = os.path.join(checkpoints_dir, log_dir_suffix)
    if config.get('optimizer', None) is not None:
        optimizer = config['optimizer']
    elif optimizer is None:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    if config.get('scheduler', None) is not None:
        scheduler = config['scheduler']
    elif scheduler is None:
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
    if config.get('criterion', None) is not None:
        criterion = config['criterion']
    elif criterion is None:
        criterion = CrossEntropyLoss()
    ## ------------------------------------------------------------------------------
    ### Check if GPU is available
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    ### Init writer for TensorBoard
    writer = SummaryWriter(logs_dir, comment=log_dir_suffix)
    ## ------------------------------------------------------------------------------
    ## Load datasets
    train_loader, test_loader = load_data(train_set, test_set, batch_size)

    model.to(device)

    # Function to load the latest checkpoint
    def load_checkpoint():
        latest_checkpoint_path = os.path.join(checkpoints_dir, 'latest_checkpoint.pth')
        if os.path.isfile(latest_checkpoint_path):
            print("Loading checkpoint...")
            checkpoint = torch.load(latest_checkpoint_path)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint['batch']
            best_acc = checkpoint['best_acc']
            return start_epoch, start_batch, best_acc
        return 0, 0, 0
    
    # Initialize or load from checkpoint
    start_epoch, start_batch, best_acc = load_checkpoint()
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_summary = pd.DataFrame(
        columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    ## ------------------------------------------------------------------------------
    ## Train loop
    for epoch in range(start_epoch, epochs):
        print(f"> Epoch {epoch+1}/{epochs} ------------------->")
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader, start=1):
            if epoch == start_epoch and batch_idx < start_batch:
                print(f"Skipping batch {batch_idx}")
                continue  # Skip batches until reaching the starting batch
            print(f"Batch {batch_idx}...")

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            cur_checkpoint = None

            # Save checkpoint after each batch if specified
            if checkpoint_each[0] == 'batch':
                if batch_idx % checkpoint_each[1] == 0:  
                    file_name = f'checkpoint_epoch_{epoch+1}_batch_{batch_idx}.pth'
                    cur_checkpoint = os.path.join(checkpoints_dir, file_name)
                    save_torch_checkpoint(checkpoints_dir, file_name=file_name,
                                          info_dict={
                                            'epoch': epoch,
                                            'batch': batch_idx,
                                            'model_state': model.state_dict(),
                                            'optimizer_state': optimizer.state_dict(),
                                            'best_acc': best_acc
                                        })
                    # Save latest checkpoint path
                    save_torch_checkpoint(checkpoints_dir, file_name='latest_checkpoint.pth',
                                            info_dict={
                                                'epoch': epoch,
                                                'batch': batch_idx,
                                                'model_state': model.state_dict(),
                                                'optimizer_state': optimizer.state_dict(),
                                                'best_acc': best_acc
                                            })
                print(f"Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_set)
        epoch_acc = running_corrects.double() / len(train_set)
        # Save checkpoint after each epoch if specified
        if checkpoint_each[0] == 'epoch':
            if (epoch+1) % checkpoint_each[1] == 0:
                file_name = f'checkpoint_epoch_{epoch+1}.pth'
                cur_checkpoint = os.path.join(checkpoints_dir, file_name)
                save_torch_checkpoint(checkpoints_dir, file_name=file_name,
                                      info_dict={
                                        'epoch': epoch,
                                        'model_state': model.state_dict(),
                                        'optimizer_state': optimizer.state_dict(),
                                        'best_acc': best_acc
                                    })
                # Save latest checkpoint path
                save_torch_checkpoint(checkpoints_dir, file_name='latest_checkpoint.pth',
                                      info_dict={
                                        'epoch': epoch,
                                        'model_state': model.state_dict(),
                                        'optimizer_state': optimizer.state_dict(),
                                        'best_acc': best_acc
                                    })
        
        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        ## ------------------------------------------------------------------------------
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(test_set)
        val_acc = val_corrects.double() / len(test_set)
        ## ------------------------------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report({"loss": loss.to('cpu').numpy() / len(test_set), 
                        "accuracy": val_corrects.double().item().to('cpu').numpy() / len(test_set)},
                        checkpoint=checkpoint)
        # Log validation metrics to TensorBoard
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        # Log validation metrics to table
        epochs_summary = epochs_summary.append({
            'epoch': epoch,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }, ignore_index=True)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model_best_checkpoint.pth'))
            
        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
    print('Best Val Acc: {:4f}'.format(best_acc))
    writer.close()  # Close the SummaryWriter to flush any remaining information to disk
    model.load_state_dict(best_model_wts)
    summary_file = os.path.join(logs_dir, log_dir_suffix, 'epochs_summary.csv')
    epochs_summary.to_csv(summary_file, index=False)
    return model, epochs_summary



def cross_validate(model_constructor, train_set, k=5,
                    epochs=10, 
                    batch_size=32, criterion=CrossEntropyLoss(), 
                    optimizer=None, scheduler=None,
                    checkpoints_dir=CHECKPOINTS_DIR, logs_dir=LOGS_DIR,
                    checkpoint_each=('batch', 1)):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=45621)
    labels = np.array([train_set.dataset.targets[i] for i in train_set.indices])

    cross_validation_results = []
    for i, (train_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f'> Fold {i+1}/{skf.get_n_splits()}...')
        train_subset = Subset(train_set, [train_set.indices[i] for i in train_index])
        test_subset = Subset(train_set, [train_set.indices[i] for i in test_index])

        # Create a new instance of the model for each fold
        model = model_constructor()
        # Define optimizer and scheduler (if not defined outside, create inside train_step)
        if optimizer is None:
            optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        if scheduler is None:
            scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
        
        fold_checkpoints_dir = os.path.join(checkpoints_dir, f'fold_{i+1}')
        fold_logs_dir = os.path.join(logs_dir, f'fold_{i+1}')

        # Training the model for the current fold
        trained_model, epochs_summary = train_step(
            model, train_subset, test_subset, epochs=epochs, batch_size=batch_size, 
            criterion=criterion, optimizer=optimizer, scheduler=scheduler, 
            checkpoints_dir=fold_checkpoints_dir, logs_dir=fold_logs_dir, 
            log_dir_suffix=f'fold_{i+1}', checkpoint_each=checkpoint_each
        )
        cross_validation_results.append({
            'fold': i+1,
            'model': trained_model,
            'epochs_summary': epochs_summary
        })
    
    return cross_validation_results



def hyperparameter_tuning(search_space, exp_name=None, num_samples=10, use_gpu=False):
    def decorator(train_func):
        @wraps(train_func)
        def wrapper(*args, **kwargs):
            # If search_space is None or empty, proceed with normal training
            if not search_space:
                print("No search space provided, running standard training...")
                return train_func(*args, **kwargs)

            # Initialize Ray, if not already done
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            
            # Define Ray Tune's scheduler
            scheduler = ASHAScheduler(
                metric="accuracy",
                mode="max",
                grace_period=1,
                reduction_factor=2
            )

            reporter = tune.JupyterNotebookReporter(
                parameter_columns={k: k for k in search_space.keys()},
                metric_columns=["loss", "accuracy", "training_iteration"]
            )

            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(
                        lambda config: train_func(config=config, *args, **kwargs)),
                    resources={"cpu": os.cpu_count(), "gpu": int(use_gpu)}
                ),
                tune_config=tune.TuneConfig(
                    scheduler=scheduler,
                    num_samples=num_samples,
                ),
                param_space=search_space,
                run_config=train.RunConfig(
                    name=exp_name if exp_name is not None else f"hp_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    progress_reporter=reporter
                )
            )

            analysis = tuner.fit()
            
            best_config = analysis.get_best_config(metric="val_acc", mode="max")
            print("Best hyperparameters found were: ", best_config)
            
            # Optionally, load and return the best model as well
            best_trial = analysis.get_best_trial("val_acc", "max", "last")
            best_checkpoint = best_trial.checkpoint.value
            model_state_dict = torch.load(os.path.join(best_checkpoint, "model.pth"))
            model = args[0]  # Assuming model is the first argument
            model.load_state_dict(model_state_dict)

            return model, best_config
        return wrapper
    return decorator

