import torch
import os
import pandas as pd
import ray
import tempfile
import zipfile
import shutil
import glob

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision.models import get_model, get_model_weights
from torch.utils.tensorboard import SummaryWriter
from functools import wraps
from ray.tune.schedulers import ASHAScheduler, TrialScheduler
from ray.tune import TuneConfig, Tuner
from ray.train import Checkpoint, ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer, prepare_data_loader, prepare_model
from filelock import FileLock
from tqdm import tqdm
from ciri_utils.datautils import CIRI_Dataset
from collections import defaultdict

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

SEED = 53653


def _load_ciri_dataset(root_base, root_augmented, transform):
    """Load the CIRI dataset from the given paths and apply the given transform.

    Args:
        root_base (str): The path to the base dataset.
        root_augmented (str): The path to the augmented dataset.
        transform (torchvision.transforms.Compose): The transformation to apply to the dataset.

    Returns:
        CIRI_Dataset: The CIRI dataset with the given transformation.
    """
    ciri_dataset = CIRI_Dataset(
        root=root_base,
        root_augmented=root_augmented,
        transform=transform,
    )
    return ciri_dataset

def _sample_indices_from_prop(proportion, 
                              root_base, 
                              root_augmented):
    """Generates a stratified random sample of indices from the dataset given a proportion.

    Args:
        proportion (float): the proportion of the dataset to sample
        root_base (str): the path to the base dataset
        root_augmented (str): the path to the augmented dataset
    
    Returns:
        tuple: the sampled indices and the dataset
    """
    ciri_dataset = _load_ciri_dataset(root_base, root_augmented, transform=None)
    info_df = ciri_dataset.info
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=proportion, random_state=SEED)
    for _, sample_index in splitter.split(info_df['path'], info_df['label']):
        sampled_indices = sample_index
    return sampled_indices, ciri_dataset

def _get_fold_indices(info_df, sample_idx=None, outer_cv_k=5, inner_cv_k=0):
    """
    Generates indices for outer (and optional inner) stratified k-fold cross-validation.

    Args:
        info_df (pandas.DataFrame): DataFrame containing at least 'path' and 'label' columns for the dataset.
        sample_idx (numpy.ndarray, optional): Indices of a sampled subset of the dataset. Defaults to None.
        outer_cv_k (int): Number of folds for outer cross-validation. Defaults to 5.
        inner_cv_k (int): Number of folds for inner cross-validation. Defaults to 0, indicating no inner CV.

    Returns:
        Generator of tuples: For each outer fold, yields (train_idx, test_idx) for outer CV, and if inner_cv_k > 0,
        each (train_idx, test_idx) is followed by a generator yielding (inner_train_idx, inner_test_idx) for inner CV.
    """
    if sample_idx is not None:
        # Filter the DataFrame to only include sampled indices if provided
        info_df = info_df.loc[sample_idx]

    outer_cv = StratifiedKFold(n_splits=outer_cv_k, shuffle=True, random_state=SEED)
    
    def _get_true_idx(info_df_sub, idx_gen):
        for train_idx, test_idx in idx_gen:
            true_train_idx = info_df_sub.iloc[train_idx].index.tolist()
            true_test_idx = info_df_sub.iloc[test_idx].index.tolist()
            yield (true_train_idx, true_test_idx)
    
    for outer_train_idx, outer_test_idx in outer_cv.split(info_df, info_df['label']):
        true_outer_train_idx = info_df.iloc[outer_train_idx].index.tolist()
        true_outer_test_idx = info_df.iloc[outer_test_idx].index.tolist()
        if inner_cv_k > 0:
            # Perform inner CV split on the outer training set
            inner_cv = StratifiedKFold(n_splits=inner_cv_k, shuffle=True, random_state=SEED)
            inner_folds = inner_cv.split(info_df.iloc[outer_train_idx], info_df.iloc[outer_train_idx]['label'])
            true_inner_folds = _get_true_idx(info_df.iloc[outer_train_idx], inner_folds)
            yield (true_outer_train_idx, true_outer_test_idx, true_inner_folds)
        else:
            yield (true_outer_train_idx, true_outer_test_idx)

def _obtain_train_test(ciri_dataset, proportion):
    """Generates the random split for train and test set from the whole dataset.

    Args:
        ciri_dataset (CIRI_Dataset): whole dataset
        proportion (float): number between 0 and 1 representing the proportion 
        of the train set

    Returns:
        train_set, test_set: 2 Subset objects
    """    
    return random_split(
        ciri_dataset,
        [proportion, 1 - proportion],
        generator=torch.Generator().manual_seed(SEED),
    )

def _sample_dataset(ciri_dataset, indices):
    """Generates a subset of the dataset from the given indices.

    Args:
        ciri_dataset (CIRI_Dataset): the whole dataset
        indices (list): the indices to sample

    Returns:
        Subset: the subset of the dataset
    """    
    return torch.utils.data.Subset(ciri_dataset, indices)

def _find_classes(subset):
    if hasattr(subset, "class_counts"):
        return subset.class_counts
    elif hasattr(subset, "dataset"):
        return _find_classes(subset.dataset)
    else:
        return None

def _collate_fn(batch):
    """Function needed to skip broken files during training.
    """    
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def _get_dataloaders(train_set, test_set, batch_size):
    """Creates the data loaders.

    Args:
        train_set (Dataset): the train set, as a Subset object
        test_set (Dataset): the test set, as a Subset object
        batch_size (int): the batch size

    Returns:
        train_loader, test_loader: 2 DataLoader objects
    """    
    pin_memory = True if DEVICE == "cuda" else False
    with FileLock(os.path.expanduser("~/.data.lock")):
        # Load datasets
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_fn,
            pin_memory=pin_memory,
        )
    return train_loader, test_loader


def _train_step(model, train_loader, epoch, criterion, optimizer):
    """The single train step for a training loop.

    Args:
        model (torchvision.model): The initialised model
        train_loader (torchvision.utils.data.DataLoader): the dataloader for the train set
        epoch (int): the current epoch
        criterion : the loss function object
        optimizer : the optimizer object

    Returns:
        tuple: the trained model for this step, the train loss and the accuracy
    """    
    if ray.train.get_context().get_world_size() > 1:
        # Required for the distributed sampler to shuffle properly across epochs.
        train_loader.sampler.set_epoch(epoch)
    model.train()
    train_loss, num_correct, num_total = 0, 0, 0
    for X, y in tqdm(train_loader, desc=f"Epoch (training) {epoch}"):
        pred = model(X)
        loss = criterion(pred, y)
        train_loss += loss.item()
        num_total += y.shape[0]
        num_correct += (pred.argmax(1) == y).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    accuracy = num_correct / num_total
    return model, train_loss, accuracy


def _eval_step(model, test_loader, epoch, criterion, additional_metrics=None):
    """The single evaluation step for a training loop.
    
    Args:
        model (torchvision.model): The initialised model
        test_loader (torchvision.utils.data.DataLoader): the dataloader for the test set
        epoch (int): the current epoch
        criterion : the loss function object
        
    Returns:
        tuple: the test loss and the accuracy
    """    
    model.eval()
    test_loss, num_correct, num_total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc=f"Epoch (test) {epoch}"):
            pred = model(X)
            loss = criterion(pred, y)

            test_loss += loss.item()
            num_total += y.shape[0]
            num_correct += (pred.argmax(1) == y).sum().item()
            
            # Collect all predictions and true labels for additional metrics
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())


    test_loss /= len(test_loader)
    accuracy = num_correct / num_total
    add_metrics = {}
    if additional_metrics:
        if "confusion_matrix" in additional_metrics:
            conf_matrix = confusion_matrix(all_labels, all_preds)
            add_metrics["confusion_matrix"] = (conf_matrix, all_labels, all_preds)
        if "precision" in additional_metrics:
            add_metrics["precision"] = precision_score(all_labels, all_preds, average='weighted')
        if "recall" in additional_metrics:
            add_metrics["recall"] = recall_score(all_labels, all_preds, average='weighted')
        if "f1" in additional_metrics:
            add_metrics["f1"] = f1_score(all_labels, all_preds, average='weighted')
    return test_loss, accuracy, add_metrics

def _unfreeze_layers(model, layers: list):
    model_layers = reversed(list(model.named_children()))
    for name, layer in model_layers:
        if name in layers:
            for param in layer.parameters():
                param.requires_grad = True
    return model

def _train_func_per_worker(config):
    """The training function for the TorchTrainer. 
    This function is called by each worker.
    
    Args:
        config (dict): the configuration dictionary. Mandatory
        items are "model", "data_folders"
    
    Returns:
        None
    """    
    # Everything expected to be in config dict, no individual args
    ## Model is always mandatory and expects the name of an existing
    ## torchvision model (as a string)
    model_name = config.get("model", None)
    if model_name is None:
        raise ValueError("model is a mandatory parameter")
    data_folders = config.get("data_folders", None)
    if data_folders is None:
        raise ValueError("data_folders is a mandatory parameter")
    data_prop = config.get("data_prop", 0.8)
    lr = config.get("lr", 1e-3)
    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 10)
    weight_decay = config.get("weight_decay", 1e-2)
    scheduler_factor = config.get("scheduler_factor", 0.1)
    scheduler_patience = config.get("scheduler_patience", 10)
    if config.get("criterion", None) is not None:
        criterion = config["criterion"]
    else:
        criterion = CrossEntropyLoss()
    
    # [0] Initialization
    ### Get the model weights
    model_weights = get_model_weights(model_name)
    if config.get("weights", None) is not None: # For transfer learning
        weights_type = config["weights"]
    else:
        weights_type = "DEFAULT"
    weights = getattr(model_weights, weights_type)
    dataset = _load_ciri_dataset(data_folders[0], data_folders[1], weights.transforms())
    if config.get("sample_indices", None) is not None and config.get("train_test_idx", None) is None:
        sample_indices = config["sample_indices"]
        dataset = _sample_dataset(dataset, sample_indices)
    ### Get the number of classes and the model adjusted
    n_classes = _find_classes(dataset).shape[0]
    if config.get("weights", None) is None:
        model = get_model(model_name, num_classes=n_classes)
    else:
        # Prepare for transfer learning
        model = get_model(model_name, weights=weights)
        for param in model.parameters(): # Freeze all layers
            param.requires_grad = False
        layers = [name for name, _ in model.named_children()]
        last_layer = layers[-1]
        setattr(model, last_layer, torch.nn.Linear(model.fc.in_features, n_classes))
        # For fine tuning
        if config.get("unfreeze", None) is not None:
            model = _unfreeze_layers(model, config["unfreeze"])
        
    if config.get("optimizer", None) is not None:
        optimizer = config["optimizer"]
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if config.get("scheduler", None) is not None:
        scheduler = config["scheduler"]
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, "min", factor=scheduler_factor, patience=scheduler_patience
        )
    ### Get train and test set
    #### If idx are provided (e.g. for cross-validation), use them
    if config.get("train_test_idx", None) is not None:
        train_idx, test_idx = config["train_test_idx"]
        train_set = _sample_dataset(dataset, train_idx)
        test_set = _sample_dataset(dataset, test_idx)
    else:
        #### Otherwise, split the dataset randomly
        train_set, test_set = _obtain_train_test(dataset, data_prop)

    # This is the summary table with statistics about each epoch
    summary = pd.DataFrame(
        columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    )

    # [1] Get the data loaders
    train_loader, test_loader = _get_dataloaders(train_set, test_set, batch_size)
    train_loader = prepare_data_loader(train_loader)
    test_loader = prepare_data_loader(test_loader)
    
    assert train_loader is not None and len(train_loader) > 0
    assert test_loader is not None and len(test_loader) > 0
    # [2] Wrap dataset
    model = prepare_model(model)
    
    checkpoint = ray.train.get_checkpoint()
    start = 1
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "latest.pth"))
            start = checkpoint_dict["epoch"] + 1
            model.load_state_dict(checkpoint_dict["model_state"])
    # [3] Train the model
    for epoch in range(start, epochs + 1):
        model, train_loss, train_acc = _train_step(
            model, train_loader, epoch, criterion, optimizer
        )
        test_loss, test_acc, add_metrics = _eval_step(model, test_loader, epoch, criterion,
                                         additional_metrics=config.get("additional_metrics", None))
        scheduler.step(test_loss)
        epoch_summary = pd.Series(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": test_loss,
                "val_acc": test_acc,
            }
        )
        summary = pd.concat([summary, epoch_summary.to_frame().T], ignore_index=True)
        metrics={"loss": test_loss, "accuracy": test_acc, "summary": summary.to_dict(), **add_metrics}
        ### This makes it trainable
        with tempfile.TemporaryDirectory() as tempdir:
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, 
                       os.path.join(tempdir, f"checkpoint_{epoch}.pth"))
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, 
                       os.path.join(tempdir, f"latest.pth"))
            ray.train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))
    with tempfile.TemporaryDirectory() as tempdir:
        summary.to_csv(os.path.join(tempdir, "training_summary.csv"), index=False)


class CIRI_trainer():
    def __init__(self, model: str, 
                 data_folders: list,
                 data_prop: float,
                 sample_indices: list | float = None):
        """Initialise the CIRI_trainer object.

        Args:
            model (str): a string representing the model to use. Valid values are
                the names of the models available in torchvision.models.
            data_folders (list): a list of 2 strings representing the paths to the
                base and augmented datasets on disk.
            data_prop (float): a number between 0 and 1 representing the proportion
                of the train set.
            sample_indices (list, optional): a list of indices to sample from the dataset.
                Alternatively a float representing the proportion of the dataset to sample 
                (e.g. 0.1 for 10%). Defaults to None.

        Raises:
            ValueError: if model is not a string
        """        
        if not isinstance(model, str):
            raise ValueError(
                "model must be a string. Accepted values are pre-defined models from torchvision")
        self.ds = None
        if sample_indices is not None and isinstance(sample_indices, float):
            sample_indices, ds = _sample_indices_from_prop(
                sample_indices, 
                root_base=data_folders[0], 
                root_augmented=data_folders[1]
            )
            self.ds = ds
        self.config = {
            "model": model,
            "data_folders": data_folders,
            "data_prop": data_prop,
            "sample_indices": sample_indices
        }
    
    def get_trainer(self, run_name: str, config=None, as_trainable=False):
        if config is None:
            config = {}
        run_config = RunConfig(
            name=run_name
        )
        if torch.cuda.is_available():
            scaling_config = ScalingConfig(
                num_workers=1,
                use_gpu=True
            )
        else:
            scaling_config = ScalingConfig(
                num_workers=os.cpu_count() // 2,
                use_gpu=False,
                resources_per_worker={'CPU': 1}
            )
        trainer = TorchTrainer(
            train_loop_per_worker=_train_func_per_worker,
            run_config=run_config,
            scaling_config=scaling_config,
            train_loop_config={
                **self.config,
                **config
            }
        )
        if as_trainable:
            return trainer.as_trainable()
        return trainer
    
    def train(self, run_name: str, config=None,
              persist_dir=None, persist_config={
                   "whole_folder": False, # whenever possible avoid setting this to true, it's very big
                    "checkpoint": True,
                    "results": True,
                    "params": True,
                    "progress": True
              }):
        trainer = self.get_trainer(run_name, config)
        result = trainer.fit()
        if persist_dir is not None and persist_config:
            _persist_cv_results(result, persist_dir, run_name, persist_config)
        return result
    
    def _get_tuner(self, run_name: str, param_space=None, scheduler=None, num_samples=10):
        if param_space is None:
            raise ValueError("param_space is None. Please provide a dictionary of hyperparameters to tune.")
        trainable_ciri = self.get_trainer(run_name, as_trainable=True)
        if scheduler is None or not isinstance(scheduler, TrialScheduler):
            print("Defaulting to ASHA scheduler (no scheduler provided or not an instance of TrialScheduler)")
            scheduler = ASHAScheduler(
                metric="accuracy",
                mode="max"
            )
        tune_config = TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
            reuse_actors=True
        )
        tuner = Tuner(
            trainable=trainable_ciri,
            param_space={
                "train_loop_config": param_space
            },
            tune_config=tune_config,
            run_config=RunConfig(
                name=run_name
            )
        )
        return tuner, trainable_ciri
    
    def tune_hyperparams(self, run_name: str, param_space=None, scheduler=None, 
                         num_samples=10, restore=True, restore_path=None): 
        tuner, trainable = self._get_tuner(run_name, param_space, scheduler, num_samples)
        if restore_path is None:
            restore_path = os.path.join(
            os.path.expanduser("~/ray_results"),
            run_name)
        exp_files = glob.glob(f"{restore_path}/experiment_state-*.json")
        if restore and tuner.can_restore(restore_path) and len(exp_files) > 0:
            print(f"Restoring tuner from path {restore_path}")
            tuner = ray.tune.Tuner.restore(restore_path, 
                                        trainable=trainable,
                                        restart_errored=True)   
        return tuner.fit()
    
    def cross_validate(self, run_name: str, 
                       config=None, 
                       outer_cv_k=5, 
                       inner_cv_k=0,
                       tune_hyperparams=False,
                       results_persist_dir=None,
                       persist_config={
                           # Choose what to persist when doing hyperparameter tuning
                           "hpt": {
                               "whole_folder": False, # whenever possible avoid setting this to true, it's very big
                               "checkpoint": False,
                               "results": True,
                               "params": True,
                               "progress": True,
                               "experiments_summary": True
                            },
                           # Choose what to persist when not doing hyperparameter tuning
                           "no_hpt": {
                               "whole_folder": False, # whenever possible avoid setting this to true, it's very big
                               "checkpoint": True,
                               "results": True,
                               "params": True,
                               "progress": True
                            }
                       }, 
                       start_fold=(0,0),
                       *args, **kwargs):
        if config is None:
            config = {}
        if self.ds is None:
            self.ds = _load_ciri_dataset(
                self.config["data_folders"][0], 
                self.config["data_folders"][1], 
                None
            )
        if not tune_hyperparams:
            inner_cv_k = 0
        cv_generator = _get_fold_indices(self.ds.info, 
                                         sample_idx=self.config.get("sample_indices", None),
                                         outer_cv_k=outer_cv_k,
                                         inner_cv_k=inner_cv_k)
        all_results = defaultdict(dict)
        for i, out_idx in enumerate(cv_generator):
            if i < start_fold[0]:
                continue
            if inner_cv_k > 0:
                ## Inner CV exists only if tune hyperparams
                _, _, inner_cv_generator = out_idx
                for j, inner_idx in enumerate(inner_cv_generator):
                    if i < start_fold[0] and j < start_fold[1]:
                        continue
                    print(f"Outer fold {i}, inner fold {j} - number of samples: {len(inner_idx[0])}")
                    config["train_test_idx"] = inner_idx
                    mod_run_name = f"{run_name}_outer_{i}_inner_{j}"
                    if tune_hyperparams:
                        print(f"Tuning hyperparameters for {mod_run_name}...")
                        results = self.tune_hyperparams(mod_run_name, config, *args, **kwargs)
                        if results_persist_dir is not None:
                            _persist_hp_results(results, results_persist_dir,
                                                mod_run_name, persist_config["hpt"])
                    all_results[i][j] = results
            else:
                print(f"Outer fold {i} - number of samples: {len(out_idx[0])}")
                config["train_test_idx"] = out_idx
                mod_run_name = f"{run_name}_outer_{i}"
                if tune_hyperparams:
                    results = self.tune_hyperparams(mod_run_name, config, *args, **kwargs)
                    if results_persist_dir is not None:
                        _persist_hp_results(results, results_persist_dir,
                                            mod_run_name, persist_config["hpt"])
                else:
                    results = self.train(mod_run_name, config)
                    _persist_cv_results(results, results_persist_dir,
                                        mod_run_name, persist_config["no_hpt"])
                all_results[i] = results  
        return all_results


def _persist_hp_results(results_grid, persist_dir, run_name, persist_config):
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    if persist_config["experiments_summary"]:
        summary = results_grid.get_dataframe()
        summary.to_csv(os.path.join(persist_dir, f"hpt_{run_name}_exp_summary.csv"), index=False)
    best_trial = results_grid.get_best_result(metric='accuracy', mode='max')
    train_summary = pd.DataFrame.from_dict(best_trial.metrics['summary'])
    train_summary.to_csv(os.path.join(persist_dir, f"hpt_{run_name}_train_summary.csv"), index=False)
    if persist_config["whole_folder"]:
        zip_and_copy(best_trial.path, persist_dir, f"{run_name}_best_trial.zip")
        # shutil.copytree(best_trial.path,
        #                 os.path.join(persist_dir, run_name))
        return
    
    run_dir = os.path.join(persist_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    if persist_config["checkpoint"]:
        best_checkpoint = best_trial.checkpoint.path
        shutil.copytree(best_checkpoint,
                        os.path.join(run_dir, "checkpoint"))
    if persist_config["results"]:
        shutil.copy(os.path.join(best_trial.path, "result.json"),
                    os.path.join(run_dir, "result.json"))
    if persist_config["params"]:
        shutil.copy(os.path.join(best_trial.path, "params.json"),
                    os.path.join(run_dir, "params.json"))
    if persist_config["progress"]:
        shutil.copy(os.path.join(best_trial.path, "progress.csv"),
                    os.path.join(run_dir, "progress.csv"))
        
def _persist_cv_results(result, persist_dir, run_name, persist_config):
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    train_summary = pd.DataFrame.from_dict(result.metrics['summary'])
    train_summary.to_csv(os.path.join(persist_dir, f"{run_name}_train_summary.csv"), index=False)
    if persist_config["whole_folder"]:
        zip_and_copy(result.path, persist_dir, f"{run_name}.zip")
        return
    
    run_dir = os.path.join(persist_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    if persist_config["results"]:
        shutil.copy(os.path.join(result.path, "result.json"),
                    os.path.join(run_dir, "result.json"))
    if persist_config["params"]:
        shutil.copy(os.path.join(result.path, "params.json"),
                    os.path.join(run_dir, "params.json"))
    if persist_config["progress"]:
        shutil.copy(os.path.join(result.path, "progress.csv"),
                    os.path.join(run_dir, "progress.csv"))
    if persist_config["checkpoint"]:
        best_checkpoint = result.get_best_checkpoint(metric='accuracy', mode='max')
        shutil.copytree(best_checkpoint.path,
                        os.path.join(run_dir, "checkpoint"))
    
    

def zip_and_copy(source_folder, destination_folder, zip_name="archive.zip"):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Path for the output zip file
    zip_path = os.path.join(destination_folder, zip_name)

    # Creating the zip file
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                # Create complete filepath of file in directory
                file_path = os.path.join(root, file)
                # Add file to zip
                zipf.write(file_path, os.path.relpath(file_path, source_folder))

    print(f"Files zipped and saved from {source_folder} to {zip_path}")
 
   
def unzip_and_extract(source_zip, destination_folder):
    """
    Unzips a zip file and extracts its contents into a specified destination folder.
    
    Args:
        source_zip (str): The path to the zip file.
        destination_folder (str): The directory to which the contents will be extracted.
    """
    # Ensure the destination directory exists
    os.makedirs(destination_folder, exist_ok=True)

    # Open the zip file
    with zipfile.ZipFile(source_zip, 'r') as zip_ref:
        # Extract all the contents into the destination directory
        zip_ref.extractall(destination_folder)
        print(f"Extracted all contents of {source_zip} to {destination_folder}")