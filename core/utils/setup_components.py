# general

import os
try:
    import wandb # type: ignore
    WANDB_ERROR = False
except:
    WANDB_ERROR = True
import uuid
import json

# torch
import torch # type: ignore
from torch.utils.data import DataLoader # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR # type: ignore

# internal
from core.utils.loss import InfoNCE, GOT
from core.models.Model import MADELEINE
from core.utils.file_utils import print_network
from core.datasets.dataset import SlideDataset, collate
from core.utils.process_args import get_args
from core.datasets.modalities import modality_dicts


# global magic numbers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HE_POSITION = 0 # HE slide is always the first one 
WHOLE_VIEW_POSITION = 0


def build_downstream_datasets(args):
    """
    Build downstream datasets for validation.

    Args:
        args: An object containing the arguments for building the datasets.

    Returns:
        val_datasets: A dictionary containing the downstream datasets for validation.
    """
    
    # BCNB
    bcnb_dataset = SlideDataset(
        dataset_name = "BCNB",
        csv_path= "../dataset_csv/BCNB/BCNB.csv",
        features_path="../data/downstream/BCNB/feats_h5",
        modalities=["HE"],
        train=False
    )

    val_datasets = {
        "BCNB": bcnb_dataset,
    }
    
    return val_datasets

def set_up_logging(args, RESULS_SAVE_PATH):
    """
    Sets up logging using wandb.

    Args:
        args (argparse.Namespace): The command-line arguments.
        RESULS_SAVE_PATH (str): The path to save the results.

    Returns:
        None
    """
    print("* Setup logging...", end="")
    wandb.init(
        project=args.wandb_project_name,
        name=args.EXP_CODE,
        tags=[args.cohort],
        id=str(uuid.uuid4()),
        config=args,
    )

    file = open(os.path.join(RESULS_SAVE_PATH, "wandbID.txt"), "w")
    file.write(wandb.run.id)
    file.close()
    print("\033[92m Done \033[0m")
    

def setup():
    """
    Set up the experiment configuration and save necessary files.
    
    Returns:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    
    # setup args
    args = get_args()
    
    # create the results save path
    print("* Creating results save path...", end="")
    os.makedirs(args.RESULS_SAVE_PATH, exist_ok=True)
    print("\033[92m Done \033[0m")
    
    # set up logging 
    if args.log_ml and not WANDB_ERROR:
        set_up_logging(args, args.RESULS_SAVE_PATH)
        
    # all modalities
    global MODALITIES
    MODALITIES = modality_dicts[args.dataset] 
    
    # stains to contrast HE with
    global STAINS
    STAINS = MODALITIES.copy() 
    STAINS.pop(HE_POSITION)
    
    # add to args
    args.MODALITIES = MODALITIES
    args.STAINS = STAINS
    
    # save the config to the results path
    print("* Saving experiment config...", end="")
    with open(os.path.join(args.RESULS_SAVE_PATH, "config.json"), 'w') as handle:
        json.dump(vars(args), handle, indent=4)
    print("\033[92m Done \033[0m")
    
    return args


def setup_dataset(args):
    
    print("* Setup dataset...", end="")
    dataset = SlideDataset(
        dataset_name=args.dataset,
        csv_path=args.csv_fpath, 
        features_path=args.data_root_dir,
        sample=args.n_subsamples,
        modalities=args.MODALITIES,
        embedding_size=args.patch_embedding_dim,
    )
    print("\033[92m Done \033[0m")
    return dataset

def setup_dataloader(args, dataset):
    print("* Setup train dataloader...", end="")
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate,
        num_workers=0,
    )
    print("\033[92m Done \033[0m")
    return dataloader

def setup_DownstreamDatasets(args):
    print("* Setup downstream datasets...", end="")
    val_datasets = build_downstream_datasets(args)
    print("\033[92m Done \033[0m")
    
    print("* Setup downstream dataloaders...", end="")
    val_dataloaders = {}
    for key in val_datasets:
        curr_dataloader = DataLoader(
            val_datasets[key], 
            batch_size=1, 
            shuffle=False, 
            collate_fn=collate,
            num_workers=4,
        )
        val_dataloaders[key] = curr_dataloader
    print("\033[92m Done \033[0m")
    
    
    return val_dataloaders

def setup_model(args):
    
    print("* Setup model...", end="")
    # init model
    ssl_model = MADELEINE(
        config=args,
        stain_encoding=args.add_stain_encoding,
    )

    # put model on gpu(s)
    if args.num_gpus > 1:
        ssl_model = nn.DataParallel(ssl_model, device_ids=list(range(args.num_gpus)))
    ssl_model.to("cuda")
    
    # save model architecture
    print_network(ssl_model, results_dir=args.RESULS_SAVE_PATH)
    print("\033[92m Done \033[0m")
    return ssl_model

def setup_optim(args, dataloader, ssl_model):
    print("* Setup optimizer...", end="")
    optimizer = optim.AdamW(ssl_model.parameters(), lr=args.lr)
    print("\033[92m Done \033[0m")
    
    # set up schedulers
    print("* Setup schedulers...", end="")
    T_max = (args.max_epochs - args.warmup_epochs) * len(dataloader) if args.warmup else args.max_epochs * len(dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=args.end_learning_rate)
    
    if args.warmup:
        scheduler_warmup = LinearLR(optimizer, start_factor=0.00001, total_iters=args.warmup_epochs * len(dataloader))
    else:
        scheduler_warmup = None
    print("\033[92m Done \033[0m")
    return optimizer,scheduler,scheduler_warmup

def setup_losses(args):
        """
        Set up the loss functions based on the arguments.

        Args:
            args (argparse.Namespace): The parsed command-line arguments.

        Returns:
            loss_fn_interMod (torch.nn.Module): The inter-modality loss function.
            loss_fn_interMod_local (torch.nn.Module): The local alignment loss function.
            loss_fn_intraMod (torch.nn.Module): The intra-modality loss function.
        """
        # global loss
        print("* Setup global loss = {}...".format(args.global_loss), end="")
        if args.global_loss == "info-nce":
            loss_fn_interMod = InfoNCE(temperature=args.temperature)
        else:
            loss_fn_interMod = None
        print("\033[92m Done \033[0m")

        # local alignment
        print("* Setup local loss = {}...".format(args.local_loss), end="")
        if args.local_loss == "got":
            loss_fn_interMod_local = GOT
        else:
            loss_fn_interMod_local = None
        print("\033[92m Done \033[0m")

        # intra modality loss
        print("* Setup intra modality loss = {}...".format("None" if args.intra_modality_loss == "-1" else args.intra_modality_loss), end="")
        if args.intra_modality_loss == "info-nce":
            loss_fn_intraMod = InfoNCE(temperature=args.temperature)
        else:
            loss_fn_intraMod = None
        print("\033[92m Done \033[0m")

        return loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod