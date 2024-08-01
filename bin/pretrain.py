import os
import numpy as np
import torch
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR

import sys
sys.path.append('../')
sys.path.append('../../')

from core.utils.loss import InfoNCE, GOT
from core.models.Model import MADELEINE
from core.utils.file_utils import print_network, save_pkl
from core.datasets.dataset import SlideDataset, collate
from core.utils.process_args import get_args
from core.utils.utils import set_deterministic_mode, smooth_rank_measure, set_model_precision
from core.datasets.modalities import modality_dicts

import pdb
import wandb
import uuid
import json

# global magic numbers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HE_POSITION = 0 # HE slide is always the first one 
WHOLE_VIEW_POSITION = 0 # for intra modality loss

# move to utils
def calculate_losses(loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod, wsi_embs, token_embs, modality_labels_withoutHE, config):
    losses = []
    atleast_two_loss_flag = False

    for stain_idx, stain in enumerate(STAINS):
        stain_mask = modality_labels_withoutHE[:, stain_idx].bool()
        if stain_mask.sum().item() > 1:
            # Global loss:
            if loss_fn_interMod:
                HE_for_stain = wsi_embs["HE"][:, WHOLE_VIEW_POSITION, :, stain_idx][stain_mask]
                stain_ind = wsi_embs[stain][:, WHOLE_VIEW_POSITION, :][stain_mask]

                if config.global_loss == "info-nce":
                    global_loss = loss_fn_interMod(query=HE_for_stain, positive_key=stain_ind, symmetric=config.symmetric_cl)
                else:
                    raise AssertionError("invalid global loss")

                # add to loss 
                losses.append(global_loss) 

            # Local loss:
            if loss_fn_interMod_local:
                HE_tokens = token_embs["HE"][:, :, :, stain_idx][stain_mask]
                IHC_tokens = token_embs[stain].squeeze()[stain_mask]
                got_loss = loss_fn_interMod_local(HE_tokens, IHC_tokens, subsample=256)
                got_loss = got_loss * config.local_loss_weight

                # add to loss 
                losses.append(got_loss)

            # Intra modality loss
            if loss_fn_intraMod:
                # view 1 
                HE_for_stain_view1 = wsi_embs["HE"][:, 1, :, stain_idx][stain_mask]
                stain_ind_view1 = wsi_embs[stain][:, 1, :][stain_mask]

                # view 2
                HE_for_stain_view2 = wsi_embs["HE"][:, 2, :, stain_idx][stain_mask]
                stain_ind_view2 = wsi_embs[stain][:, 2, :][stain_mask]

                # intra modal loss for HE and stain
                l_HE = loss_fn_intraMod(query=HE_for_stain_view1, positive_key=HE_for_stain_view2, symmetric=config.symmetric_cl)
                l_stain = loss_fn_intraMod(query=stain_ind_view1, positive_key=stain_ind_view2, symmetric=config.symmetric_cl)

                # add to loss 
                losses.append(l_HE)
                losses.append(l_stain)

            # there is at least one stain in addition to HE in this batch, so we keep this batch
            atleast_two_loss_flag = True
            
    if len(losses) > 0:
        loss = sum(losses)
    else:
        loss = -1
        assert loss == -1 and not atleast_two_loss_flag, "Loss should be -1 if there are no losses to calculate"
        
    return loss, atleast_two_loss_flag

# move to utils
def train_loop(config, loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler):

    if loss_fn_intraMod:
        n_views = 3
    else:
        n_views = 1
        
    ssl_model.train()
    ssl_model.to(DEVICE)
    ssl_model, torch_precision = set_model_precision(ssl_model, args.precision)

    ep_loss = 0.
    fb_time = 0.
    all_embeds = []
    
    for b_idx, data in enumerate(dataloader):
        
        if epoch == 0 and b_idx == 0:
            print("Using precision:", torch_precision)
        
        s_fb = time.time()
        
        # set data on device and dtype
        data['feats'] = data['feats'].to(DEVICE).to(torch_precision)
        data['modality_labels'] = data['modality_labels'].to(torch_precision)
        
        # clean modality labels to be without HE
        modality_labels = data['modality_labels']
        modality_labels_withoutHE = modality_labels[:, HE_POSITION+1:]
        
        # begin forward pass
        optimizer.zero_grad()
             
        # get model outputs
        wsi_embs, token_embs = ssl_model(data, device=DEVICE, n_views=n_views)
        
        # calculate losses
        loss, atleast_two_loss_flag = calculate_losses(loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod, wsi_embs, token_embs, modality_labels_withoutHE, config)
            
        # get the train embeds to calculate rank
        all_embeds.extend(wsi_embs['HE'][:, WHOLE_VIEW_POSITION, :, 0].detach().to(torch.float32).cpu().numpy())
        
        # if we have a batch with only HE then continue
        if not atleast_two_loss_flag:
            print("Skipping batch with only HE")
            continue
        
        # if we have made it, then we must have had more than HE stain, so we update model
        loss.backward()
        optimizer.step()

        if epoch <= config.warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler.step()  
            
        if (b_idx % 3) == 0:
            print(f"Loss for batch: {b_idx} = {loss:.3f}")
            break
            
        ep_loss += loss.item()
        
        e_fb = time.time()
        fb_time += e_fb - s_fb
        
    # track rank on all HE slides
    all_embeds_tensor = torch.Tensor(np.array(all_embeds))
    rank = smooth_rank_measure(all_embeds_tensor)  
        
    return ep_loss, rank

# move to utils
def val_loop(config, ssl_model, val_dataloader):
    """
    Perform validation loop for the SSL model.

    Args:
        config (object): Configuration object containing model settings.
        ssl_model (object): SSL model to be evaluated.
        val_dataloader (object): Dataloader for validation dataset.

    Returns:
        tuple: A tuple containing the results dictionary and the rank measure.
            - results_dict (dict): Dictionary containing the embeddings and slide IDs.
            - rank (float): Rank measure calculated from the embeddings.
    """

    # set model to eval 
    ssl_model.eval()
    ssl_model.to(DEVICE)
    ssl_model, torch_precision = set_model_precision(ssl_model, config.precision)
    
    all_embeds = []
    all_slide_ids = []
    
    # do everything without grads 
    with torch.no_grad():
        for data in tqdm(val_dataloader):
        
            # unpack data and process
            data['feats'] = data['feats'].to(DEVICE).to(torch_precision)
            
            # forward
            wsi_embed = ssl_model(data=data, device=DEVICE, train=False)
            wsi_embed = wsi_embed['HE']
            
            all_embeds.extend(wsi_embed.squeeze(dim=1).to(torch.float32).detach().cpu().numpy())
            all_slide_ids.append(data['slide_ids'][0])
            
    all_embeds = np.array(all_embeds)
    all_embeds_tensor = torch.Tensor(all_embeds)
    rank = smooth_rank_measure(all_embeds_tensor)  
    results_dict = {"embeds": all_embeds, 'slide_ids': all_slide_ids}
    
    return results_dict, rank

# move to utils
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
        embedding_size=args.patch_embedding_dim,
        train=False
    )

    val_datasets = {
        "BCNB": bcnb_dataset,
    }
    
    return val_datasets

# move to utils
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
    
    # save the config to the results path
    print("* Saving experiment config...", end="")
    with open(os.path.join(args.RESULS_SAVE_PATH, "config.json"), 'w') as handle:
        json.dump(vars(args), handle, indent=4)
    print("\033[92m Done \033[0m")
    
    # set up logging 
    if args.log_ml:
        set_up_logging(args, args.RESULS_SAVE_PATH)
        
    # all modalities
    global MODALITIES
    MODALITIES = modality_dicts[args.dataset] 
    
    # stains to contrast HE with
    global STAINS
    STAINS = MODALITIES.copy() 
    STAINS.pop(HE_POSITION)
    
    return args


def setup_dataset(args, MODALITIES):
    
    print("* Setup dataset...", end="")
    dataset = SlideDataset(
        dataset_name=args.dataset,
        csv_path=args.csv_fpath, 
        features_path=args.data_root_dir,
        sample=args.n_subsamples,
        modalities=MODALITIES,
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

def setup_DownstreamDatasets(build_downstream_datasets, args):
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

def setup_model(DEVICE, args, MODALITIES):
    
    print("* Setup model...", end="")
    # init model
    ssl_model = MADELEINE(
        config=args,
        modalities=MODALITIES,
        stain_encoding=args.add_stain_encoding,
    ).to(DEVICE)

    # put model on gpu(s)
    if args.num_gpus > 1:
        ssl_model = nn.DataParallel(ssl_model, device_ids=list(range(args.num_gpus)))
    ssl_model.to("cuda:0")
    
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

def extract_slide_level_embeddings(args, val_dataloaders, ssl_model):
    for dataset_name in val_dataloaders:
        print(f"\n* Extracting slide-level embeddings of {dataset_name}")
        curr_loader = val_dataloaders[dataset_name]
        curr_results_dict, curr_val_rank = val_loop(args, ssl_model, curr_loader)
        print("Rank for {} = {}".format(dataset_name, curr_val_rank))
        print("\033[92mDone \033[0m")
        
        if args.log_ml:
            wandb.run.summary["{}_rank".format(dataset_name)] = curr_val_rank
            
        save_pkl(os.path.join(args.RESULS_SAVE_PATH, f"{dataset_name}.pkl"), curr_results_dict)

def load_checkpoint(args, ssl_model):
    state_dict = torch.load(os.path.join(args.RESULS_SAVE_PATH, "model.pt"))
    try:
        ssl_model.load_state_dict(state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        ssl_model.load_state_dict(new_state_dict)
        print('Model loaded by removing module in state dict...')

if __name__ == "__main__":
    
    # set seed
    set_deterministic_mode(SEED=42)
    
    # do setup 
    args = setup()
    
    # setup dataset
    dataset = setup_dataset(args)
    
    # set up dataloader
    dataloader = setup_dataloader(args, dataset)
    
    # set up the downstream datasets
    val_dataloaders = setup_DownstreamDatasets(args)
    
    ssl_model = setup_model(args)
    
    # set up optimizers
    optimizer, scheduler, scheduler_warmup = setup_optim(args, dataloader, ssl_model)
    
    # set up losses
    loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod = setup_losses(args)

    # main training loop
    best_rank = 0.
    for epoch in range(args.max_epochs):
        
        print(f"\nTraining for epoch {epoch}...\n")
        
        # train
        start = time.time()
        ep_loss, train_rank = train_loop(args, loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler)
        
        if args.log_ml:
            wandb.log({"train_loss": ep_loss, "train_rank": train_rank})

        end = time.time()

        print(f"\nDone with epoch {epoch}")
        print(f"Total loss = {ep_loss:.3f}")
        print(f"Train rank = {train_rank:.3f}")
        print(f"Total time = {end-start:.3f} seconds")

        # Stop training based on rank of the training samples. 
        # dont save for the first 20 epochs
        if epoch > 20 and train_rank > best_rank:
            print('Better rank: {} --> {}. Saving model'.format(best_rank, train_rank))
            best_rank = train_rank
            torch.save(ssl_model.state_dict(), os.path.join(args.RESULS_SAVE_PATH, "model.pt"))
            print()
        
        # HACK
        torch.save(ssl_model.state_dict(), os.path.join(args.RESULS_SAVE_PATH, "model.pt"))
        break
    
    print("\nDone with training\n")

    # load the wsi_embedder model
    load_checkpoint(args, ssl_model)
        
    # get slide-level embeddings of downstream datasets
    extract_slide_level_embeddings(val_loop, args, val_dataloaders, ssl_model)
    
    print()
    print(100*"-")
    print("End of experiment, bye!")
    print(100*"-")
    print()
    