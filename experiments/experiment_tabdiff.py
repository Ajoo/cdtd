#%%
import os
import time
import itertools as it
import pickle

import numpy as np
import pandas as pd
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_ema import ExponentialMovingAverage

from tqdm import tqdm

from experiments.tabdiff.tabdiff.modules.main_modules import UniModMLP, Model
from experiments.tabdiff.tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion

from experiments.experiment import Experiment
from experiments.utils import set_seeds
#%%
from omegaconf import OmegaConf


config = OmegaConf.load('configs/tabdiff/default.yaml')

def _get_lr(scheduler, initial=None):
    try:
        return scheduler.get_last_lr()[0]
    except AttributeError:
        return initial

class Experiment_TabDiff(Experiment):
    def __init__(self, config, exp_path, dataset):
        super().__init__(config, exp_path, dataset)
        
        
    def make_model(self):
        d_numerical = self.data_wrangler.num_cont_features
        categories = np.asarray(self.data_wrangler.num_cats)
        
        set_seeds(self.seed, cuda_deterministic=True)
        backbone = UniModMLP(
            d_numerical,
            categories + 1, # need to add 1 for mask
            **self.config.unimodmlp_params
        )
        model = Model(backbone, **self.config.diffusion_params.edm_params)
        model = model.to(self.device)
        
        diffusion = UnifiedCtimeDiffusion(
            num_classes=categories,
            num_numerical_features=d_numerical,
            denoise_fn=model,
            y_only_model=None,
            **self.config['diffusion_params'],
            device=self.device,
        )
        diffusion.to(self.device)
        diffusion.train()
        
        num_params = sum(p.numel() for p in diffusion.parameters())
        print("the number of parameters", num_params)
        
        return diffusion
    
    def make_ema_model(self, diffusion):
        return { 
            'denoise_fn': ExponentialMovingAverage(
                diffusion._denoise_fn.parameters(), 
                decay=self.config.train.ema_decay, 
                use_num_updates=False,
            ),
            'num_schedule': ExponentialMovingAverage(
                diffusion.num_schedule.parameters(), 
                decay=self.config.train.ema_decay, 
                use_num_updates=False,
            ),
            'cat_schedule': ExponentialMovingAverage(
                diffusion.cat_schedule.parameters(), 
                decay=self.config.train.ema_decay, 
                use_num_updates=False,
            )
        }

    @staticmethod
    def _run_step(model, optimizer, x, closs_weight, dloss_weight):        
        optimizer.zero_grad()

        dloss, closs = model.mixed_loss(x)

        loss = dloss_weight * dloss + closs_weight * closs
        loss.backward()
        optimizer.step()

        return dloss, closs
    
    def compute_loss(self, model, data_iter):      # eval loss is not weighted
        curr_dloss = 0.0
        curr_closs = 0.0
        curr_count = 0
        model.eval()
        for X_cat, X_num, _ in data_iter:
            x = torch.cat([X_num, X_cat], dim=1).float().to(self.device)

            with torch.no_grad():
                batch_dloss, batch_closs = model.mixed_loss(x)
            curr_dloss += batch_dloss.item() * len(x)
            curr_closs += batch_closs.item() * len(x)
            curr_count += len(x)
        mloss = np.around(curr_dloss / curr_count, 4)
        gloss = np.around(curr_closs / curr_count, 4)
        return mloss, gloss
        
        
    def train(self, **kwargs):
        config = self.config.train
        d_numerical = self.data_wrangler.num_cont_features
        categories = self.data_wrangler.num_cats
        
        train_loader = self.data_wrangler.get_train_loader(
            config.batch_size, partition="train"
        )
        
        with open(os.path.join(self.logdir, 'data_wrangler.pkl'), 'wb') as file:
            pickle.dump(self.data_wrangler, file)
        
        # val_loader = self.data_wrangler.get_train_loader(
        #     config.batch_size, partition="val"
        # )
        
        # # filter out observation in validation set with categories not appearing in training set
        # idx_unknown_cat = (val_loader.X_cat == 9999).sum(1).to(torch.bool)
        # print(f"filter out {idx_unknown_cat.sum().item()} rows with unknown cat")
        # X_test_num = val_loader.X_cont[~idx_unknown_cat].to(self.device)
        # X_test_cat = val_loader.X_cat[~idx_unknown_cat].to(self.device)
        
        model = self.make_model()
        ema_model = self.make_ema_model(model)
    
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=config.factor, patience=config.reduce_lr_patience,
        )
        
        ### START TRAIN ###
        closs_weight, dloss_weight = config.c_lambda, config.d_lambda
        best_loss = np.inf
        best_ema_loss = np.inf

        start_time = time.time()
        
        # TODO: check config.get(max_steps_per_epoch)
        steps_per_epoch = min(len(train_loader), config.get('max_steps_per_epoch', 1000))
        
        log = []
        pbar = tqdm(range(config.num_epochs))
        for epoch in pbar:
            # Set up pbar
            
            #pbar = tqdm(it.islice(train_loader, steps_per_epoch), total=steps_per_epoch, desc=f"Epoch {epoch+1}/{config.num_epochs}")
            #pbar.set_description(f"Epoch {epoch+1}/{config.num_epochs}")
            model.train()
            
            # Compute the loss weights
            if config.closs_weight_schedule == "fixed":
                pass
            elif config.closs_weight_schedule == "anneal":
                frac_done = epoch / config.num_epochs
                closs_weight = config.c_lambda * (1 - frac_done)
            else:
                raise NotImplementedError(f"The continuous loss weight schedule {self.closs_weight_schedule} is not implemneted")

            # Training Step
            curr_dloss = 0.0
            curr_closs = 0.0
            curr_count = 0
            curr_lr = _get_lr(scheduler, config.lr)
            for X_cat, X_num, _ in it.islice(train_loader, steps_per_epoch):
                x = torch.cat([X_num, X_cat], dim=1).float().to(self.device)
                batch_dloss, batch_closs = self._run_step(model, optimizer, x, closs_weight, dloss_weight)
                curr_dloss += batch_dloss.item() * len(x)
                curr_closs += batch_closs.item() * len(x)
                curr_count += len(x)
                
            # Log training Loss
            mloss = np.around(curr_dloss / curr_count, 4)
            gloss = np.around(curr_closs / curr_count, 4)
            total_loss = mloss + gloss
            if np.isnan(gloss):
                print('Finding Nan in gaussian loss')
                break
            
            log_dict = {
                "epoch": epoch + 1,
                "lr": curr_lr,
                "closs_weight": closs_weight,
                "dloss_weight": dloss_weight,
                "loss/c_loss": gloss,
                "loss/d_loss": mloss,
                "loss/total_loss": total_loss
            }
            pbar.set_postfix(log_dict)
            
            # Log the learned noise schedules for numerical dimensions
            if d_numerical > 0:    # numerical data is not empty
                if model.num_schedule.rho().dim() == 0:   # non-learnable num schedule
                    log_dict["num_noise/rho"] = model.num_schedule.rho().item()
                else:
                    log_dict.update({f"num_noise/rho_col_{i}": value.item() for i, value in enumerate(model.num_schedule.rho())})

            # Log the learned noise schedules for categlrical dimensions
            if len(categories) > 0:    # categorical data is not empty
                if model.cat_schedule.k().dim() == 0:   # non-learnable cat schedule
                    log_dict["cat_noise/k"] = model.cat_schedule.k().item()
                else:
                    log_dict.update({f"cat_noise/k_col_{i}": value.item() for i, value in enumerate(model.cat_schedule.k())})
                    
            
            # Adjust learning rate
            scheduler.step(total_loss)

            # Save ckpt base on the best training loss
            if total_loss < best_loss and epoch >= config.num_epochs // 2:
                best_loss = total_loss
                model_path = os.path.join(self.ckpt_restore_dir, f'best_model.pt')
                if os.path.isfile(model_path):
                    os.remove(model_path)
                state_dicts = {
                    'denoise_fn': model._denoise_fn.state_dict(), 
                    'num_schedule': model.num_schedule.state_dict(), 
                    'cat_schedule': model.cat_schedule.state_dict(),
                    'current_epoch': epoch + 1,
                }
                torch.save(state_dicts, model_path)
            
                
            # Update EMA models and compute loss
            for m in ema_model.values():
                m.update()
                m.store()
                m.copy_to()
            ema_mloss, ema_gloss = self.compute_loss(model, train_loader)
            ema_total_loss = ema_mloss + ema_gloss
            
            # Save the best ema ckpt
            if ema_total_loss < best_ema_loss and epoch >= config.num_epochs // 2:
                best_ema_loss = ema_total_loss
                ema_model_path = os.path.join(self.ckpt_restore_dir, f'best_ema_model.pt')
                if os.path.isfile(ema_model_path):
                    os.remove(ema_model_path)
                state_dicts = {
                    'denoise_fn': model._denoise_fn.state_dict(), 
                    'num_schedule': model.num_schedule.state_dict(), 
                    'cat_schedule': model.cat_schedule.state_dict(),
                    'current_epoch': epoch + 1,
                }
                torch.save(state_dicts, ema_model_path)
            
            for m in ema_model.values():
                m.restore()
            log_dict.update({
                'ema_loss/c_loss': ema_gloss,
                'ema_loss/d_loss': ema_mloss,
                'ema_loss/total_loss': ema_total_loss,
            })
            
            log.append(log_dict)

        # save logs
        pd.DataFrame(log).to_csv(os.path.join(self.logdir, 'losses.csv'))

        end_time = time.time()
        print(f"Ending Trainnig Loop, totoal training time = {end_time - start_time}")
        self.save_train_time(end_time - start_time)
        
    
    @torch.inference_mode()
    def sample_tabular_data(self, num_samples, keep_nan_samples=False, **kwargs):
        self.model.eval()
        
        syn_data = self.model.sample_all(num_samples, self.config.sample.batch_size, keep_nan_samples=keep_nan_samples)
        
        if keep_nan_samples:
            num_all_zero_row = (syn_data.sum(dim=1) == 0).sum()
            if num_all_zero_row:
                print(f"The generated samples contain {num_all_zero_row} Nan instances!!!")
        
        d_numerical = self.data_wrangler.num_cont_features
        X_cont_gen, X_cat_gen, y_gen = syn_data[:, :d_numerical], syn_data[:, d_numerical:], None
        
        X_cat_gen, X_cont_gen, y_gen = self.data_wrangler.postprocess_gen_data(
            X_cat_gen.to(torch.long).numpy(),
            X_cont_gen.numpy(),
            y_gen.numpy() if y_gen is not None else None,
        )
        X_cat_gen = X_cat_gen.astype(int)

        return X_cat_gen, X_cont_gen, y_gen
        
    def save_model(self):
        return

    def load_model(self):
        state_dicts = torch.load(os.path.join(self.ckpt_restore_dir, f'best_ema_model.pt'))
        self.model = self.make_model()
        self.model._denoise_fn.load_state_dict(state_dicts['denoise_fn'])
        self.model.num_schedule.load_state_dict(state_dicts['num_schedule'])
        self.model.cat_schedule.load_state_dict(state_dicts['cat_schedule'])
        
        with open(os.path.join(self.logdir, 'data_wrangler.pkl'), 'rb') as file:
            self.data_wrangler = pickle.load(file)

