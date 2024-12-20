import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from eval import eval_ptbrgestures
from data_loaders.get_data import get_dataset_loader
import utils.rotation_conversions as geometry


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(self, args, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        self.log_wandb = args.wandb
        if self.log_wandb:
            if args.dataset == 'ptbr':
                self.evaluator = eval_ptbrgestures.PTBREvaluator(args, self.model, self.diffusion)

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                            split=args.eval_split,
                                            hml_mode='eval')

            self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                                   split=args.eval_split,
                                                   hml_mode='gt')
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
            self.eval_data = {
                'test': lambda: eval_humanml.get_mdm_loader(
                    model, diffusion, args.eval_batch_size,
                    gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
                    args.eval_num_samples, scale=1.,
                )
            }
        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            check = dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
            missing_keys, unexpected_keys = self.model.load_state_dict(check, strict=False)
            assert len(unexpected_keys) == 0
            assert all([k.startswith('clip_model.') for k in missing_keys])
            #self.model.load_state_dict(
            #    dist_util.load_state_dict(
            #        resume_checkpoint, map_location=dist_util.dev()
            #    )
            #)

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):

        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')

            if self.log_wandb:
                self.model.log_train = True
                size = len(self.data)*self.batch_size
                dictlog = {'text':   np.zeros(size),
                          'vad':     np.zeros(size),
                          'seed':    np.zeros(size),
                          'timestep':np.zeros(size), 
                          'audio':   np.zeros(size),
                          'poses':   np.zeros(size)}
            for stepcount, (motion, cond) in enumerate(tqdm(self.data)):
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}  

                self.run_step(motion, cond)   

                if self.log_wandb:
                    i = self.batch_size*stepcount
                    e = i + self.batch_size
                    dictlog['text'][i:e] = self.model.batch_log['text'] if self.model.batch_log['text'] != [] else np.zeros(self.batch_size)
                    dictlog['vad'][i:e] = self.model.batch_log['vad'] if self.model.batch_log['vad'] != [] else np.zeros(self.batch_size)
                    dictlog['seed'][i:e] = self.model.batch_log['seed']
                    dictlog['timestep'][i:e] = self.model.batch_log['timestep']
                    dictlog['audio'][i:e] = self.model.batch_log['audio']
                    dictlog['poses'][i:e] = self.model.batch_log['poses']

                if self.step % self.log_interval == 0 and self.log_wandb:

                    mean_, std_ = self.model.batch_log['embs'][1], self.model.batch_log['embs'][0]
                    mean = [ [str(i), v] for i,v in enumerate(mean_)]
                    std = [ [str(i), v] for i,v in enumerate(std_)]

                    table_mean = self.log_wandb.wandb.Table(data=mean, columns=['dim', 'mean'])
                    table_std = self.log_wandb.wandb.Table(data=std, columns=['dim', 'std'])

                    mean_scatter = self.log_wandb.wandb.plot.scatter(table_mean, x='dim', y='mean', title='embs mean')
                    std_scatter = self.log_wandb.wandb.plot.scatter(table_std, x='dim', y='std', title='embs std')

                    self.log_wandb.wandb.log({'embs_mean_plot': mean_scatter, 'embs_std_plot': std_scatter})
                    for k,v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))
                            if self.log_wandb:
                                self.log_wandb.wandb.log({'loss': v, 'step': self.step+self.resume_step})

                        if k in ['step', 'samples'] or '_q' in k:
                            continue

                if self.step % self.save_interval == 0:
                    self.save()
                    
                    if self.log_wandb:
                        print('Logging epoch wandb')
                        stds = np.zeros(len(dictlog))
                        
                        for i, (k,v) in enumerate(dictlog.items()):
                            self.log_wandb.wandb.log({k+'_mean': v})
                        
                        stds = [ [str(i), np.std(v)] for i,v in enumerate(dictlog.values())]
                        table_std = self.log_wandb.wandb.Table(data=stds, columns=['dim', 'std'])
                        std_scatter = self.log_wandb.wandb.plot.scatter(table_std, x='dim', y='std', title='trn data emb std over batch')
                        self.log_wandb.wandb.log({'epoch': epoch, 'trn_data_emb_std_plot': std_scatter})

                        self.model.eval()
                        self.valwandb()
                        self.model.train()

                self.step += 1
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            #self.evaluate()

    def valwandb(self):
        assert self.log_wandb
        fgd, histfig, cross_fgd = self.evaluator.eval()
        if histfig is not None:
            self.log_wandb.wandb.log({'FGD Validation': fgd, 'Rot Vel Hist': self.log_wandb.wandb.Image(histfig)})
        if cross_fgd is not None:
            self.log_wandb.wandb.log(cross_fgd)
        self.log_wandb.wandb.log({'FGD Validation': fgd})
        

    def run_debugemb(self):
        print(f'Starting debug embedding')
        batchs = 10
        for i, (motion, cond) in enumerate(tqdm(self.data)):
            motion = motion.to(self.device)
            cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

            self.run_step(motion, cond)
            if i>= batchs:
                break
        return self.model.debug_seed,self.model.debug_text,self.model.debug_timestep,self.model.debug_audio,self.model.debug_vad,self.model.debug_poses



    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.data.dataset
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
