import torch
from torch import allclose, argmax, autograd, exp, linspace, nn, sigmoid, sqrt
import torch.nn.functional as F
import numpy as np

vocab_size=256
####__________________________
#Linear Noise Implementation
###____________________________

class LinearGammaSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        

    def forward(self, t):
        """
        Args:
            t: [B] or [B, 1] tensor containing times in [0,1]
        Returns:
            gamma_t with shape [B]
        """
        return self.gamma_min + t * (self.gamma_max - self.gamma_min)



###_________________
# Our VDM
##___________________


class VDM(nn.Module):
    def __init__(self, model, image_shape, gamma_min, gamma_max):
        super().__init__()
        self.model = model
        self.image_shape = image_shape
         # Store them directly for easy access
        # self.gamma_min = gamma_min
        # self.gamma_max = gamma_max
        device = next(model.parameters()).device
        self.gamma_min = torch.as_tensor(gamma_min, dtype=torch.float32, device=device)
        self.gamma_max = torch.as_tensor(gamma_max, dtype=torch.float32, device=device)

        # self.gamma_min = torch.tensor(gamma_min)
        # self.gamma_max = torch.tensor(gamma_max)

        self.gamma = LinearGammaSchedule(gamma_min, gamma_max)

    #helper function to see what is going on on the device
    @property
    def device(self):
        return next(self.model.parameters()).device


###______________
#q(z_t|t)
#####

    def sample_q_t_0(self, x, times, noise=None): #forward diffusion (in the paper we use z_t instead of x_t but during the forward pass they are equivalent)
        """Samples from the distributions q(x_t | x_0) at the given time steps.
        
        Args:
            x: clean image (x0)
            times: [B] float32 times in [0,1]
        
        Returns:
            A tuple (z_t, gamma_t) where:
                z_t: noisy image at time t
                gamma_t: gamma value at time t
        """
        with torch.enable_grad():  # Need gradient to compute loss even when evaluating
            gamma_t = self.gamma(times) 

        # Pad gamma to match image shape (B, C, H, W)
        gamma_t = gamma_t[:, None, None, None]
        
        # compute alpha and sigma from gamma
        alpha = torch.sqrt(torch.sigmoid(-gamma_t))       # √(sigmoid(-γ))
        sigma = torch.sqrt(torch.sigmoid(gamma_t))        # √(sigmoid(γ))

        if noise is None:
            #raise Warning("No Noise is applied")
            noise = torch.randn_like(x)

        return alpha * x + sigma * noise, gamma_t


    ########_______________
    #sampling t
    #######################
    def sample_times(self, batch_size):
        """
        Args:
            batch_size: int of number of times to sample
        Returns:
            a tensor of size batch_size of random times in [0,1]
        """
        times = torch.rand(batch_size, device=self.device, requires_grad=True)
        return times


    # ---------------------------
    # Data decoding
    # ---------------------------
    def data_decode(self, z_0_rescaled, gamma_0):
        """
        Compute log p(x | z_0) for discrete data.
        Uses discretized Gaussian likelihood.
        
        Args:
            z_0_rescaled: [B,C,H,W] rescaled latent variable at time 0
            gamma_0: scalar tensor of gamma at time 0
        Returns:
            [B,D,vocab_size] log probabilities for each possible discrete symbol
        """
        B = z_0_rescaled.shape[0]
        D = np.prod(z_0_rescaled.shape[1:])
        z_flat = z_0_rescaled.view(B, D)
        
        # Discretized values in [-1, 1]
        # Map 0-255 → [-1, 1]
        x_vals = torch.arange(256, device=z_0_rescaled.device).float()
        x_vals = (x_vals / 127.5) - 1.0  # [0,255] → [-1,1]
        x_vals = x_vals.view(1, 1, 256)  # [1, 1, 256]
        
        # Compute bin edges for discretized Gaussian
        bin_width = 2.0 / 255  # width of each discrete bin
        x_upper = x_vals + bin_width / 2
        x_lower = x_vals - bin_width / 2
        
        # Handle boundaries
        x_upper[:, :, -1] = float('inf')
        x_lower[:, :, 0] = float('-inf')
        
        # Standard deviation from gamma
        sigma_0 = torch.sqrt(torch.sigmoid(gamma_0))
        
        # z_flat shape: [B, D], need [B, D, 1]
        z_expanded = z_flat.unsqueeze(-1)  # [B, D, 1]
        
        # Compute CDF values for discretized likelihood
        cdf_upper = torch.erf((x_upper - z_expanded) / (sigma_0 * np.sqrt(2)))
        cdf_lower = torch.erf((x_lower - z_expanded) / (sigma_0 * np.sqrt(2)))
        
        # Log probabilities (with numerical stability)
        log_probs = torch.log(torch.clamp((cdf_upper - cdf_lower) / 2, min=1e-12))
        
        return log_probs  # [B, D, 256]
    
    def data_logprob(self, x, z_0_rescaled, gamma_0):
        """
        Compute log p(x | z_0) for the actual observed x.
        Args:
            x: [B,C,H,W] discrete observed data (values in 0-255)
            z_0_rescaled: [B,C,H,W] rescaled latent variable at time 0
            gamma_0: scalar tensor of gamma at time 0

        Returns:
            log_probs: [B] log probabilities of observed x given z_0
        """
        B = x.shape[0]
        D = np.prod(x.shape[1:])
        
        # Flatten x to [B, D]
        x_flat = x.view(B, -1).long()
        
        # Get log probabilities for all values
        log_probs = self.data_decode(z_0_rescaled, gamma_0)  # [B, D, 256]
        
        # Select the log probability for observed x
        # Use gather: [B, D, 1] → [B, D]
        selected_log_probs = torch.gather(log_probs, 2, x_flat.unsqueeze(-1)).squeeze(-1)
        
        # Sum over dimensions: [B, D] → [B]
        return selected_log_probs.sum(dim=1)
    
    # def sample(self, batch_size, n_sample_steps=50, clip_samples=True):
    #     """
    #     Generate samples from the trained VDM model.
        
    #     Args:
    #         batch_size (int): number of samples to generate
    #         n_sample_steps (int): number of reverse diffusion steps
    #         clip_samples (bool): whether to clip samples to [-1,1] for visualization

    #     Returns:
    #         x: [B,C,H,W] sampled images
    #     """
    #     device = self.device
    #     B, C, H, W = batch_size, *self.image_shape
    #     # Start from standard normal noise
    #     x_t = torch.randn(batch_size, *self.image_shape, device=device)
        
    #     # Linear time steps from 1 to 0
    #     times = torch.linspace(1.0, 0.0, n_sample_steps, device=device)

    #     for t in times:
    #         t_batch = torch.full((batch_size,), t, device=device)
    #         gamma_t = self.gamma(t_batch)[:, None, None, None]

    #         # Predict noise using the model
    #         with torch.no_grad():
    #             pred_noise = self.model(x_t, gamma_t)

    #         # Compute alpha and sigma
    #         alpha = torch.sqrt(torch.sigmoid(-gamma_t))
    #         sigma = torch.sqrt(torch.sigmoid(gamma_t))

    #         # Reverse diffusion step: simple ancestral step
    #         x0_pred = (x_t - sigma * pred_noise) / alpha
    #         x_t = alpha * x0_pred + sigma * pred_noise  # update x_t

    #         if clip_samples:
    #             x_t = x_t.clamp(-1, 1)

    #     return x_t

    def sample(self, batch_size, n_sample_steps=250, clip_samples=True):
        device = self.device
        x_t = torch.randn(batch_size, *self.image_shape, device=device)

        times = torch.linspace(1.0, 0.0, n_sample_steps + 1, device=device)

        for i in range(n_sample_steps):
            t = times[i]
            t_next = times[i + 1]

            t_batch = torch.full((batch_size,), t, device=device)
            t_next_batch = torch.full((batch_size,), t_next, device=device)

            gamma_t = self.gamma(t_batch)[:, None, None, None]
            gamma_next = self.gamma(t_next_batch)[:, None, None, None]

            alpha_t = torch.sqrt(torch.sigmoid(-gamma_t))
            sigma_t = torch.sqrt(torch.sigmoid(gamma_t))

            alpha_next = torch.sqrt(torch.sigmoid(-gamma_next))
            sigma_next = torch.sqrt(torch.sigmoid(gamma_next))

            with torch.no_grad():
                eps = self.model(x_t, gamma_t)

            # Predict x0
            x0_pred = (x_t - sigma_t * eps) / alpha_t

            # DDIM update (η = 0)
            x_t = alpha_next * x0_pred + sigma_next * eps

            if clip_samples:
                x_t = x_t.clamp(-1, 1)

        return x_t




    def forward(self, x, *, noise=None):
        bpd_factor = 1 / (np.prod(x.shape[1:]) * np.log(2)) #converts the loss into bits-per-dimension.This is standard in generative modeling to report likelihood per pixel in bits (Appendix C in VDM paper).
        
        #Ensure the input is from -1 to 1
        x1= x * 2 - 1

        # Sample from q(x_t | x_0) with random t.
        times = self.sample_times(x.shape[0]).requires_grad_(True)
        if noise is None:
            noise = torch.randn_like(x)
        x_t, gamma_t = self.sample_q_t_0(x=x1, times=times, noise=noise) #eq 1,3,4 in the paper. still have a question regarding th SNR #this line is sampling from q(x_t|x_0) which is the encoder distribution
        model_pred = self.model(x_t, gamma_t)

        #Diffusion loss
        # Simple MSE loss weighted by the noise schedule derivative
        mse_loss = F.mse_loss(model_pred, noise, reduction='none')
        mse_loss = mse_loss.sum(dim=(1, 2, 3))  # Sum over spatial dims
        
        # Get dγ/dt analytically from your noise schedule
        gamma_grad = self.gamma_max -self.gamma_min
        
        # When using variance minimsation, compute dγ/dt via autograd
        # gamma_grad = autograd.grad( 
        #     gamma_t,  
        #     times, 
        #     grad_outputs=torch.ones_like(gamma_t),
        #     create_graph=True,
        #     retain_graph=True,
        # )[0]
        
        # Final diffusion loss
        diffusion_loss = 0.5 * mse_loss * gamma_grad * bpd_factor

        # Latent loss (bpd): KL divergence from q(z_1 | x) to N(0, 1)
        gamma_1 = self.gamma(torch.ones(1, device=x.device))  # gamma at t=1
        sigma_1_sq = torch.sigmoid(gamma_1)
        alpha_1_sq = 1 - sigma_1_sq  # alpha^2 = sigmoid(-gamma) = 1 - sigmoid(gamma)
        
        # Mean of q(z_1 | x) is alpha_1 * x, variance is sigma_1^2
        mean_sq = alpha_1_sq * (x1 ** 2)  # shape: (B, C, H, W)
        
        kl_std =0.5 * (sigma_1_sq+ mean_sq - 1 - torch.log(sigma_1_sq))

        latent_loss = kl_std.sum(dim=(1, 2, 3)) * bpd_factor #need it to be 1,2,3 so that we can skip the batch size

        # Reconstruction loss (bpd): - E_{q(z_0 | x)} [log p(x | z_0)]. equation 11
        # Compute log p(x | z_0) for all possible values of each pixel in x.
        gamma_0 = self.gamma_min
        alpha_0 = torch.sqrt(torch.sigmoid(-gamma_0))
        
        # Use the mean of q(z_0 | x) as the reconstruction point
        z_0_mean = alpha_0 * x1
        
        # Discrete likelihood
        x_discrete = (x * 255).round().clamp(0, 255).long()
        loss_recon = -self.data_logprob(x_discrete, z_0_mean, gamma_0) * bpd_factor
        

        # Overall loss in bpd. Shape (B, ).
        loss = diffusion_loss + latent_loss + loss_recon #this is the VLB the latent loss is the prior loss
        
        with torch.no_grad():
            gamma_0 = self.gamma(torch.tensor([0.0], device=self.device))
        metrics = {
            "diff_loss": diffusion_loss.mean(),
            "latent_loss": latent_loss.mean(),
            "loss_recon": loss_recon.mean(),
            "gamma_0": gamma_0.item(),
            "gamma_1": gamma_1.item(),
        }
        return loss.mean(), metrics

#TODO: the code below is an edited leftover from the cookiecutter file structure of the model. 
# #in reality this is only needed if we want this file to be runable 

if __name__ == "__main__":
    model = VDM()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")
