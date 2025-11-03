import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from typing import Tuple


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """
    Round with Straight-Through Estimator (STE).
    """
    z_hat = torch.round(z)
    return z + (z_hat - z).detach()


class FSQ(nn.Module):
    """
    Finite Scalar Quantization (FSQ) Module
    
    This is a PyTorch implementation of the FSQ method from: https://arxiv.org/abs/2309.15505
    """
    def __init__(self, levels: list[int]):
        super().__init__()
        # TODO: check dtypes
        # [d]
        self.levels = torch.tensor(levels, dtype=torch.float32)
        self.d = len(levels) # Number of dimensions
        
        # [d], e.g., [1, L1, L1*L2, ...]
        basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0)
        self.register_buffer('basis', basis.to(torch.int64))
        
        self.codebook_size = np.prod(levels)
        
        # Pre-calculate for bound function
        self.register_buffer('_levels_np', torch.tensor(levels, dtype=torch.float32))
        self.register_buffer('half_width', self._levels_np // 2)
        
        eps = 1e-3
        # [d]
        half_l = (self._levels_np - 1) * (1 - eps) / 2
        # [d]
        offset = torch.where(self._levels_np % 2 == 1, 0.0, 0.5)
        # [d]
        shift = torch.tan(offset / half_l)
        
        self.register_buffer('half_l', half_l)
        self.register_buffer('offset', offset)
        self.register_buffer('shift', shift)

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """
        Applies the bounding function f(z) before rounding.
        """
        # This function is a bit complex, but it's a general
        # way to map z to a range that, when rounded,
        # produces L distinct integer values.
        # A simpler version is f:z -> floor(L/2) * tanh(z)
        return torch.tanh(z + self.shift) * self.half_l - self.offset

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Quantizes z, returns the quantized z_hat (normalized).
        """
        # 1. Bound the input
        z_bounded = self.bound(z)
        
        # 2. Round with STE
        z_hat_integers = round_ste(z_bounded)
        
        # 3. Renormalize to [-1, 1] range for the decoder
        z_hat_normalized = z_hat_integers / self.half_width
        
        return z_hat_normalized

    def _scale_and_shift(self, z_hat_normalized: torch.Tensor) -> torch.Tensor:
        """Helper to convert normalized codes to {0, 1, ..., L-1} indices."""
        return (z_hat_normalized * self.half_width) + self.half_width

    def _scale_and_shift_inverse(self, z_hat_indices: torch.Tensor) -> torch.Tensor:
        """Helper to convert {0, 1, ..., L-1} indices to normalized codes."""
        return (z_hat_indices - self.half_width) / self.half_width

    def codes_to_indexes(self, z_hat_normalized: torch.Tensor) -> torch.Tensor:
        """
        Converts normalized quantized vectors to single integer indices.
        
        Args:
            z_hat_normalized (Tensor): Shape (..., d)
        Returns:
            indices (Tensor): Shape (...,)
        """
        # Convert from normalized e.g. [-1, 0, 1] to {0, 1, 2}
        z_hat_indices = self._scale_and_shift(z_hat_normalized)
        z_hat_indices = z_hat_indices.round().to(torch.uint32)
        
        # Project to 1D index
        return (z_hat_indices * self.basis).sum(dim=-1).to(torch.uint32)

    def indexes_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Converts single integer indices back to normalized quantized vectors.
        
        Args:
            indices (Tensor): Shape (...,)
        Returns:
            z_hat_normalized (Tensor): Shape (..., d)
        """
        indices = indices.unsqueeze(-1) # (..., 1)
        
        # Cast to int64 (Long) for floor division, as uint32 is not supported
        indices_long = indices.to(torch.int64)
        basis_long = self.basis.to(torch.int64)

        # (..., d)
        codes_non_centered = (indices_long // basis_long) % self._levels_np
        
        # Convert from {0, 1, 2} back to normalized e.g. [-1, 0, 1]
        z_hat_normalized = self._scale_and_shift_inverse(codes_non_centered)
        
        return z_hat_normalized


class MLPBlock(nn.Module):
    """A simple MLP block"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU() 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x))


class MLPEncoder(nn.Module):
    """MLP Encoder for FSQ"""
    def __init__(self, input_dim: int, hidden_dim: int, fsq_dim: int, depth: int):
        super().__init__()
                
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.proj_act = nn.GELU()

        # A stack of MLP blocks
        layers = []
        for _ in range(depth):
            layers.append(MLPBlock(hidden_dim=hidden_dim))

        self.blocks = nn.Sequential(*layers)
        
        # Final layer norm and projection head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, fsq_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Input layer
        x = self.proj_act(self.proj(x))
        
        # Pass through blocks
        x = self.blocks(x)
        
        # Normalize
        x = self.norm(x)
        
        # Project to FSQ's latent dimension 'd'
        z_e = self.head(x)
        return z_e

class MLPDecoder(nn.Module):
    """MLP Decoder for FSQ"""
    def __init__(self, fsq_dim: int, hidden_dim: int, output_dim: int, depth: int):
        super().__init__()
        
        # Project from FSQ's dim 'd' back to MLP hidden dim
        self.proj = nn.Linear(fsq_dim, hidden_dim)
        self.proj_act = nn.GELU()
        
        # A stack of MLP blocks
        layers = []
        for _ in range(depth):
            layers.append(MLPBlock(hidden_dim=hidden_dim))

        self.blocks = nn.Sequential(*layers)
        
        # Final norm and projection head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.sigm = nn.Sigmoid()

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        
        # Project back to hidden_dim
        x = self.proj_act(self.proj(z_q))
                
        # Pass through decoder
        x = self.blocks(x)
        x = self.norm(x)
                
        # Pass through decoder head        
        x = self.sigm(self.head(x))
        return x

class FSQ_VAE(nn.Module):
    """
    An MLP-based autoencoder using FSQ.
    """
    def __init__(
        self, 
        levels: list[int],
        input_dim: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        encoder_depth: int,
        decoder_depth: int,
    ):
        super().__init__()
        
        self.fsq_dim = len(levels)
        
        # MLP Encoder
        self.encoder = MLPEncoder(input_dim, encoder_hidden_dim, self.fsq_dim, encoder_depth)
        
        # FSQ Module
        self.fsq = FSQ(levels)
        
        # MLP Decoder
        self.decoder = MLPDecoder(self.fsq_dim, decoder_hidden_dim, input_dim, decoder_depth)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full pass for training.
        x shape: (B, L), normalized to [0, 1].
        """
        
        # Encode
        z_e = self.encoder(x)  # (B, d)
        
        # Quantize
        # FSQ forward applies to the last dimension
        z_q_normalized = self.fsq(z_e)  # (B, d)
        
        # Decode
        x_hat = self.decoder(z_q_normalized)  # (B, L)
                
        return x_hat, z_e, z_q_normalized

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compresses the input array into a sequence of integer indices.
        x_in shape: (B, L), normalized to [0, 1]
        """        
        z_e = self.encoder(x)    # (B, d)
        
        # Quantize (no STE, but fsq.forward doesn't use it anyway)
        z_q_normalized = self.fsq(z_e)
        
        # Get indices (this is the compressed token indices)
        indices = self.fsq.codes_to_indexes(z_q_normalized) # (B,)
        
        return indices

    @torch.no_grad()
    def decompress(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decompresses a sequence of integer indices back into an array.
        indices shape: (B,)
        """
        # (B,) -> (B, d)
        z_q_normalized = self.fsq.indexes_to_codes(indices)
        
        # Decode
        x_hat = self.decoder(z_q_normalized)  # (B, L)
        
        return x_hat


def setup_distributed():
    """
    Initializes the distributed process group. torchrun sets RANK, LOCAL_RANK, and WORLD_SIZE environment variables.
    """
    # Initialize distributed process group
    dist.init_process_group(backend="nccl")
    
    # Get distributed environment variables
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Pin the current process to a specific GPU
    torch.cuda.set_device(local_rank)
    print(f"Distributed setup: Rank {rank}/{world_size} on device {local_rank}")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()
    print("Distributed cleanup complete.")


def get_patches_column_major(data_array: np.ndarray, patch_size: Tuple[int, int]) -> np.ndarray:
    """
    Pads a 2D array and extracts patches in column-major order.

    Given a 2D (n, t) array and a patch size (p0, p1), this function:
    1. Pads the array with zeros so that n is divisible by p0 and t is divisible by p1.
    2. Extracts all (p0, p1) patches.
    3. Returns the patches as a 3D array (num_patches, p0, p1) ordered in column-major fashion.

    Args:
        data_array: The 2D input NumPy array of dtype uint8 (or any other).
        patch_size: A tuple (p0, p1) specifying the patch dimensions.

    Returns:
        A 3D NumPy array containing the extracted patches. The first
        dimension iterates through the patches in column-major order.
    """
    if data_array.ndim != 2:
        raise ValueError(f"Input array must be 2-dimensional, but got {data_array.ndim} dimensions.")
        
    p0, p1 = patch_size
    if not (p0 > 0 and p1 > 0):
        raise ValueError(f"Patch dimensions must be positive, but got ({p0}, {p1}).")

    n, t = data_array.shape

    # 1. Calculate padding
    # The (p0 - (n % p0)) % p0 formula correctly handles the case where n % p0 == 0
    pad_n = (p0 - (n % p0)) % p0
    pad_t = (p1 - (t % p1)) % p1

    # 2. Apply padding if needed
    if pad_n > 0 or pad_t > 0:
        # Pad at the bottom (0, pad_n) and at the right (0, pad_t)
        padded_array = np.pad(
            data_array,
            ((0, pad_n), (0, pad_t)),
            mode='constant',
            constant_values=0
        )
    else:
        # No padding was necessary
        padded_array = data_array
    
    # Get the new, padded dimensions
    N, T = padded_array.shape
    
    # Calculate the number of patches along each dimension
    num_patches_n = N // p0
    num_patches_t = T // p1

    # Reshape into a 4D array: (num_patches_n, p0, num_patches_t, p1)
    # This groups the data by patch row, then row-in-patch,
    # then patch-column, then col-in-patch.
    # We can think of this as a (num_patches_n, num_patches_t) grid of (p0, p1) patches
    reshaped = padded_array.reshape(num_patches_n, p0, num_patches_t, p1)

    # To get column-major order, we transpose the first and third axes.
    # Axes: (0: num_patches_n, 1: p0, 2: num_patches_t, 3: p1)
    # Transpose to: (2: num_patches_t, 0: num_patches_n, 1: p0, 3: p1)
    # This groups by patch-column, then patch-row.
    transposed = reshaped.transpose(2, 0, 1, 3)

    # Finally, reshape to flatten the patches.
    # The (num_patches_t, num_patches_n) dimensions are flattened in
    # row-major order (C order), which, due to the transpose,
    # gives us the desired column-major patch order:
    # (col 0, row 0), (col 0, row 1), ..., (col 1, row 0), ...
    patches = transposed.reshape(-1, p0, p1)

    return patches


if __name__ == "__main__":

    rank, world_size, local_rank = setup_distributed()

    # Data dimension
    BATCH_SIZE = 8
    PATCH_SIZE = (4, 16)
    INPUT_DIM = np.prod(PATCH_SIZE)

    # FSQ levels (e.g., codebook size 4096)
    levels = [7, 5, 5, 5, 5]
    d = len(levels)

    # MLP Hypeparameters
    ENCODER_HIDDEN_DIM = 4096
    DECODER_HIDDEN_DIM = 4096
    ENCODER_DEPTH = 4
    DECODER_DEPTH = 4

    # Training hyperparameters
    EPOCHS = 1

    # Create the model and wrap in DDP
    model = FSQ_VAE(
        levels=levels,
        input_dim=INPUT_DIM,
        encoder_hidden_dim=ENCODER_HIDDEN_DIM,
        decoder_hidden_dim=DECODER_HIDDEN_DIM,
        encoder_depth=ENCODER_DEPTH,
        decoder_depth=DECODER_DEPTH,
    )

    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Set up optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    # set up dataset, sampler, and loader
    ds = load_dataset("eminorhan/neural-pile-rodent", split="train")
    train_sampler = DistributedSampler(
        ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True  # Sampler handles the shuffling
    )
    train_loader = DataLoader(
        ds, 
        batch_size=1,
        sampler=train_sampler,
        shuffle=False
    )

    # ====== training loop ======
    model.train()
    optimizer.zero_grad()
    train_loss = 0.0

    print(f"[Rank {rank}] Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):

        train_sampler.set_epoch(epoch)            
        
        for i, batch in enumerate(train_loader):

            batch = np.array(batch["spike_counts"], dtype=np.uint8)
            batch = batch.squeeze(-1)
            print(f"1. Rank: {rank}; batch size: {batch.shape}")

            batch = get_patches_column_major(batch, PATCH_SIZE)
            print(f"2. Rank: {rank}; batch size: {batch.shape}")
            # batch = torch.from_numpy(batch)

            # # Move data to the correct GPU
            # data = batch.to(local_rank, non_blocking=True)
            
            # # Forward pass
            # x_reconstructed, _, _ = model(data)
            
            # # Compute loss
            # loss = loss_fn(x_reconstructed, data)
            
            # # Backward pass and optimizer step
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            # train_loss += loss.item()
            
            # if rank == 0 and i % 50 == 0:
            #     print(f"Epoch {epoch} | Batch {i}/{len(train_loader)} | Train Loss: {loss.item():.6f}")


    # # ====== compression & decompression eval (after training) ======
    # model.eval()

    # sample = data_normalized[0].unsqueeze(0) # (1, 1024, 1024)

    # # Compress
    # # Original (1, 1024, 1024) float32 array: ~4 MB
    # # Compressed (1, 1024) uint32 array: ~4 KB
    # # (num_patches = (1024*1024) / (32*32) = 1024)
    # compressed_indices = model.compress(sample)

    # # Decompress
    # decompressed_sample = model.decompress(compressed_indices)

    # print(f"Original shape: {sample.shape}")
    # print(f"Compressed shape: {compressed_indices.shape}") # (1, 1024)
    # print(f"Decompressed shape: {decompressed_sample.shape}")