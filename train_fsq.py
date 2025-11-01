import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

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
        
        # [d]
        self.levels = torch.tensor(levels, dtype=torch.float32)
        self.d = len(levels) # Number of dimensions
        
        # [d], e.g., [1, L1, L1*L2, ...]
        basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0)
        self.register_buffer('basis', basis.to(torch.uint32))
        
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


class PatchEmbed(nn.Module):
    """
    2D spike count array to patch embedding
    
    Treats the (n, t) array as a 1-channel image and converts it into a sequence of flattened patch embeddings.
    """
    def __init__(self, img_size=(1000, 2000), patch_size=(32, 32), in_chans=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, N, T)
        x = self.proj(x)  # (B, embed_dim, n_patches, t_patches)
        # Flatten the spatial dimensions and permute
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class ViTEncoder(nn.Module):
    """Transformer Encoder for FSQ"""
    def __init__(self, num_patches: int, embed_dim: int, fsq_dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,  # Expects (B, Seq, Feat)
            activation='gelu'
        )

        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Final layer norm and projection head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, fsq_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Pass through transformer
        x = self.blocks(x)
        
        # Normalize
        x = self.norm(x)
        
        # Project to FSQ's latent dimension 'd'
        z_e = self.head(x)  # (B, num_patches, d)
        return z_e

class ViTDecoder(nn.Module):
    """Transformer Decoder for FSQ"""
    def __init__(self, num_patches: int, embed_dim: int, fsq_dim: int, grid_size: tuple[int, int], patch_size: tuple[int, int], depth: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        
        # Project from FSQ's dim 'd' back to the transformer's embed_dim
        self.in_proj = nn.Linear(fsq_dim, embed_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # Transformer blocks
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            activation='gelu'
        )
        self.blocks = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        
        # Final norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # "Un-patching" head
        # This ConvTranspose2d stitches the patches back together
        self.head = nn.Sequential(
            nn.ConvTranspose2d(
                embed_dim,
                out_channels=1,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.Sigmoid()  # Map output to [0, 1]
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        # z_q shape: (B, num_patches, d)
        
        # Project back to embed_dim
        x = self.in_proj(z_q)  # (B, num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Pass through transformer
        x = self.blocks(x)
        x = self.norm(x)
        
        # --- Un-patch and reconstruct ---
        # 1. Permute back to (B, embed_dim, num_patches)
        x = x.transpose(1, 2)
        
        # 2. Un-flatten to grid (B, embed_dim, n_patches, t_patches)
        x = x.view(-1, self.embed_dim, self.grid_size[0], self.grid_size[1])
        
        # 3. Decode patches to full array
        x_hat = self.head(x)  # (B, 1, N, T)
        
        return x_hat


class FSQ_VAE(nn.Module):
    """
    A Transformer-based (ViT) Autoencoder using FSQ.
    """
    def __init__(
        self, 
        levels: list[int],
        img_size: tuple[int, int] = (1024, 1024),
        patch_size: tuple[int, int] = (32, 32),
        embed_dim: int = 256,
        encoder_depth: int = 6,
        decoder_depth: int = 6,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.fsq_dim = len(levels)
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Calculate patch grid dimensions
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim
        )
        
        # 2. Transformer Encoder
        self.encoder = ViTEncoder(
            num_patches=self.num_patches,
            embed_dim=embed_dim,
            fsq_dim=self.fsq_dim,
            depth=encoder_depth,
            num_heads=num_heads
        )
        
        # 3. FSQ Module
        self.fsq = FSQ(levels)
        
        # 4. Transformer Decoder
        self.decoder = ViTDecoder(
            num_patches=self.num_patches,
            embed_dim=embed_dim,
            fsq_dim=self.fsq_dim,
            grid_size=self.grid_size,
            patch_size=self.patch_size,
            depth=decoder_depth,
            num_heads=num_heads
        )

    def forward(self, x_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full pass for training.
        x_in shape: (B, N, T), normalized to [0, 1].
        """
        # Add channel dim: (B, N, T) -> (B, 1, N, T)
        x = x_in.unsqueeze(1)
        
        # 1. Embed patches
        patch_embeddings = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # 2. Encode
        z_e = self.encoder(patch_embeddings)  # (B, num_patches, d)
        
        # 3. Quantize
        # FSQ forward applies to the last dimension
        z_q_normalized = self.fsq(z_e)  # (B, num_patches, d)
        
        # 4. Decode
        x_hat = self.decoder(z_q_normalized)  # (B, 1, N, T)
        
        # Remove channel dim
        x_hat = x_hat.squeeze(1)  # (B, N, T)
        
        return x_hat, z_e, z_q_normalized

    @torch.no_grad()
    def compress(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        Compresses the input array into a sequence of integer indices.
        x_in shape: (B, N, T), normalized to [0, 1]
        """
        x = x_in.unsqueeze(1)  # (B, 1, N, T)
        
        patch_embeddings = self.patch_embed(x)  # (B, num_patches, embed_dim)
        z_e = self.encoder(patch_embeddings)    # (B, num_patches, d)
        
        # Quantize (no STE, but fsq.forward doesn't use it anyway)
        z_q_normalized = self.fsq(z_e)
        
        # Get indices
        # This is the compressed data!
        indices = self.fsq.codes_to_indexes(z_q_normalized) # (B, num_patches)
        
        return indices

    @torch.no_grad()
    def decompress(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decompresses a sequence of integer indices back into an array.
        indices shape: (B, num_patches)
        """
        # (B, num_patches) -> (B, num_patches, d)
        z_q_normalized = self.fsq.indexes_to_codes(indices)
        
        # Decode
        x_hat = self.decoder(z_q_normalized)  # (B, 1, N, T)
        
        return x_hat.squeeze(1)  # (B, N, T)


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


if __name__ == "__main__":

    rank, world_size, local_rank = setup_distributed()

    # Data dimensions (must be divisible by patch_size)
    N_DIM = 1024 
    T_DIM = 1024 
    P_DIM = 16 # Patch size (16 x 16)

    # FSQ levels (e.g., codebook size 4096)
    levels = [7, 5, 5, 5, 5]
    d = len(levels)

    # ViT Hypeparameters
    EMBED_DIM = 256  # Transformer working dimension
    ENC_DEPTH = 6    # Encoder layers
    DEC_DEPTH = 6    # Decoder layers
    NUM_HEADS = 8    # Attention heads

    # Training hyperparameters
    EPOCHS = 1

    # Create the model and wrap in DDP
    model = FSQ_VAE(
        levels=levels,
        img_size=(N_DIM, T_DIM),
        patch_size=(P_DIM, P_DIM),
        embed_dim=EMBED_DIM,
        encoder_depth=ENC_DEPTH,
        decoder_depth=DEC_DEPTH,
        num_heads=NUM_HEADS
    )
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Set up optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    # set up dataset and data loaders
    ds = load_dataset("eminorhan/neural-pile-primate", split="train")
    ds = split_dataset_by_node(ds, rank, world_size)
    print(f"Rank {rank}: Sharded dataset size: {len(ds)}")

    train_loader = DataLoader(
        ds, 
        batch_size=32, 
        shuffle=True  # this shuffles the local shard
    )

    # ====== training loop ======
    model.train()
    optimizer.zero_grad()
    train_loss = 0.0

    print(f"[Rank {rank}] Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):            
        for i, batch_data in enumerate(train_loader):
            # Move data to the correct GPU
            data = batch_data.to(local_rank, non_blocking=True)
            
            # Forward pass
            x_reconstructed, _, _ = model(data)
            
            # Compute loss
            loss = loss_fn(x_reconstructed, data)
            
            # Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if rank == 0 and i % 50 == 0:
                print(f"Epoch {epoch} | Batch {i}/{len(train_loader)} | Train Loss: {loss.item():.6f}")


    # ====== compression & decompression eval (after training) ======
    model.eval()

    sample = data_normalized[0].unsqueeze(0) # (1, 1024, 1024)

    # Compress
    # Original (1, 1024, 1024) float32 array: ~4 MB
    # Compressed (1, 1024) uint32 array: ~4 KB
    # (num_patches = (1024*1024) / (32*32) = 1024)
    compressed_indices = model.compress(sample)

    # Decompress
    decompressed_sample = model.decompress(compressed_indices)

    print(f"Original shape: {sample.shape}")
    print(f"Compressed shape: {compressed_indices.shape}") # (1, 1024)
    print(f"Decompressed shape: {decompressed_sample.shape}")