"""
Example models for DNALongBench CMP task.
Save as: my_models.py

Then run:
  python train_eval_cmp.py --model "my_models:SparseTransformer" --steps 1000
"""
import torch
import torch.nn as nn


class SparseTransformer(nn.Module):
    """
    Your model must:
    - Accept input: (batch_size, 1048576, 4) dtype int8
    - Return output: (batch_size, 99681) dtype float32
    
    This is a placeholder - replace with your actual architecture!
    """
    def __init__(self):
        super().__init__()
        
        # Example: Simple pooling + transformer
        # Replace this with your sparse attention architecture
        
        d_model = 128
        self.pool = nn.AdaptiveAvgPool1d(448)  # 1M seq -> 448 bins
        self.embed = nn.Linear(4, d_model)
        
        # Your transformer layers here
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=512,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1048576, 4) - one-hot DNA sequence
        Returns:
            pred: (B, 99681) - contact map upper triangle
        """
        # Step 1: Pool to 448 bins (or use your sparse attention on full 1M)
        x = x.float().permute(0, 2, 1)    # (B, 4, 1048576)
        x = self.pool(x).permute(0, 2, 1)  # (B, 448, 4)
        
        # Step 2: Embed and transform
        x = self.embed(x)  # (B, 448, d_model)
        x = self.transformer(x)  # (B, 448, d_model)
        e = self.proj(x)  # (B, 448, d_model)
        
        # Step 3: Create 448x448 contact map via outer product
        cm = torch.matmul(e, e.transpose(1, 2))  # (B, 448, 448)
        
        # Step 4: Extract upper triangle (offset=2)
        triu = torch.triu_indices(448, 448, offset=2, device=x.device)
        return cm[:, triu[0], triu[1]]  # (B, 99681)


class SimpleBaseline(nn.Module):
    """
    Minimal trainable baseline - useful for debugging.
    """
    def __init__(self, d: int = 32):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(448)
        self.proj = nn.Conv1d(4, d, kernel_size=1)
        self.scale = d ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().permute(0, 2, 1)      # (B, 4, 1048576)
        x = self.pool(x)                     # (B, 4, 448)
        e = self.proj(x).permute(0, 2, 1)    # (B, 448, d)
        cm = torch.matmul(e, e.transpose(1, 2)) * self.scale  # (B, 448, 448)
        triu = torch.triu_indices(448, 448, offset=2, device=x.device)
        return cm[:, triu[0], triu[1]]


# Test your model
if __name__ == '__main__':
    print("Testing SparseTransformer...")
    model = SparseTransformer()
    
    # Create dummy input
    x = torch.randint(0, 2, (2, 1048576, 4), dtype=torch.int8)
    
    # Forward pass
    pred = model(x)
    
    # Check shapes
    print(f"✓ Input:  {x.shape} {x.dtype}")
    print(f"✓ Output: {pred.shape} {pred.dtype}")
    
    if pred.shape == (2, 99681):
        print("✓ Model is compatible!")
    else:
        print(f"✗ Wrong shape! Expected (2, 99681), got {pred.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {params:,}")