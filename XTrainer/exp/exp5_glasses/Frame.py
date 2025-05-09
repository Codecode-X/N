import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)
from get_model import Clip_model
from utils import setup_logger, set_random_seed
setup_logger(os.path.join(current_dir, "log.txt")) # Redirect output to log.txt file
set_random_seed(3407)  # Set random seed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from GlassesDataset import GlassesDataset
from torch.utils.data import DataLoader

class CLIPGlassesFrame(nn.Module):
    def __init__(self, cfg, embed_dim=512, hidden_dim=2048):
        super().__init__()
        self.cfg = cfg
        self.lambda_0 = cfg['lambda_0']
        self.register_buffer('logit_scale', Clip_model.logit_scale.detach())
        
        # Deep cross-modal interaction module (3-layer Transformer)
        self.cross_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.3,
                activation='gelu',
                batch_first=True,
                layer_norm_eps=1e-6 # Improve numerical stability
            ),
            num_layers=3
        )
        
        # Dynamic lambda generator (cross-attention mechanism)
        self.lambda_generator = nn.ModuleDict({
            'cross_attn': nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=4,
                dropout=0.2,
                batch_first=True
            ),
            'gate_controller': nn.Sequential(
                nn.Linear(2*embed_dim, 1),
                nn.Sigmoid()
            )
        })
        
        # Adaptive residual coefficient
        self.alpha = nn.Parameter(torch.ones(1)*0.1)  # Multi-stage residual
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Transformer initialization
        for p in self.cross_transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        

    def forward(self, I, h, h_neg, neg_mask=None, chunk_size=-1):
        """
        Parameters:
            - I: Image features [batch_size, embed_dim]
            - h: Text features from the last layer of the CLIP text encoder (EOS features) [batch_size, embed_dim]
            - h_neg: Text features of the negated object [batch_size, embed_dim]
            - neg_mask: Mask indicating the presence of negated objects [batch_size] | 1: has negated object | 0: no negated object
            - chunk_size: Chunk size, -1 means no chunking | set to powers of 2
        
        Returns:
            - scores: Matching scores computed by CLIPGlassesFrame [N_caps, N_imgs]
        """
        # Feature normalization
        I_norm = F.normalize(I, p=2, dim=-1)
        h_norm = F.normalize(h, p=2, dim=-1)
        h_neg_norm = F.normalize(h_neg, p=2, dim=-1) + 1e-8
        
        # Deep cross-modal interaction
        cross_features = self.cross_transformer(
            torch.cat([h_norm.unsqueeze(1), I_norm.unsqueeze(1)], dim=1)
        )
        h_attn = self.alpha*h_norm + cross_features[:,0]
        
        # Dynamic lambda generation (cross-attention mechanism)
        attn_output, _ = self.lambda_generator['cross_attn'](
            query=h_attn.unsqueeze(1),
            key=h_neg_norm.unsqueeze(1),
            value=h_neg_norm.unsqueeze(1)
        )
        gate_input = torch.cat([h_attn, attn_output.squeeze(1)], dim=-1)
        lambda_base = self.lambda_generator['gate_controller'](gate_input)
        # lambda_dynamic = torch.sigmoid(self.lambda_0 * lambda_base) # Dynamic lambda generator, constrained to [0,1]
        lambda_dynamic = F.gelu(self.lambda_0 * lambda_base) # Dynamic lambda generator
        
        # Adaptive matching to maintain CLIP's base capabilities
        with torch.amp.autocast('cuda', enabled=True):
            logit_scale = self.logit_scale.exp()
            
            # Base matching scores predicted by CLIP for h and I
            scores_base = logit_scale * (h_norm @ I_norm.t())
            
            if chunk_size < 1: # No chunking
                # Negation-aware adjustment
                scores_N2I = logit_scale * (h_neg_norm @ I_norm.t())
                adjusted_scores = lambda_dynamic * scores_N2I # Different negation penalties for different content
                adjusted_scores = torch.clamp(adjusted_scores, min=0.0) # Ensure adjusted_scores > 0
                
                # Conditional blending
                if neg_mask is not None:
                    neg_mask = neg_mask.to(scores_base.dtype)
                    scores = torch.where(
                        neg_mask.bool().view(-1,1),  # Convert mask to [B,1] for row broadcasting
                        scores_base.detach()-adjusted_scores,  # Use adjusted scores when True
                        scores_base  # Use original scores when False
                    )
                else:
                    scores = scores_base.detach()-adjusted_scores  # Use original logic when no mask | scores_base.detach() prevents gradient backpropagation to the original CLIP model

            else: # Chunked computation
                batch_size = h.size(0)
                num_images = I.size(0)
                scores = torch.zeros(batch_size, num_images, dtype=torch.float32)

                # Double chunking: text chunking × image chunking
                for txt_start in range(0, batch_size, chunk_size):
                    txt_end = min(txt_start + chunk_size, batch_size)
                    
                    # Text chunk features
                    h_chunk = h_norm[txt_start:txt_end].view(-1, h_norm.size(-1))  # [C_txt, D]
                    h_neg_chunk = h_neg_norm[txt_start:txt_end].view(-1, h_neg_norm.size(-1))  # [C_txt, D]
                    lambda_chunk = lambda_dynamic[txt_start:txt_end].flatten()  # [C_txt]
                    
                    # Image chunk loop
                    for img_start in range(0, num_images, chunk_size):
                        img_end = min(img_start + chunk_size, num_images)
                        
                        # Image chunk features
                        I_chunk = I_norm[img_start:img_end].view(-1, I_norm.size(-1))  # [C_img, D]
                        
                        # Compute base scores for the chunk (force 2D output)
                        scores_base = logit_scale * torch.mm(h_chunk, I_chunk.t())  # [C_txt, C_img]
                        
                        # Compute negation adjustment for the chunk (force 2D output)
                        scores_N2I = logit_scale * torch.mm(h_neg_chunk, I_chunk.t()) # [C_txt, C_img]
                        adjusted = lambda_chunk.unsqueeze(-1) * scores_N2I  # [C_txt, C_img]
                        adjusted.clamp_(min=0.0)
                        
                        # Conditional blending
                        if neg_mask is not None:
                            mask_chunk = neg_mask[txt_start:txt_end].bool().view(-1, 1)  # [C_txt, 1]
                            chunk_scores = torch.where(mask_chunk, scores_base - adjusted, scores_base)
                        else:
                            chunk_scores = scores_base - adjusted
                        
                        # Write directly to the result matrix
                        scores[txt_start:txt_end, img_start:img_end] = chunk_scores
                        
                        # Clear memory
                        del scores_base, scores_N2I, adjusted, chunk_scores
                        torch.cuda.empty_cache()
                        assert scores.shape[0] == scores.shape[1], f"Chunked computation resulted in abnormal scores dimensions: {scores.shape}"
        
        return scores.float()
 
        
    @staticmethod
    def load_model(cfg):
        """
        Load the trained CLIPGlassesFrame model from a checkpoint
        
        Parameters:
            - cfg: Configuration parameters
            - model_path: Path to the model checkpoint
            
        Returns:
            - model: Loaded model
        """
        model = CLIPGlassesFrame(cfg)
        if 'model_path' in cfg.keys() and cfg['model_path'] is not None:
            print(f"Loading CLIPGlassesFrame model weights: {cfg['model_path']}")
            model.load_state_dict(torch.load(cfg['model_path'], weights_only=False))
        model = model.to(cfg['device'])
        model.eval()
        return model
