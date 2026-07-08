import torch
import torch.nn as nn
import torch.nn.functional as F
from models.edsr import make_edsr_baseline
from utils import make_coord

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class BF_NIR_conv_Cell(nn.Module):
    def __init__(self, feat_dim=64, guide_dim=64, spa_edsr_num=4, mlp_dim=[256, 128], NIR_dim=32, cell_decode=True):
        super().__init__()
        self.guide_dim = guide_dim
        self.NIR_dim = NIR_dim
        self.cell_decode = cell_decode

        
        # n_colors=16 (6 Pan + 8 LMS)
        self.spatial_encoder = make_edsr_baseline(n_resblocks=spa_edsr_num, n_feats=self.guide_dim, n_colors=16, no_upsampling=True)
        
        
        imnet_in_dim = self.guide_dim + 2
        if self.cell_decode:
            imnet_in_dim += 2

        self.imnet = MLP(imnet_in_dim, out_dim=NIR_dim, hidden_list=mlp_dim)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(NIR_dim, NIR_dim, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(NIR_dim, 8, kernel_size=5, padding=2, bias=False)
        )

    def query(self, feat, coord, cell):
        # feat: [B, C, h, w] (Base Feature Grid)
        # coord: [B, N, 2] (Query Coordinates)
        # cell: [B, N, 2] (Query Pixel Size)

        b, c, h, w = feat.shape
        B, N, _ = coord.shape
        
        
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)

        
        rx = 1 / h
        ry = 1 / w

        preds = []
        areas = []
        
        
        
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx
                coord_[:, :, 1] += vy * ry
                
                
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                
                
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                
                
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                
                inp_list = [q_feat, rel_coord]
                if self.cell_decode and cell is not None:
                    
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= h
                    rel_cell[:, :, 1] *= w
                    inp_list.append(rel_cell)
                
                inp = torch.cat(inp_list, dim=-1)

                
                
                pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1)
                preds.append(pred)

                
                
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        
        
        
        
        # 0:(-1,-1), 1:(-1,1), 2:(1,-1), 3:(1,1)
        
        
        
        tot_area = torch.stack(areas).sum(dim=0)
        
        # Swap
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        
        return ret

    def forward(self, HR_MSI, lms, LR_HSI=None, cell=None, target_shape=None):
        """
        HR_MSI: Guide (Pan) [B, 8, H_in, W_in]
        lms: Upsampled MS [B, 8, H_in, W_in]
        target_shape: (H_out, W_out) 目标输出分辨率
        """
        
        
        
        feat_spa = torch.cat([HR_MSI, lms], dim=1)
        hr_spa = self.spatial_encoder(feat_spa) # [B, 64, H_in, W_in]
        
        
        if target_shape is None:
            H_out, W_out = HR_MSI.shape[-2:]
        else:
            H_out, W_out = target_shape
        
        H_out, W_out = int(H_out), int(W_out)
        
        
        coord = make_coord((H_out, W_out)).to(HR_MSI.device).unsqueeze(0).expand(HR_MSI.shape[0], -1, -1)

        
        if self.cell_decode:
            if cell is None:
                
                cell = torch.ones_like(coord)
                cell[:, :, 0] *= 2 / H_out
                cell[:, :, 1] *= 2 / W_out
            else:
                pass 
                

        
        NIR_feature = self.query(hr_spa, coord, cell) # [B, H_out*W_out, NIR_dim]
        
        
        NIR_feature = NIR_feature.permute(0, 2, 1).view(HR_MSI.shape[0], -1, H_out, W_out)

        
        output = self.decoder(NIR_feature)
        
        
        if (lms.shape[-2] != H_out) or (lms.shape[-1] != W_out):
            lms_out = F.interpolate(lms, size=(H_out, W_out), mode='bicubic', align_corners=False)
        else:
            lms_out = lms
            
        output = output + lms_out

        return output