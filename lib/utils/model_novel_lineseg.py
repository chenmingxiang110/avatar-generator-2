import time
from tqdm import tqdm, trange

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.model_pspnet import PSPNet, PSPUpsample

#########################################################################################################

class ResidualCNN(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.cnn1 = nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1)
        self.cnn2 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.is_shortcut = in_channels!=out_channels or stride!=1
        if self.is_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = self.shortcut(x) if self.is_shortcut else x
        x = self.cnn1(x)
        x = self.cnn2(x)
        x += residual
        x = self.bn(x)
        x = self.act(x)
        return x

class BNLRCNN(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, xs):
        return self.block(xs)

class Bottleneck(nn.Module):

    def __init__(self, dim_in, dim_mid, dim_out, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_mid, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(dim_mid)
        self.conv2 = nn.Conv2d(dim_mid, dim_mid, kernel_size=4 if stride==2 else 3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(dim_mid)
        self.conv3 = nn.Conv2d(dim_mid, dim_out, kernel_size=1)
        self.bn_out = nn.BatchNorm2d(dim_out)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        if dim_in!=dim_out or stride!=1:
            self.shortcut = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=stride, padding=0)
        else:
            self.shortcut = None

    def forward(self, x, is_output_activate=True):
        residual = x
        out = self.conv1(x)

        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.act(out)
        out = self.conv3(out)

        if self.shortcut is not None:
            residual = self.shortcut(x)

        out += residual
        if is_output_activate:
            out = self.bn_out(out)
            out = self.act(out)
        
        return out

class LineSegDetBackbone(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        
        self.down128 = nn.Sequential(
            BNLRCNN(3, 64, 7, 2, 3),
            BNLRCNN(64, 128, 5, 2, 2),
        ) # 4x
        self.down64 = nn.Sequential(
            Bottleneck(128, 64, 256, stride=2),
            Bottleneck(256, 64, 256),
            Bottleneck(256, 64, 256),
        ) # 8x
        self.down32 = nn.Sequential(
            Bottleneck(256, 128, 512, stride=2),
            Bottleneck(512, 128, 512),
            Bottleneck(512, 128, 512),
            Bottleneck(512, 128, 512),
        ) # 16x
        self.down16 = nn.Sequential(
            Bottleneck(512, 256, 1024, stride=2),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
        ) # 32x
        self.down8 = nn.Sequential(
            Bottleneck(1024, 512, 2048, stride=2),
            Bottleneck(2048, 512, 2048),
            Bottleneck(2048, 512, 2048),
        ) # 64x
        self.up16 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 2, stride=2, padding=0),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
        ) # 16x
        self.up32 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=0),
            Bottleneck(512, 128, 512),
            Bottleneck(512, 128, 512),
            Bottleneck(512, 128, 512),
        ) # 16x
        self.up64 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0),
            Bottleneck(256, 64, 256),
            Bottleneck(256, 64, 256),
            Bottleneck(256, 64, 256),
        ) # 8x
        self.up128 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0),
            Bottleneck(128, 32, 128),
            Bottleneck(128, 32, 128),
            Bottleneck(128, 32, 128),
        ) # 4x
        self.out = BNLRCNN(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.out_feature_channels = 128
    
    def forward(self, xs):
        down_128 = self.down128(xs)
        down_64 = self.down64(down_128)
        down_32 = self.down32(down_64)
        down_16 = self.down16(down_32)
        down_8 = self.down8(down_16)
        up_16 = self.up16(down_8) + down_16
        up_32 = self.up32(up_16) + down_32
        up_64 = self.up64(up_32) + down_64
        up_128 = self.up128(up_64) + down_128
        hs = self.out(up_128)
        return hs

def jloc_loss(logits, positive):
    return F.binary_cross_entropy(logits, positive)

def joff_loss(logits, targets, mask=None):
    loss = torch.abs(logits-targets)
    if mask is not None:
        w = mask.mean(3, True).mean(2,True)
        w[w==0] = 1
        loss = loss*(mask/w)
    return loss.mean()

def non_maximum_suppression(a, kernel_size=3):
    ap = F.max_pool2d(a, kernel_size, stride=1, padding=kernel_size//2)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask

def get_junctions(jloc, topk=300, th=0):
    height, width = jloc.size(1), jloc.size(2)
    jloc = jloc.reshape(-1)

    scores, index = torch.topk(jloc, k=topk)
    r = torch.div(index, width, rounding_mode='floor').float()
    c = (index % width).float()

    junctions = torch.stack((r,c)).t()

    return junctions[scores>th], scores[scores>th]

class LineSegDetector(nn.Module):
    def __init__(
        self, n_class, dim_fc=256, n_pts0=32, n_pts1=8, span_1=12, line_adj_dist=8, lpre_adj_dist=4,
        n_dyn_junc=120, n_dyn_posl=300, n_dyn_negl=300, backbone_path=None,
        train_feat_extractor=True, device=torch.device("cpu"),
    ):
        super().__init__()
        
        self.device = device
        
        self.n_class = n_class
        self.n_pts0 = n_pts0
        self.n_pts1 = n_pts1
        self.span_1 = span_1
        self.dim_fc = dim_fc
        self.n_dyn_junc = n_dyn_junc
        self.n_dyn_posl = n_dyn_posl
        self.n_dyn_negl = n_dyn_negl
        self.line_adj_dist = line_adj_dist
        self.lpre_adj_dist = lpre_adj_dist
        self.train_feat_extractor = train_feat_extractor
        
        # self.feat_extractor = LineSegDetBackbone(self.n_class)
        self.feat_extractor = PSPNet(n_classes=self.n_class, sizes=(1, 2, 3, 6), psp_size=2048, backend='resnet50')
        if backbone_path is not None:
            self.feat_extractor.load_state_dict(torch.load(backbone_path))
        
        self.feat_scale_factor = 1
        self.out_feature_channels = 64
        
        self.line_feat_extractor = nn.Sequential(
            nn.Conv2d(self.out_feature_channels, self.dim_fc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.dim_fc),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(self.dim_fc, self.dim_fc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.dim_fc),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_fc, self.dim_fc, kernel_size=4, stride=4, padding=0),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.conv_junc_segs = nn.Sequential(
            PSPUpsample(1024, 256),
            nn.Dropout2d(p=0.15),
            PSPUpsample(256, 64),
            nn.Dropout2d(p=0.15),
            PSPUpsample(64, 64),
            nn.Dropout2d(p=0.15),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        
        self.fc_class = nn.Sequential(
            nn.Linear(self.dim_fc, 2 * self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.dim_fc, 2 * self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.dim_fc, self.n_class),
        )
        self.fc_width = nn.Sequential(
            nn.Linear(self.dim_fc, 2 * self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.dim_fc, 2 * self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.dim_fc, 1),
        )
        
        self.criterion_bce = nn.BCELoss()
        self.criterion_ce = nn.CrossEntropyLoss(reduction="none")
        self.criterion_mse = nn.MSELoss(reduction="none")
        
        self.register_buffer('tspan_0', torch.linspace(0, 1, self.n_pts0)[None,None,:].to(self.device))
        self.register_buffer('tspan_1', (torch.linspace(-1, 1, self.n_pts1) * self.span_1 / 2)[None,None,:].to(self.device))
        self.register_buffer('rot_mat_90', torch.from_numpy(np.array([[0,-1],[1,0]], np.float32)).to(self.device))
        self.register_buffer('rot_mat_counter_90', torch.from_numpy(np.array([[0,1],[-1,0]], np.float32)).to(self.device))
        self.to(self.device)
    
    def line_suppress(self, lines, T_seg_sup_dist=5, T_seg_sup_angle=20):
        tmp_batch_lines_hs = lines.detach().cpu().numpy()
        tmp_batch_lines_hs_0 = [] # 水平或垂直的
        tmp_batch_lines_hs_1 = [] # 斜的
        for x in tmp_batch_lines_hs:
            if min(abs(x[2]-x[0]), abs(x[3]-x[1]))<=T_seg_sup_dist:
                tmp_batch_lines_hs_0.append(x)
            else:
                tmp_batch_lines_hs_1.append(x)
        tmp_batch_lines_hs_0 = sorted(tmp_batch_lines_hs_0, key=lambda x: np.linalg.norm([x[2]-x[0],x[3]-x[1]]))
        tmp_batch_lines_hs_1 = sorted(tmp_batch_lines_hs_1, key=lambda x: np.linalg.norm([x[2]-x[0],x[3]-x[1]]))
        tmp_batch_lines_hs = tmp_batch_lines_hs_0 + tmp_batch_lines_hs_1
        batch_lines_hs = []
        for line in tmp_batch_lines_hs:
            is_valid = True
            for registered_line in batch_lines_hs:
                orders = [
                    [(0,1), (2,3), (0,1), (2,3)],
                    [(0,1), (2,3), (2,3), (0,1)],
                    [(2,3), (0,1), (0,1), (2,3)],
                    [(2,3), (0,1), (2,3), (0,1)],
                ]
                dists = [np.linalg.norm([
                    line[o[0][0]]-registered_line[o[2][0]],
                    line[o[0][1]]-registered_line[o[2][1]]
                ]) for o in orders]
                
                i_order = np.argmin(dists)
                o = orders[i_order]
                min_dist = dists[i_order]
                
                if min_dist<T_seg_sup_dist:
                    vec0 = np.array([
                        line[o[1][0]]-line[o[0][0]],
                        line[o[1][1]]-line[o[0][1]],
                    ])
                    vec1 = np.array([
                        registered_line[o[3][0]]-registered_line[o[2][0]],
                        registered_line[o[3][1]]-registered_line[o[2][1]],
                    ])
                    _cos = vec0.dot(vec1) / np.linalg.norm(vec0) / np.linalg.norm(vec1)
                    angle = np.arccos(_cos) / np.pi * 180
                    if angle<T_seg_sup_angle:
                        is_valid = False
                        break
            if is_valid:
                batch_lines_hs.append(line)
        batch_lines_hs = np.array(batch_lines_hs, np.float32)
        
        return torch.from_numpy(batch_lines_hs).to(self.device)
        
    def propose_lines(self, pts):
        lines = []
        for i_pt in range(len(pts)-1):
            pt0 = pts[i_pt]
            for j_pt in range(i_pt+1, len(pts)):
                pt1 = pts[j_pt]
                if np.random.random()<0.5:
                    lines.append(torch.stack([pt0[0], pt0[1], pt1[0], pt1[1]]))
                else:
                    lines.append(torch.stack([pt1[0], pt1[1], pt0[0], pt0[1]]))
        if len(lines)>0:
            lines = torch.stack(lines)
        return lines # (N, 4)
    
    def pooling(self, features_per_image, lines_per_im, extras=None):
        h,w = features_per_image.size(1), features_per_image.size(2)
        U,V = lines_per_im[:,:2], lines_per_im[:,2:]
        line_dir = F.normalize(V-U, p=2.0, dim=1)
        
        if np.random.random()<0.5:
            line_norm = torch.matmul(line_dir, self.rot_mat_90)
        else:
            line_norm = torch.matmul(line_dir, self.rot_mat_counter_90)
        
        dir_points = U[:,:,None]*self.tspan_0 + V[:,:,None]*(1-self.tspan_0)
        dir_points = dir_points.permute((0,2,1))
        
        norm_points = line_norm[:,:,None] * self.tspan_1
        norm_points = norm_points.permute((0,2,1))
        sampled_points = dir_points[:,:,None,:] + norm_points[:,None,:,:]
        
        if extras is not None:
            extras["lines_per_im"] = lines_per_im
            extras["line_dir"] = line_dir
            extras["line_norm"] = line_norm
            extras["dir_points"] = dir_points
            extras["norm_points"] = norm_points
            extras["sampled_points"] = sampled_points
        
        sampled_points = sampled_points.reshape([-1,2])
        
        px, py = sampled_points[:,0],sampled_points[:,1]
        px0 = px.floor().clamp(min=0, max=w-1)
        py0 = py.floor().clamp(min=0, max=h-1)
        px1 = (px0 + 1).clamp(min=0, max=w-1)
        py1 = (py0 + 1).clamp(min=0, max=h-1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

        xp = (
            features_per_image[:, py0l, px0l] * (py1-py) * (px1 - px) + \
            features_per_image[:, py1l, px0l] * (py - py0) * (px1 - px) + \
            features_per_image[:, py0l, px1l] * (py1 - py) * (px - px0) + \
            features_per_image[:, py1l, px1l] * (py - py0) * (px - px0)
        ).reshape(self.out_feature_channels, -1, self.n_pts0, self.n_pts1).permute(1,0,2,3)
        
        features_per_line = self.line_feat_extractor(xp).squeeze(3).squeeze(2)
        
        if extras is not None:
            extras["sampled_points_rs"] = sampled_points
            extras["xp"] = xp
            extras["features_per_line"] = features_per_line
        
        hs_logits = self.fc_class(features_per_line)
        hs_widths = self.fc_width(features_per_line)
        return hs_logits, hs_widths
    
    def forward(self, xs, targets=None, mode=None, T_juncs=None, T_lines=None, T_seg_sup_dist=5, T_seg_sup_angle=20, has_extras=False):
        if mode is None:
            is_training = self.training
        else:
            assert mode in ["train", "eval"]
            is_training = mode=="train"
        if is_training:
            return self.forward_train(xs, targets, has_extras)
        else:
            return self.forward_eval(xs, T_juncs, T_lines, T_seg_sup_dist, T_seg_sup_angle)
    
    def forward_train(self, xs, targets, has_extras):
        loss_dict = {
            'loss_pos': 0.0,
            'loss_neg': 0.0,
            'loss_width': 0.0,
        }
        extras = {}
        
        jloc_gt = torch.squeeze(torch.stack([target["jloc_gt"] for target in targets]), dim=1)
        _seg_gt = torch.stack([target["seg_lines"] for target in targets]) # (N, 320, 480)
        seg_gt = torch.zeros(_seg_gt.shape[0], 4, 512, 512, dtype=_seg_gt.dtype).to(_seg_gt.device)
        seg_gt[:, :, 96:-96, 16:-16] = _seg_gt # change the shape of gt
        
        """
        line_segs.shape : torch.Size([2, 4, 512, 512])
        feats0.shape    : torch.Size([2, 1024, 64, 64])
        feats1.shape    : torch.Size([2, 256, 128, 128])
        feats2.shape    : torch.Size([2, 64, 256, 256])
        feats.shape     : torch.Size([2, 64, 512, 512])
        """
        if self.train_feat_extractor:
            line_segs, feats0, feats1, feats2, feats = self.feat_extractor(xs, output_features=True, is_dropout=False)
        else:
            self.feat_extractor.eval()
            with torch.no_grad():
                line_segs, feats0, feats1, feats2, feats = self.feat_extractor(xs, output_features=True, is_dropout=False)
        
        jloc_pred = self.conv_junc_segs(feats0).sigmoid()
        
        loss_dict['loss_jloc'] = jloc_loss(torch.squeeze(jloc_pred, dim=1), jloc_gt)
        if self.train_feat_extractor:
            loss_dict['loss_seg'] = self.criterion_bce(line_segs.sigmoid(), seg_gt).mean()
        
        if has_extras:
            extras["line_segs"] = line_segs
            extras["seg_gt"] = seg_gt
            extras["feats"] = feats
        
        jloc_pred_nms = []
        for i in range(len(jloc_pred)):
            jloc_pred_nms.append(non_maximum_suppression(jloc_pred[i], kernel_size=7))
        jloc_pred_nms = torch.stack(jloc_pred_nms)
        if has_extras:
            extras["jloc_pred_nms"] = jloc_pred_nms
        
        lines_batch = []
        batch_size = feats.size(0)

        for i, target in enumerate(targets):
            junction_gt = torch.from_numpy(target["batch_juncs"].astype(np.float32)).to(self.device)
            N = junction_gt.size(0)
            juncs_pred, _ = get_junctions(
                jloc_pred_nms[i], topk=min(N*2+2, self.n_dyn_junc),
            )
            lines_pred = self.propose_lines(juncs_pred)
            
            if has_extras:
                extras["juncs_pred_"+str(i)] = juncs_pred
                extras["lines_pred_"+str(i)] = lines_pred
            
            dis_junc_to_end1, idx_junc_to_end1 = torch.sum(
                (lines_pred[:, :2] - juncs_pred[:, None]) ** 2, dim=-1
            ).min(0)
            dis_junc_to_end2, idx_junc_to_end2 = torch.sum(
                (lines_pred[:, 2:] - juncs_pred[:, None]) ** 2, dim=-1
            ).min(0)
            
            idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
            idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)
            iskeep = idx_junc_to_end_min<idx_junc_to_end_max
            idx_lines_for_junctions = torch.cat(
                (idx_junc_to_end_min[iskeep,None],idx_junc_to_end_max[iskeep,None]), dim=1
            ).unique(dim=0)
            # if len(idx_lines_for_junctions)==0???
            idx_lines_for_junctions_mirror = torch.cat(
                (idx_lines_for_junctions[:,1,None],idx_lines_for_junctions[:,0,None]), dim=1
            )
            idx_lines_for_junctions = torch.cat((idx_lines_for_junctions, idx_lines_for_junctions_mirror))
            lines_adjusted = torch.cat(
                (juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]), dim=1
            )
            
            cost_, match_ = torch.sum((juncs_pred-junction_gt[:,None])**2, dim=-1).min(0)
            match_[cost_>self.line_adj_dist*self.line_adj_dist] = N
            
            lpre = target["lpre"]
            # Add randomness
            # lpre = target["lpre"] + (torch.rand(target["lpre"].shape).to(self.device) * 2 - 1) * self.lpre_adj_dist
            
            lpre_width = target["lpre_width"]
            lpre_label = target["lpre_label"]
            lbl_mat = target["lbl_mat"]
            width_mat = target["width_mat"]
            
            labels_class = lbl_mat[
                match_[idx_lines_for_junctions[:,0]], match_[idx_lines_for_junctions[:,1]]
            ]
            labels_width = width_mat[
                match_[idx_lines_for_junctions[:,0]], match_[idx_lines_for_junctions[:,1]]
            ]
            
            iskeep = torch.zeros_like(labels_class, dtype=torch.bool)
            
            if self.n_dyn_posl > 0:
                cdx = labels_class.nonzero().flatten()
                if len(cdx) > self.n_dyn_posl:
                    perm = torch.randperm(len(cdx),device=self.device)[:self.n_dyn_posl]
                    cdx = cdx[perm]
                iskeep[cdx] = 1
            
            if self.n_dyn_negl > 0:
                cdx = (
                    lbl_mat[match_[idx_lines_for_junctions[:,0]],match_[idx_lines_for_junctions[:,1]]]==0
                ).nonzero().flatten()
                if len(cdx) > self.n_dyn_negl:
                    perm = torch.randperm(len(cdx), device=self.device)[:self.n_dyn_negl]
                    cdx = cdx[perm]
                iskeep[cdx] = 1
            
            lines_selected = lines_adjusted[iskeep]
            labels_class_selected = labels_class[iskeep]
            labels_width_selected = labels_width[iskeep]

            lines_for_train = torch.cat((lines_selected, lpre))
            labels_class_for_train = torch.cat((labels_class_selected, lpre_label))
            labels_width_for_train = torch.cat((labels_width_selected.float(), lpre_width))
            
            if has_extras:
                extras["lines_selected_"+str(i)] = lines_for_train
                extras["labels_class_selected_"+str(i)] = labels_class_for_train
                extras["labels_width_selected_"+str(i)] = labels_width_for_train
            
            if i==0 and has_extras:
                hs_logits, hs_widths = self.pooling(feats[i], lines_for_train / self.feat_scale_factor, extras)
            else:
                hs_logits, hs_widths = self.pooling(feats[i], lines_for_train / self.feat_scale_factor)
            
            hs_widths = hs_widths.flatten()
            
            loss_class = self.criterion_ce(hs_logits, labels_class_for_train)
            loss_width = self.criterion_mse(hs_widths, labels_width_for_train)

            loss_class_positive = loss_class[labels_class_for_train!=0].mean()
            loss_class_negative = loss_class[labels_class_for_train==0].mean()
            loss_width = loss_width[labels_class_for_train!=0].mean()

            loss_dict['loss_pos'] += loss_class_positive/batch_size
            loss_dict['loss_neg'] += loss_class_negative/batch_size
            loss_dict['loss_width'] += loss_width/batch_size
            
        return loss_dict, extras
    
    def forward_eval(self, xs, T_juncs, T_lines, T_seg_sup_dist, T_seg_sup_angle):
        assert T_juncs is not None and T_lines is not None, "T_juncs or T_lines is not given."
        assert xs.shape[0]==1
        
        self.feat_extractor.eval()
        with torch.no_grad():
            """
            line_segs.shape : torch.Size([2, 4, 512, 512])
            feats0.shape    : torch.Size([2, 1024, 64, 64])
            feats1.shape    : torch.Size([2, 256, 128, 128])
            feats2.shape    : torch.Size([2, 64, 256, 256])
            feats.shape     : torch.Size([2, 64, 512, 512])
            """
            line_segs, feats0, feats1, feats2, feats = self.feat_extractor(xs, output_features=True, is_dropout=False)
            
        jloc_pred = self.conv_junc_segs(feats0).sigmoid()
        
        jloc_pred_nms = []
        for i in range(len(jloc_pred)):
            jloc_pred_nms.append(non_maximum_suppression(jloc_pred[i], kernel_size=7))
        jloc_pred_nms = torch.stack(jloc_pred_nms)
        
        lines_batch = []
        
        juncs_pred, _ = get_junctions(
            jloc_pred_nms[0], topk=min(self.n_dyn_junc, int((jloc_pred_nms>T_juncs).float().sum().item())),
        )
        lines_pred = self.propose_lines(juncs_pred)
        if len(lines_pred)>0:
            lines_pred = self.line_suppress(lines_pred, T_seg_sup_dist=T_seg_sup_dist, T_seg_sup_angle=T_seg_sup_angle)

            dis_junc_to_end1, idx_junc_to_end1 = torch.sum(
                (lines_pred[:, :2] - juncs_pred[:, None]) ** 2, dim=-1
            ).min(0)
            dis_junc_to_end2, idx_junc_to_end2 = torch.sum(
                (lines_pred[:, 2:] - juncs_pred[:, None]) ** 2, dim=-1
            ).min(0)

            idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
            idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)
            iskeep = idx_junc_to_end_min<idx_junc_to_end_max

            idx_lines_for_junctions = torch.cat(
                (idx_junc_to_end_min[iskeep,None],idx_junc_to_end_max[iskeep,None]), dim=1
            ).unique(dim=0)
            lines_adjusted = torch.cat(
                (juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]), dim=1
            )

            hs_logits, hs_widths = [], []
            for i in range(0, lines_adjusted.shape[0], 300):
                h_logits, h_widths = self.pooling(feats[0], lines_adjusted[i:i+300] / self.feat_scale_factor)
                hs_logits.append(h_logits)
                hs_widths.append(h_widths)
            hs_logits = torch.cat(hs_logits)
            hs_widths = torch.cat(hs_widths)

            hs_scores = F.softmax(hs_logits, dim=1)
            hs_true = (hs_scores[:,0]<T_lines) + (torch.argmax(hs_logits, dim=1)!=0)

            lines_final = lines_adjusted[hs_true]
            score_final = hs_scores[hs_true]
            width_final = hs_widths[hs_true].flatten()

            juncs_final = juncs_pred[idx_lines_for_junctions.unique()]
            juncs_score = _[idx_lines_for_junctions.unique()]
        else:
            lines_adjusted = []
            lines_final = []
            hs_scores = []
            score_final = []
            width_final = []
            juncs_final = juncs_pred
            juncs_score = _
        
        res = {
            "feats": feats,
            "jloc_pred": jloc_pred,
            "line_segs": line_segs,
            "hs_scores": hs_scores,
            "juncs_pred": juncs_pred,
            "lines_pred": lines_pred,
            "lines_refine": lines_adjusted,
            "lines_final": lines_final,
            "score_final": score_final,
            "width_final": width_final,
            "juncs_final": juncs_final,
            "juncs_score": juncs_score,
        }
        return res
    
    def forward_train_legacy(self, xs, targets):
        loss_dict = {
            'loss_seg': 0.0,
            'loss_jloc': 0.0,
            'loss_pos': 0.0,
            'loss_neg': 0.0,
            'loss_width': 0.0,
        }
        extras = {}
        
        jloc_gt = torch.squeeze(torch.stack([target["jloc_gt"] for target in targets]), dim=1)
        _seg_gt = torch.stack([target["seg_lines"] for target in targets]) # (N, 320, 480)
        seg_gt = torch.zeros(_seg_gt.shape[0], 512, 512, dtype=_seg_gt.dtype).to(_seg_gt.device)
        seg_gt[:, 96:-96, 16:-16] = _seg_gt # change the shape of gt
        
        feats = self.feat_extractor(xs)
        line_segs = self.conv_line_segs(feats)
        jloc_pred = self.pix_shuffle(self.conv_junc_segs(feats)).sigmoid()
        
        loss_dict['loss_seg'] = self.criterion_ce(line_segs.softmax(dim=1), seg_gt).mean()
        loss_dict['loss_jloc'] = jloc_loss(torch.squeeze(jloc_pred, dim=1), jloc_gt)
        
        jloc_pred_nms = []
        for i in range(len(jloc_pred)):
            jloc_pred_nms.append(non_maximum_suppression(jloc_pred[i], kernel_size=7))
        jloc_pred_nms = torch.stack(jloc_pred_nms)
        
        lines_batch = []
        batch_size = feats.size(0)

        for i, target in enumerate(targets):
            junction_gt = torch.from_numpy(target["batch_juncs"].astype(np.float32)).to(self.device)
            N = junction_gt.size(0)
            juncs_pred, _ = get_junctions(
                jloc_pred_nms[i], topk=min(N*2+2, self.n_dyn_junc),
            )
            lines_pred = self.propose_lines(juncs_pred)
            
            # extras["juncs_pred_"+str(i)] = juncs_pred
            # extras["lines_pred_"+str(i)] = lines_pred
            
            dis_junc_to_end1, idx_junc_to_end1 = torch.sum(
                (lines_pred[:, :2] - juncs_pred[:, None]) ** 2, dim=-1
            ).min(0)
            dis_junc_to_end2, idx_junc_to_end2 = torch.sum(
                (lines_pred[:, 2:] - juncs_pred[:, None]) ** 2, dim=-1
            ).min(0)
            
            idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
            idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)
            iskeep = idx_junc_to_end_min<idx_junc_to_end_max
            idx_lines_for_junctions = torch.cat(
                (idx_junc_to_end_min[iskeep,None],idx_junc_to_end_max[iskeep,None]), dim=1
            ).unique(dim=0)
            # if len(idx_lines_for_junctions)==0???
            idx_lines_for_junctions_mirror = torch.cat(
                (idx_lines_for_junctions[:,1,None],idx_lines_for_junctions[:,0,None]), dim=1
            )
            idx_lines_for_junctions = torch.cat((idx_lines_for_junctions, idx_lines_for_junctions_mirror))
            lines_adjusted = torch.cat(
                (juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]), dim=1
            )
            
            cost_, match_ = torch.sum((juncs_pred-junction_gt[:,None])**2, dim=-1).min(0)
            match_[cost_>self.line_adj_dist*self.line_adj_dist] = N
            
            lpre = target["lpre"]
            # Add randomness
            # lpre = target["lpre"] + (torch.rand(target["lpre"].shape).to(self.device) * 2 - 1) * self.lpre_adj_dist
            
            lpre_width = target["lpre_width"]
            lpre_label = target["lpre_label"]
            lbl_mat = target["lbl_mat"]
            width_mat = target["width_mat"]
            
            labels_class = lbl_mat[
                match_[idx_lines_for_junctions[:,0]], match_[idx_lines_for_junctions[:,1]]
            ]
            labels_width = width_mat[
                match_[idx_lines_for_junctions[:,0]], match_[idx_lines_for_junctions[:,1]]
            ]
            
            iskeep = torch.zeros_like(labels_class, dtype=torch.bool)
            
            if self.n_dyn_posl > 0:
                cdx = labels_class.nonzero().flatten()
                if len(cdx) > self.n_dyn_posl:
                    perm = torch.randperm(len(cdx),device=self.device)[:self.n_dyn_posl]
                    cdx = cdx[perm]
                iskeep[cdx] = 1
            
            if self.n_dyn_negl > 0:
                cdx = (
                    lbl_mat[match_[idx_lines_for_junctions[:,0]],match_[idx_lines_for_junctions[:,1]]]==0
                ).nonzero().flatten()
                if len(cdx) > self.n_dyn_negl:
                    perm = torch.randperm(len(cdx), device=self.device)[:self.n_dyn_negl]
                    cdx = cdx[perm]
                iskeep[cdx] = 1
            
            lines_selected = lines_adjusted[iskeep]
            labels_class_selected = labels_class[iskeep]
            labels_width_selected = labels_width[iskeep]

            lines_for_train = torch.cat((lines_selected, lpre))
            labels_class_for_train = torch.cat((labels_class_selected, lpre_label))
            labels_width_for_train = torch.cat((labels_width_selected.float(), lpre_width))
            
            # extras["lpre_"+str(i)] = lpre
            # extras["lines_selected_"+str(i)] = lines_for_train
            # extras["labels_class_selected_"+str(i)] = labels_class_for_train
            # extras["labels_width_selected_"+str(i)] = labels_width_for_train
            
            hs_logits, hs_widths = self.pooling(feats[i], lines_for_train / self.feat_scale_factor)
            hs_widths = hs_widths.flatten()
            
            loss_class = self.criterion_ce(hs_logits, labels_class_for_train)
            loss_width = self.criterion_mse(hs_widths, labels_width_for_train)

            loss_class_positive = loss_class[labels_class_for_train!=0].mean()
            loss_class_negative = loss_class[labels_class_for_train==0].mean()
            loss_width = loss_width[labels_class_for_train!=0].mean()

            loss_dict['loss_pos'] += loss_class_positive/batch_size
            loss_dict['loss_neg'] += loss_class_negative/batch_size
            loss_dict['loss_width'] += loss_width/batch_size
            
        return loss_dict, extras
    
    def forward_eval_legacy(self, xs, T_juncs, T_lines):
        assert T_juncs is not None and T_lines is not None, "T_juncs or T_lines is not given."
        assert xs.shape[0]==1
        
        feats = self.feat_extractor(xs)
        line_segs = self.conv_line_segs(feats)
        jloc_pred = self.pix_shuffle(self.conv_junc_segs(feats)).sigmoid()
        
        jloc_pred_nms = []
        for i in range(len(jloc_pred)):
            jloc_pred_nms.append(non_maximum_suppression(jloc_pred[i], kernel_size=7))
        jloc_pred_nms = torch.stack(jloc_pred_nms)
        
        lines_batch = []
        
        juncs_pred, _ = get_junctions(
            jloc_pred_nms[0], topk=min(self.n_dyn_junc, int((jloc_pred_nms>T_juncs).float().sum().item())),
        )
        lines_pred = self.propose_lines(juncs_pred)
        if len(lines_pred)>0:
            lines_pred = self.line_suppress(lines_pred)

            dis_junc_to_end1, idx_junc_to_end1 = torch.sum(
                (lines_pred[:, :2] - juncs_pred[:, None]) ** 2, dim=-1
            ).min(0)
            dis_junc_to_end2, idx_junc_to_end2 = torch.sum(
                (lines_pred[:, 2:] - juncs_pred[:, None]) ** 2, dim=-1
            ).min(0)

            idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
            idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)
            iskeep = idx_junc_to_end_min<idx_junc_to_end_max

            idx_lines_for_junctions = torch.cat(
                (idx_junc_to_end_min[iskeep,None],idx_junc_to_end_max[iskeep,None]), dim=1
            ).unique(dim=0)
            lines_adjusted = torch.cat(
                (juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]), dim=1
            )

            hs_logits, hs_widths = [], []
            for i in range(0, lines_adjusted.shape[0], 300):
                h_logits, h_widths = self.pooling(feats[0], lines_adjusted[i:i+300] / self.feat_scale_factor)
                hs_logits.append(h_logits)
                hs_widths.append(h_widths)
            hs_logits = torch.cat(hs_logits)
            hs_widths = torch.cat(hs_widths)

            hs_scores = F.softmax(hs_logits, dim=1)
            hs_true = (hs_scores[:,0]<T_lines) + (torch.argmax(hs_logits, dim=1)!=0)

            lines_final = lines_adjusted[hs_true]
            score_final = hs_scores[hs_true]
            width_final = hs_widths[hs_true].flatten()

            juncs_final = juncs_pred[idx_lines_for_junctions.unique()]
            juncs_score = _[idx_lines_for_junctions.unique()]
        else:
            lines_adjusted = []
            lines_final = []
            hs_scores = []
            score_final = []
            width_final = []
            juncs_final = juncs_pred
            juncs_score = _
        
        res = {
            "feats": feats,
            "jloc_pred": jloc_pred,
            "line_segs": line_segs,
            "hs_scores": hs_scores,
            "juncs_pred": juncs_pred,
            "lines_pred": lines_pred,
            "lines_refine": lines_adjusted,
            "lines_final": lines_final,
            "score_final": score_final,
            "width_final": width_final,
            "juncs_final": juncs_final,
            "juncs_score": juncs_score,
        }
        return res