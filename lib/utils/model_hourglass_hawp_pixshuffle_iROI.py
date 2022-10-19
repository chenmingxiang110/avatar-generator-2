import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck2D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2D, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class MultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class, head_size):
        super(MultitaskHead, self).__init__()

        m = int(input_channels / 4)
        heads = []
        for output_channels in sum(head_size, []):
            heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(head_size, []))

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


class AngleDistanceHead(nn.Module):
    def __init__(self, input_channels, num_class, head_size):
        super(AngleDistanceHead, self).__init__()

        m = int(input_channels/4)

        heads = []
        for output_channels in sum(head_size, []):
            if output_channels != 2:
                heads.append(
                    nn.Sequential(
                        nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(m, output_channels, kernel_size=1),
                    )
                )
            else:
                heads.append(
                    nn.Sequential(
                        nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        CosineSineLayer(m)
                    )
                )
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(head_size, []))
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


class HourglassNet(nn.Module):
    """Hourglass model from Newell et al ECCV 2016"""

    def __init__(self, inplanes, num_feats, block, head, depth, num_stacks, num_blocks, num_classes):
        super(HourglassNet, self).__init__()

        self.inplanes = inplanes
        self.num_feats = num_feats
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats * block.expansion
        # vpts = []
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, depth))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(head(ch, num_classes))
            # vpts.append(VptsHead(ch))
            # vpts.append(nn.Linear(ch, 9))
            # score.append(nn.Conv2d(ch, num_classes, kernel_size=1))
            # score[i].bias.data[0] += 4.6
            # score[i].bias.data[2] += 4.6
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        # self.vpts = nn.ModuleList(vpts)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)

            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out[::-1], y


def build_hourglass(
    is_junction=True, is_line=True, head_size=None,
    inplanes=64, num_feats=128, num_stacks=2, depth=4, num_blocks=1,
):
    if head_size is None:
        if is_junction and is_line:
            head_size = [[3], [1], [1], [2], [2]]
        elif is_junction:
            head_size = [[2], [2]]
        elif is_line:
            head_size = [[3], [1], [1]]
        else:
            raise ValueError("Both is_junction and is_line are False. Cannot determine the head_size.")
        
    num_class = sum(sum(head_size, []))
    model = HourglassNet(
        block=Bottleneck2D,
        inplanes = inplanes,
        num_feats= num_feats,
        depth=depth, # hawp uses 4 但最大处要缩放128倍不使用320*480
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size),
        num_stacks = num_stacks,
        num_blocks = num_blocks,
        num_classes = num_class
    )
    model.out_feature_channels = num_feats * 2
    return model


class Model_line_veri(nn.Module):
    
    def __init__(self, n_points, pool_stride, dim_loi, dim_fc, device):
        super().__init__()
        self.n_points = n_points
        self.pool_stride = pool_stride
        self.dim_loi = dim_loi
        self.dim_fc = dim_fc
        
        self.register_buffer('tspan', torch.linspace(0, 1, self.n_points)[None,None,:].to(device))
        
        self.pool1d = nn.MaxPool1d(self.pool_stride, self.pool_stride)
        self.fc = nn.Sequential(
            nn.Linear(self.dim_loi * (self.n_points // self.pool_stride) , self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, 4),
        )
        
    def extract_pt_feats(self, features_per_image, lines_per_im):
        n_channels, h, w = features_per_image.shape
        self.n_points = self.tspan.shape[-1]
        U,V = lines_per_im[:,:2], lines_per_im[:,2:]
        sampled_points = U[:,:,None]*self.tspan + V[:,:,None]*(1-self.tspan) - 0.5
        sampled_points = sampled_points.permute((0,2,1)).reshape(-1,2)
        px,py = sampled_points[:,0],sampled_points[:,1]
        px0 = px.floor().clamp(min=0, max=w-1)
        py0 = py.floor().clamp(min=0, max=h-1)
        px1 = (px0 + 1).clamp(min=0, max=w-1)
        py1 = (py0 + 1).clamp(min=0, max=h-1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

        xp = (( # 这四个与hawp的xy是相反的
            features_per_image[:, px0l, py0l] * (py1 - py) * (px1 - px) + \
            features_per_image[:, px0l, py1l] * (py - py0) * (px1 - px) + \
            features_per_image[:, px1l, py0l] * (py1 - py) * (px - px0) + \
            features_per_image[:, px1l, py1l] * (py - py0) * (px - px0)
        ).reshape(n_channels,-1,self.n_points)).permute(1,0,2)
        return xp
    
    def forward(self, features_per_image, lines_per_im):
        xp = self.extract_pt_feats(features_per_image, lines_per_im)
        features_per_line = self.pool1d(xp).view(xp.size(0), -1)
        logits = self.fc(features_per_line)
        return logits

#########################################################################################################

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

def get_junctions(jloc, scale_factor, topk=300, th=0):
    height, width = jloc.size(1), jloc.size(2)
    jloc = jloc.reshape(-1)

    scores, index = torch.topk(jloc, k=topk)
    r = torch.div(index, width, rounding_mode='floor').float()
    c = (index % width).float()

    junctions = torch.stack((r,c)).t()

    return junctions[scores>th] * scale_factor, scores[scores>th]

class LineSegDetector(nn.Module):
    def __init__(
        self, n_class, dim_loi=128, dim_fc=1024, n_pts0=36, n_pts1=3, span_1=3, d_max=5,
        n_dyn_junc=300, n_dyn_posl=300, n_dyn_negl=300, line_adj_dist=8,
        hour_glass_params=None, device=torch.device("cpu"),
    ):
        """
        span_1: line normal span on the feature map! should be the actual span divided by the scale factor.
        """
        super().__init__()
        
        self.device = device
        
        self.n_class = n_class
        self.n_pts0 = n_pts0
        self.n_pts1 = n_pts1
        self.span_1 = span_1
        self.dim_loi = dim_loi
        self.dim_fc = dim_fc
        self.n_dyn_junc = n_dyn_junc
        self.n_dyn_posl = n_dyn_posl
        self.n_dyn_negl = n_dyn_negl
        self.d_max = d_max
        self.line_adj_dist = line_adj_dist
        
        if hour_glass_params is None:
            self.feat_extractor = build_hourglass(head_size=[[3], [1], [1], [16]])
        else:
            self.feat_extractor = build_hourglass(
                head_size=[[3], [1], [1], [16]],
                inplanes=hour_glass_params["inplanes"],
                num_feats=hour_glass_params["num_feats"],
                num_stacks=hour_glass_params["num_stacks"],
                depth=hour_glass_params["depth"],
                num_blocks=hour_glass_params["num_blocks"],
            )
        self.pix_shuffle = nn.PixelShuffle(4)
        self.fc_loi = nn.Conv2d(self.feat_extractor.out_feature_channels, self.dim_loi, 1)
        # self.pool1d = nn.MaxPool1d(self.n_pts0//self.n_pts1, self.n_pts0//self.n_pts1)
        # self.fc_class = nn.Sequential(
        #     nn.Linear(self.dim_loi * self.n_pts1, self.dim_fc),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.dim_fc, self.dim_fc),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.dim_fc, self.n_class),
        # )
        # self.fc_width = nn.Sequential(
        #     nn.Linear(self.dim_loi * self.n_pts1, self.dim_fc),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.dim_fc, self.dim_fc),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.dim_fc, 1),
        # )
        
        self.line_feat_extractor = nn.Sequential(
            nn.Conv2d(self.dim_loi, self.dim_fc, kernel_size=3, stride=(2,1), padding=1),
            nn.BatchNorm2d(self.dim_fc),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_fc, self.dim_fc, kernel_size=3, stride=(2,1), padding=1),
            nn.BatchNorm2d(self.dim_fc),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_fc, self.dim_fc, kernel_size=3, stride=3, padding=0),
            nn.AdaptiveAvgPool2d(1)
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
        
        self.criterion_ce = nn.CrossEntropyLoss(reduction="none")
        self.criterion_mse = nn.MSELoss(reduction="none")
        
        # self.register_buffer('tspan', torch.linspace(0, 1, self.n_pts0)[None,None,:].to(self.device))
        self.register_buffer('tspan_0', torch.linspace(0, 1, self.n_pts0)[None,None,:].to(self.device))
        self.register_buffer('tspan_1', (torch.linspace(-1, 1, self.n_pts1) * self.span_1 / 2)[None,None,:].to(self.device))
        self.register_buffer('rot_mat_90', torch.from_numpy(np.array([[0,-1],[1,0]], np.float32)).to(self.device))
        self.register_buffer('rot_mat_counter_90', torch.from_numpy(np.array([[0,1],[-1,0]], np.float32)).to(self.device))
        self.to(self.device)
    
    def proposal_lines(self, md_maps, dis_maps, scale_factor):
        height, width = md_maps.size(1), md_maps.size(2)
        
        _r = torch.arange(0,height,device=self.device).float()
        _c = torch.arange(0,width, device=self.device).float()
        r0, c0 = torch.meshgrid(_r, _c, indexing='ij')
        
        dist = dis_maps[0]
        theta = (md_maps[0] - 0.5) * 2 * np.pi
        theta_l = md_maps[1] * np.pi / 2
        theta_r = md_maps[2] * np.pi / 2
        d_l = dist/torch.cos(theta_l)
        d_r = dist/torch.cos(theta_r)
        
        pt_l0 = ((r0 + torch.cos(theta+theta_l) * d_l)).clamp(min=0,max=height-1)
        pt_l1 = ((c0 + torch.sin(theta+theta_l) * d_l)).clamp(min=0,max=width-1)
        pt_r0 = ((r0 + torch.cos(theta-theta_r) * d_r)).clamp(min=0,max=height-1)
        pt_r1 = ((c0 + torch.sin(theta-theta_r) * d_r)).clamp(min=0,max=width-1)

        lines = torch.stack((pt_l0, pt_l1, pt_r0, pt_r1)).permute((1,2,0)) * scale_factor
        return lines # (80, 120, 4)
    
    # def pooling(self, features_per_image, lines_per_im):
    #     h,w = features_per_image.size(1), features_per_image.size(2)
    #     U,V = lines_per_im[:,:2], lines_per_im[:,2:]
    #     sampled_points = U[:,:,None]*self.tspan + V[:,:,None]*(1-self.tspan) - 0.5
    #     sampled_points = sampled_points.permute((0,2,1)).reshape(-1,2)
    #     px, py = sampled_points[:,0],sampled_points[:,1]
    #     px0 = px.floor().clamp(min=0, max=w-1)
    #     py0 = py.floor().clamp(min=0, max=h-1)
    #     px1 = (px0 + 1).clamp(min=0, max=w-1)
    #     py1 = (py0 + 1).clamp(min=0, max=h-1)
    #     px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

    #     xp = (
    #         (
    #             features_per_image[:, py0l, px0l] * (py1-py) * (px1 - px) + \
    #             features_per_image[:, py1l, px0l] * (py - py0) * (px1 - px) + \
    #             features_per_image[:, py0l, px1l] * (py1 - py) * (px - px0) + \
    #             features_per_image[:, py1l, px1l] * (py - py0) * (px - px0)
    #         ).reshape(self.dim_loi, -1, self.n_pts0)
    #     ).permute(1,0,2)
    #     xp = self.pool1d(xp)
    #     features_per_line = xp.view(-1, self.n_pts1*self.dim_loi)
    #     hs_logits = self.fc_class(features_per_line)
    #     hs_widths = self.fc_width(features_per_line)
    #     return hs_logits, hs_widths
    
    def pooling(self, features_per_image, lines_per_im):
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
        ).reshape(self.dim_loi, -1, self.n_pts0, self.n_pts1).permute(1,0,2,3)
        
        features_per_line = self.line_feat_extractor(xp).squeeze(3).squeeze(2)
        
        hs_logits = self.fc_class(features_per_line)
        hs_widths = self.fc_width(features_per_line)
        return hs_logits, hs_widths
    
    def forward(
        self, xs, targets=None, mode=None, junc_scale_factor=1, afm_scale_factor=4,
        T_juncs=0.01, T_lines=0.35
    ):
        if mode is None:
            is_training = self.training
        else:
            assert mode in ["train", "eval"]
            is_training = mode=="train"
        
        if is_training:
            assert targets is not None
            return self.forward_train(xs, targets)
        else:
            return self.forward_eval(xs, T_juncs, T_lines, junc_scale_factor, afm_scale_factor)
    
    def forward_train(self, xs, targets):
        junc_scale_factor = targets[0]["junc_scale_factor"]
        afm_scale_factor = targets[0]["afm_scale_factor"]
        outputs, features = self.feat_extractor(xs)
        
        loss_dict = {
            'loss_md': 0.0,
            'loss_dis': 0.0,
            'loss_res': 0.0,
            'loss_jloc': 0.0,
            'loss_pos': 0.0,
            'loss_neg': 0.0,
            'loss_width': 0.0,
        }
        extras = {}
        
        afm_gt = torch.stack([target["afm"] for target in targets])
        jloc_gt = torch.squeeze(torch.stack([target["jloc_gt"] for target in targets]), dim=1)
        
        mask = (afm_gt[:,0]>=0).float()
        for output in outputs:
            md_pred = output[:,:3].sigmoid()
            dis_pred = output[:,3:4].sigmoid()
            res_pred = output[:,4:5].sigmoid()
            jloc_pred = torch.squeeze(self.pix_shuffle(output[:,5:].sigmoid()), dim=1)
            
            loss_map = torch.mean(F.l1_loss(md_pred, afm_gt[:,1:], reduction='none'), dim=1, keepdim=True)
            loss_dict['loss_md']  += torch.mean(loss_map*mask) / torch.mean(mask) / len(outputs)
            loss_map = F.l1_loss(dis_pred, afm_gt[:,:1] / self.d_max, reduction='none')
            loss_dict['loss_dis'] += torch.mean(loss_map*mask) /torch.mean(mask) / len(outputs)
            loss_residual_map = F.l1_loss(res_pred, loss_map, reduction='none')
            loss_dict['loss_res'] += torch.mean(loss_residual_map*mask)/torch.mean(mask) / len(outputs)
            loss_dict['loss_jloc'] += jloc_loss(jloc_pred, jloc_gt) / len(outputs)
        
        output = outputs[0] # shape = N, 9, 80, 120
        loi_features = self.fc_loi(features)
        
        md_pred = output[:,:3].sigmoid()
        dis_pred = output[:,3:4].sigmoid()
        res_pred = output[:,4:5].sigmoid()
        jloc_pred = self.pix_shuffle(output[:,5:].sigmoid())
        jloc_pred_nms = []
        for i in range(len(jloc_pred)):
            jloc_pred_nms.append(non_maximum_suppression(jloc_pred[i], kernel_size=7))
        jloc_pred_nms = torch.stack(jloc_pred_nms)
        extras["jloc_pred_nms"] = jloc_pred_nms
        
        lines_batch = []
        batch_size = md_pred.size(0)

        for i, (md_pred_per_im, dis_pred_per_im, res_pred_per_im, target) in enumerate(
            zip(md_pred, dis_pred, res_pred, targets)
        ):
            junction_gt = torch.from_numpy(target["batch_juncs"].astype(np.float32)).to(self.device)
            N = junction_gt.size(0)
            
            lines_pred = []
            for scale in [-1.0,0.0,1.0]:
                _ = self.proposal_lines(
                    md_pred_per_im, (dis_pred_per_im+scale*res_pred_per_im) * self.d_max, afm_scale_factor
                ).view(-1, 4)
                lines_pred.append(_)
            lines_pred = torch.cat(lines_pred)
            
            juncs_pred, _ = get_junctions(
                jloc_pred_nms[i], junc_scale_factor, topk=min(N*2+2, self.n_dyn_junc),
            )
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

            hs_logits, hs_widths = self.pooling(loi_features[i], lines_for_train / afm_scale_factor)
            hs_widths = hs_widths.flatten()
            
            loss_width = self.criterion_mse(hs_widths, labels_width_for_train)
            loss_class = self.criterion_ce(hs_logits, labels_class_for_train)

            loss_class_positive = loss_class[labels_class_for_train!=0].mean()
            loss_class_negative = loss_class[labels_class_for_train==0].mean()
            loss_width = loss_width[labels_class_for_train!=0].mean()

            loss_dict['loss_pos'] += loss_class_positive/batch_size
            loss_dict['loss_neg'] += loss_class_negative/batch_size
            loss_dict['loss_width'] += loss_width/batch_size
            
        return loss_dict, extras
    
    def forward_eval(self, xs, T_juncs, T_lines, junc_scale_factor, afm_scale_factor):
        assert xs.shape[0]==1
        
        outputs, features = self.feat_extractor(xs)
        output = outputs[0] # shape = N, 9, 80, 120
        loi_features = self.fc_loi(features)
        
        md_pred = output[:,:3].sigmoid()
        dis_pred = output[:,3:4].sigmoid()
        res_pred = output[:,4:5].sigmoid()
        jloc_pred = self.pix_shuffle(output[:,5:].sigmoid())
        jloc_pred_nms = []
        for i in range(len(jloc_pred)):
            jloc_pred_nms.append(non_maximum_suppression(jloc_pred[i], kernel_size=7))
        jloc_pred_nms = torch.stack(jloc_pred_nms)
        
        md_pred_per_im, dis_pred_per_im, res_pred_per_im = md_pred[0], dis_pred[0], res_pred[0]
        
        lines_pred = []
        for scale in [-1.0,0.0,1.0]:
            _ = self.proposal_lines(
                md_pred_per_im, (dis_pred_per_im+scale*res_pred_per_im) * self.d_max, afm_scale_factor
            ).view(-1, 4)
            lines_pred.append(_)
        lines_pred = torch.cat(lines_pred)
        jloc_pred_nms = non_maximum_suppression(jloc_pred[0])
        
        juncs_pred, _ = get_junctions(
            jloc_pred_nms, junc_scale_factor,
            topk=min(self.n_dyn_junc, int((jloc_pred_nms>T_juncs).float().sum().item())),
        )
        
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
        
        hs_logits, hs_widths = self.pooling(loi_features[0], lines_adjusted / afm_scale_factor)
        hs_scores = F.softmax(hs_logits, dim=1)
        # hs_true = torch.argmax(hs_logits, dim=1)!=0
        hs_true = (hs_scores[:,0]<T_lines) + (torch.argmax(hs_logits, dim=1)!=0)

        lines_final = lines_adjusted[hs_true]
        score_final = hs_scores[hs_true]
        width_final = hs_widths[hs_true].flatten()

        juncs_final = juncs_pred[idx_lines_for_junctions.unique()]
        juncs_score = _[idx_lines_for_junctions.unique()]
        
        res = {
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