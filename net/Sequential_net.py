import torch
import torch.nn as nn
import torch.nn.functional as nnf

from utils import BaseNet, Timer
#from mmcv.cnn.utils import flops_counter
import time     # only for display
from .NAFNet_net import NAFBlock
from .forswin_network_swinir import SwinUnit
from torchinfo import summary
from ptflops import get_model_complexity_info

class Block(nn.Module):
    def __init__(self, nf_in, nf_out, if_cicr, if_cicr_add, block_num):
        super().__init__()
        self.preconv_ = nf_in != nf_out
        self.if_cicr = if_cicr
        self.if_cicr_add = if_cicr_add
        
        if if_cicr:
            self.qeprocess = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=nf_out, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid(),
            )
            if if_cicr_add:
                self.qeprocess_conv  = nn.Sequential(
                nn.Conv2d(in_channels=nf_in, out_channels=nf_in, kernel_size=3, padding=3//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=nf_in, out_channels=nf_in, kernel_size=3, padding=3//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=nf_in, out_channels=nf_in, kernel_size=3, padding=3//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=nf_in, out_channels=nf_in, kernel_size=3, padding=3//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=nf_in, out_channels=nf_in, kernel_size=3, padding=3//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=nf_in, out_channels=nf_in, kernel_size=3, padding=3//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=nf_in, out_channels=nf_in, kernel_size=3, padding=3//2),
                nn.ReLU(inplace=True),
                )
                self.qeprocess_2 = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=nf_in, kernel_size=1, stride=1, padding=0),
                    nn.Tanh(),
                )
        self.block = nn.Sequential(
            *[NAFBlock(nf_out) for _ in range(block_num)],
        )
        if nf_in != nf_out:
            self.preconv = nn.Sequential(
                nn.Conv2d(in_channels=nf_in, out_channels=nf_out, kernel_size=1, padding=0)
            )


    def forward(self, inp, qe=0.5):
        if self.preconv_:
            inp = self.preconv(inp)

        if self.if_cicr:
            if self.if_cicr_add:
                feat_ = self.qeprocess_conv(inp)
                feat_ = feat_ * self.qeprocess(qe)
                feat_ = torch.add(feat_, self.qeprocess_2(qe))
                feat = inp + feat_
            else:
                feat = inp * self.qeprocess(qe)

            feat = self.block(feat)
        else:
            feat = self.block(inp)
            
        return feat



class Up(nn.Module):
    def __init__(self, nf_in_small, nf_out, method):

        super().__init__()

        if method == 'upsample':
            assert nf_in_small == nf_out, '> Output channel number should be equal to input channel number (upsampling).'
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
            )
        elif method == 'transpose2d':
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=nf_in_small,
                    out_channels=nf_out,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            )
        elif method == 'pixelshuffler_w-conv':
            assert nf_in_small == nf_out, '> Output channel number should be equal to input channel number (upsampling).'
            self.up = nn.Sequential(
                nn.Conv2d(nf_in_small, nf_in_small*4, 1, bias=False),
                nn.PixelShuffle(2),
            )


    def forward(self, small_t, *normal_t_lst):
        feat = self.up(small_t)
        if len(normal_t_lst) > 0:
            h_s, w_s = feat.size()[2:]  # B C H W
            h, w = normal_t_lst[0].size()[2:]

            dh = h - h_s
            dw = w - w_s
            if dh < 0:
                feat = feat[:, :, :h, :]
                dh = 0
            if dw < 0:
                feat = feat[:, :, :, :w]
                dw = 0

            feat = nnf.pad(
                input=feat,
                pad=[
                    dw//2, (dw-dw//2),  # only pad H and W; left (diffW//2); right remaining (diffW - diffW//2)
                    dh//2, (dh-dh//2)
                ],
                mode='constant',
                value=0,  # pad with constant 0
            )

            feat = torch.cat((feat, *normal_t_lst), dim=1)

        return feat


class Down(nn.Module):
    def __init__(self, nf_in, nf_out, method):
        super().__init__()

        if method == 'avepool2d':
            assert nf_in == nf_out, '> Output channel number should be equal to input channel number (upsampling).'
            self.down = nn.Sequential(
                nn.AvgPool2d(kernel_size=2),
            )

        elif method == 'strideconv':
            self.down = nn.Sequential(
                nn.Conv2d(in_channels=nf_in, out_channels=nf_out, kernel_size=3, padding=3//2, stride=2),
            )
        elif method == 'strideconv_NAF':
            self.down = nn.Sequential(
                nn.Conv2d(in_channels=nf_in, out_channels=nf_out, kernel_size=2, stride=2),
            )

    def forward(self, inp):
        feat = self.down(inp)
        return feat


class CEBranch(nn.Module):
    def __init__(self, nf_in=64, nf_base=64):   # conv3 *10, conv1
        super(CEBranch, self).__init__()
        self.qe_estimation = nn.Sequential(
            Block(nf_in = nf_in, nf_out = nf_base, if_cicr = False, if_cicr_add = False, block_num = 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=nf_base, out_channels=nf_base//2, kernel_size=1,),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nf_base//2, out_channels=nf_base//4, kernel_size=1,),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nf_base//4, out_channels=nf_base//8, kernel_size=1,),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=nf_base//8, out_channels=1, kernel_size=1,),
            nn.Sigmoid(),
        )

    def forward(self, input_map, if_cicr=True):
        if if_cicr:
            score_quality_tensor = self.qe_estimation(input_map)
        else:
            score_quality_tensor = torch.ones([input_map.size()[0],1,1,1])
        
        return score_quality_tensor


def split_img(inp_t, if_split):
    inp_t_batch = []
    _, _, inp_t_h, inp_t_w = inp_t.shape
    h, w, overlap_h, overlap_w = if_split[0]['size_h'], if_split[0]['size_w'], if_split[0]['overlap_h'], if_split[0]['overlap_w']
    stride_h, stride_w = h - overlap_h, w - overlap_w
    h_index_list = list(range(0, inp_t_h-overlap_h, stride_h))
    w_index_list = list(range(0, inp_t_w-overlap_w, stride_w))
    h_index_list[-1], w_index_list[-1] = max(0, inp_t_h - h), max(0, inp_t_w - w)

    for h_idx in h_index_list:
        for w_idx in w_index_list:
            inp_t_batch.append(inp_t[:, :, h_idx:h_idx+h, w_idx:w_idx+w])

    return inp_t_batch, h_index_list, w_index_list


def combine_img(out_t_batch, if_split, map_size, h_index_lst, w_index_lst, if_train):
    size_h, size_w = if_split[0]['size_h'], if_split[0]['size_w']
    map_out = torch.zeros(map_size).to('cuda')
    overlap_h, overlap_w = [], []
    for y in range(len(h_index_lst)-1):
        overlap_h.append(size_h - (h_index_lst[y+1] - h_index_lst[y]))
    for x in range(len(w_index_lst)-1):
        overlap_w.append(size_w - (w_index_lst[x+1] - w_index_lst[x]))

    def overlap_adjust(map_to_add, h_idx, w_idx):
        # 0,1,2,...
        h_num, w_num = h_index_lst.index(h_idx), w_index_lst.index(w_idx)

        if h_num < len(h_index_lst)-1 and overlap_h[h_num] != 0:
            map_to_add[..., -overlap_h[h_num]//2:, :] *= 0
        if w_num < len(w_index_lst)-1 and overlap_w[w_num] != 0:
            map_to_add[..., :, -overlap_w[w_num]//2:] *= 0
        if h_num > 0 and overlap_h[h_num-1] != 0:
            map_to_add[..., :overlap_h[h_num-1]//2, :] *= 0
        if w_num > 0 and overlap_w[w_num-1] != 0:
            map_to_add[..., :, :overlap_w[w_num-1]//2] *= 0

        return map_to_add

    if if_train:   # input: [ [ tensor(4ch) ]*index ]*batch
        map_out_lst = []
        for index_num in range(len(out_t_batch[0])):
            batch_num = 0
            for h_idx in h_index_lst:
                for w_idx in w_index_lst:
                    map_to_add = overlap_adjust(out_t_batch[batch_num][index_num], h_idx, w_idx)
                    map_out[:, :, h_idx:h_idx+size_h, w_idx:w_idx+size_w].add_(map_to_add)
                    map_out_lst.append(map_out)
                    batch_num = batch_num + 1
        return map_out_lst

    else:   # input: [ tensor(4ch) ]*batch
        batch_num = 0
        for h_idx in h_index_lst:
            for w_idx in w_index_lst:
                map_to_add = overlap_adjust(out_t_batch[batch_num], h_idx, w_idx)
                map_out[:, :, h_idx:h_idx+size_h, w_idx:w_idx+size_w].add_(map_to_add)
                batch_num = batch_num + 1

    return map_out


class Network(nn.Module):
    def __init__(self, if_train, nf_in=3, nf_base=64, nlevel_step=[5], 
                 down_method='strideconv', up_method='transpose2d', if_cicr=False, if_plus=False, 
                 nf_out=3, if_residual=True, block_num_enc=2, block_num_dec=2, window_size=7,
                 depths_swin=[6,6], num_heads=[6,6], mlp_ratio=2, resi_connection='1conv', init_type='default', comp_type='jpeg'):
        assert down_method in ['avepool2d', 'strideconv', 'strideconv_NAF'], '> not supported yet.'
        assert up_method in ['upsample', 'transpose2d', 'pixelshuffler_w-conv'], '> not supported yet.'
        if type(block_num_enc) is int:
            l_block_enc = [block_num_enc] * max(nlevel_step)
            block_num_enc = l_block_enc
        if type(block_num_dec) is int:
            l_block_dec = [block_num_dec] * max(nlevel_step)
            block_num_dec = l_block_dec
        assert len(block_num_enc) == len(block_num_dec) <= max(nlevel_step), '> <block_num> should be int or a list whose length is equal to nlevel.'

        super().__init__()

        self.if_train = if_train
        self.nlevel = nlevel_step
        self.if_residual = if_residual
        self.if_cicr = if_cicr
        self.if_cicr_add = if_plus

        # input conv
        self.inconvs = nn.Sequential(
            nn.Conv2d(in_channels=nf_in, out_channels=nf_base, kernel_size=3, padding=3//2),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=nf_base, out_channels=nf_base, kernel_size=3, padding=3//2),
        )

        # down then up at each nested u-net
        for step in range(len(nlevel_step)):
            idx_num = nlevel_step[step]
            for idx in range(idx_num-1):
                if step == 0 or idx == 0:
                    setattr(self, f'block_enc_{step}_{idx}', Block(
                        nf_in = nf_base,
                        nf_out = nf_base,
                        if_cicr = self.if_cicr,
                        if_cicr_add = self.if_cicr_add,
                        block_num = block_num_enc[idx],
                    ))
                else:
                    setattr(self, f'block_enc_{step}_{idx}', Block(
                        nf_in = nf_base * 2,
                        nf_out = nf_base,
                        if_cicr = self.if_cicr,
                        if_cicr_add = self.if_cicr_add,
                        block_num = block_num_enc[idx],
                    ))
                setattr(self, f'down_{step}_{idx}', Down(
                    nf_in = nf_base,
                    nf_out = nf_base,
                    method = down_method,
                ))
                setattr(self, f'up_{step}_{idx}', Up(
                    nf_in_small = nf_base,
                    nf_out = nf_base, 
                    method = up_method,
                ))
                if idx == idx_num-2:
                    setattr(self, f'block_dec_{step}_{idx}', Block(
                        nf_in = nf_base * 3,
                        nf_out = nf_base,
                        if_cicr = self.if_cicr,
                        if_cicr_add = self.if_cicr_add,
                        block_num = block_num_dec[idx],
                    ))
                else:
                    setattr(self, f'block_dec_{step}_{idx}', Block(
                        nf_in = nf_base * 2,
                        nf_out = nf_base,
                        if_cicr = self.if_cicr,
                        if_cicr_add = self.if_cicr_add,
                        block_num = block_num_dec[idx],
                    ))
            if step == 0:
                setattr(self, f'block_mid_{step}', Block(
                    nf_in = nf_base,
                    nf_out = nf_base,
                    if_cicr = self.if_cicr,
                    if_cicr_add = self.if_cicr_add,
                    block_num = block_num_enc[idx_num-1],
                ))
            else:
                setattr(self, f'block_mid_{step}', Block(
                    nf_in = nf_base * 2,
                    nf_out = nf_base,
                    if_cicr = self.if_cicr,
                    if_cicr_add = self.if_cicr_add,
                    block_num = block_num_enc[idx_num-1],
                ))


        # out, each step
        self.outconv_lst = nn.ModuleList([
                nn.Conv2d(in_channels=nf_base, out_channels=nf_out, kernel_size=3, padding=3//2),
                nn.Conv2d(in_channels=nf_base, out_channels=nf_out, kernel_size=3, padding=3//2),
                nn.Conv2d(in_channels=nf_base, out_channels=nf_out, kernel_size=3, padding=3//2),
                nn.Conv2d(in_channels=nf_base, out_channels=nf_out, kernel_size=3, padding=3//2),
                nn.Conv2d(in_channels=nf_base, out_channels=nf_out, kernel_size=3, padding=3//2),
                nn.Conv2d(in_channels=nf_base, out_channels=nf_out, kernel_size=3, padding=3//2),
            ]
        )
        
        # IQA module
        self.ce = CEBranch(nf_in = nf_base, nf_base = nf_base)

        # Global Context by Swin Transformer
        self.gc = SwinUnit(embed_dim=nf_base, depths=depths_swin, num_heads=num_heads,
                 window_size=window_size, mlp_ratio=mlp_ratio, resi_connection=resi_connection)

    def forward(self, inp_t_wholeimg, if_iqa=False, idx_in=-1, if_split=None, **_):
        if if_iqa:
            timer_wo_iqam = Timer()
            timer_wo_iqam.record()

        if if_split is not None:
            inp_t_splitted, h_index_lst, w_index_lst = split_img(inp_t_wholeimg, if_split)
        else:
            inp_t_splitted = [inp_t_wholeimg]   # need to be fixed(when size are not multiple of 4/8/16)

        feat_lst_splitted = []
        for b in inp_t_splitted:
            feat_lst_splitted.append(self.inconvs(b))

        out_t_lst = []

        qe_est_splitted = []
        for b in feat_lst_splitted:
            qe_est_splitted.append(self.ce(b)) # QF(jpeg) = 100(1-qe_est), QP(bpg) = 51*qe_est

        qe_est = sum(qe_est_splitted) / len(qe_est_splitted)
        ut = time.time()
        if int(ut*1000000)%1000 == 1:
            print('QF(estimation, mean:std) : ',float(torch.mean(100*(1-qe_est))), ':', float(torch.std(100*(1-qe_est))))

        split_num = 0
        out_t_lst_splitted = []
        for _ in range(len(self.nlevel)):
            out_t_lst_splitted.append([])

        for f in feat_lst_splitted:
            for step in range(len(self.nlevel)):
                idx_num = self.nlevel[step]
                feat_lst_lst = [f]
                
                for idx in range(idx_num-1): # process each idx like U-Net
                    enc = getattr(self, f'block_enc_{step}_{idx}')
                    f = enc(f, qe_est)
                    feat_lst_lst.append(f)
                    down = getattr(self, f'down_{step}_{idx}')
                    if idx == idx_num-2:    # idx_num-2, 0
                        gl = self.gc(f)
                    f = down(f)
                    if step != 0:
                        f = torch.cat((f, feat_lst_skip[idx_num-idx-2]), dim=1)
                mid = getattr(self, f'block_mid_{step}')
                f = mid(f, qe_est)
                feat_lst_skip = [f]
                for idx in range(idx_num-2, -1, -1):
                    up = getattr(self, f'up_{step}_{idx}')
                    f = up(f)
                    f = torch.cat((f, feat_lst_lst[idx+1]), dim=1)
                    if idx == idx_num-2:    # idx_num-2, 0
                        f = torch.cat((f, gl), dim=1)
                    dec = getattr(self, f'block_dec_{step}_{idx}')
                    f = dec(f, qe_est)
                    feat_lst_skip.append(f)
                f = f + feat_lst_lst[0]    # skip connection

                res_split = self.outconv_lst[step](f)
                if self.if_residual:
                    res_split += inp_t_splitted[split_num]
                out_t_lst_splitted[step].append(res_split)
            split_num += 1

        if if_iqa:
            timer_wo_iqam.record()
        for step in range(len(self.nlevel)):
            if if_split is not None:
                out_t_lst.append(combine_img(out_t_lst_splitted[step], if_split, inp_t_wholeimg.size(), h_index_lst, w_index_lst, self.if_train))
            else:
                out_t_lst.append(out_t_lst_splitted[step][0])

        if idx_in==-1:  # train, with all step
            return torch.stack(out_t_lst, dim=0), qe_est  # nlevel B C H W
        elif if_iqa:    # test/val with required time
            return sum(timer_wo_iqam.inter_lst), out_t_lst[-1], qe_est  # B=1 C H W
        else:   # test/val without required time
            return out_t_lst[-1], qe_est  # B=1 C H W


class SequentialModel(BaseNet):
    def __init__(self, opts_dict, if_train):
        self.net = dict(net=Network(if_train, **opts_dict['net']))
        get_model_complexity_info(Network(if_train, **opts_dict['net']), (3,256,256), as_strings=False)
        summary(Network(if_train, **opts_dict['net']), input_size=(1,3,256,256))
        super().__init__(opts_dict=opts_dict, if_train=if_train, infer_subnet='net')