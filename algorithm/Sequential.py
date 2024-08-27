import net
import torch
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
import io

from utils import BaseAlg, CUDATimer, Recorder, tensor2im
import re

class SequentialAlgorithm(BaseAlg):
    def __init__(self, opts_dict, if_train, if_dist):
        model_cls = getattr(net, 'SequentialModel')  # FIXME
        super().__init__(opts_dict=opts_dict, model_cls=model_cls, if_train=if_train, if_dist=if_dist)
        self.opts_dict = opts_dict

    def accum_gradient(self, module, stage, group, data, inter_step, additional):
        data_lq_in = data['lq']
        data_gt_in = data['gt']

        data_lq = data_lq_in.cuda(non_blocking=True)
        data_gt = data_gt_in.cuda(non_blocking=True)

        data_out_lst, qe_est = module(inp_t_wholeimg=data_lq, if_iqa=False, if_train=True) # QF(jpeg) = 100(1-qe_est), QP(bpg) = 51*qe_est
        nl, nb = data_out_lst.shape[0:2]  # step, batch size

        num_show = 3
        self._im_lst = dict(
            data_lq=data['lq'][:num_show],
            data_gt=data['gt'][:num_show],
            generated=data_out_lst[-1].detach()[:num_show].cpu().clamp_(0., 1.),
        )  # show images from the last exit

        loss_total = torch.tensor(0., device="cuda")
        for loss_name in self.loss_lst[stage][group]:
            loss_dict = self.loss_lst[stage][group][loss_name]

            loss_unweighted = 0.
            for idx_data in range(nb):
                im_type = data['name'][idx_data].split('_')[-1].split('.')[0]
                comp_index_gt = float(re.sub(r"\D", "", im_type))
                assert 'qf' or 'qp' in im_type, 'CONFIRM INPUT DATA!'
                if 'qf' in im_type: # QF(jpeg) = 100(1-qe_est), 
                    q_gt = 1 - comp_index_gt/100.
                elif 'qp' in im_type:   # QP(bpg) = 51*qe_est
                    q_gt = comp_index_gt/51.
                q_est = qe_est[idx_data]
                loss_weight_lst = []
                for idx_level in range(nl):
                    loss_weight_lst.append(nl - (nl-1.) * abs(q_gt - idx_level/nl))

                for idx_level in range(nl):
                    opts_dict_ = dict(
                        inp=data_out_lst[idx_level, idx_data, ...],     # step, batch, c, h, w
                        ref=data_gt[idx_data, ...],
                        q_inp=q_est.squeeze(),
                        q_ref=q_gt,
                    )
                    loss_unweighted += loss_weight_lst[idx_level] * \
                        loss_dict['fn'](**opts_dict_) / sum(loss_weight_lst)
            loss_unweighted /= float(nb)

            setattr(self, f'{loss_name}_{group}', loss_unweighted.item())  # for recorder

            loss_ = loss_dict['weight'] * loss_unweighted
            loss_total += loss_

        loss_total /= float(inter_step)  # multiple backwards and step once, thus mean
        loss_total.backward()  # must backward only once; otherwise the graph is freed after the first backward
        setattr(self, f'loss_{group}', loss_total.item())  # for recorder

    @torch.no_grad()
    def test(self, data_fetcher, num_samples, if_baseline=False, if_return_each=False, img_save_folder=None,
             if_train=True, if_split=None):
        """
        val (in training): idx_in=0(all QF)
        train: idx_in=-1(default, all), test: idx_in=-2, record time wo. iqa
        """
        if if_baseline or if_train:
            assert self.crit_lst is not None, 'NO METRICS!'

        if self.crit_lst is not None:
            if_tar_only = False
            msg = 'dst vs. src | ' if if_baseline else 'tar vs. src | '
        else:
            if_tar_only = True
            msg = 'only get dst | '

        report_dict = None

        recorder_dict = dict()
        for crit_name in self.crit_lst:
            recorder_dict[crit_name] = Recorder()

        write_dict_lst = []
        timer = CUDATimer()

        # validation baseline: no iqa, no parse name
        # validation, not baseline: no iqa, parse name
        # test baseline: no iqa, no parse name
        # test, no baseline, iqa, no parse name
        if_iqa = True if (not if_train) and (not if_baseline) else False
        if if_iqa:
            timer_wo_iqam = Recorder()
            idx_in = -2  # testing; judge by IQAM
        if_parse_name = True if if_train and (not if_baseline) else False

        self.set_eval_mode()

        data_fetcher.reset()
        test_data = data_fetcher.next()
        assert len(test_data['name']) == 1, 'ONLY SUPPORT bs==1!'

        pbar = tqdm(total=num_samples, ncols=100)

        while test_data is not None:
            im_lq = test_data['lq'].cuda(non_blocking=True)  # assume bs=1
            im_name = test_data['name'][0]  # assume bs=1

            if if_parse_name:
                idx_in = 0

            timer.start_record()
            if if_tar_only:
                if if_iqa:
                    time_wo_iqa, im_out, qe_est = self.model.net[self.model.infer_subnet](inp_t_wholeimg=im_lq, if_iqa=if_iqa, idx_in=idx_in, if_split=if_split).clamp_(0., 1.)
                else:
                    im_out, qe_est = self.model.net[self.model.infer_subnet](inp_t_wholeimg=im_lq, if_iqa=if_iqa, idx_in=idx_in, if_split=if_split).clamp_(0., 1.)
                timer.record_inter()
            else:
                im_gt = test_data['gt'].cuda(non_blocking=True)  # assume bs=1
                if if_baseline:
                    im_out = im_lq
                else:
                    if if_iqa:
                        time_wo_iqa, im_out, qe_est = self.model.net[self.model.infer_subnet](inp_t_wholeimg=im_lq, if_iqa=if_iqa, idx_in=idx_in, if_split=if_split)
                        im_out = im_out.clamp_(0., 1.)
                    else:
                        im_out, qe_est = self.model.net[self.model.infer_subnet](inp_t_wholeimg=im_lq, if_iqa=if_iqa, idx_in=idx_in, if_split=if_split)
                        im_out = im_out.clamp_(0., 1.)
                timer.record_inter()

                _msg = f'{im_name} | '

                for crit_name in self.crit_lst:
                    crit_fn = self.crit_lst[crit_name]['fn']
                    crit_unit = self.crit_lst[crit_name]['unit']

                    perfm = crit_fn(torch.squeeze(im_out, 0), torch.squeeze(im_gt, 0))
                    recorder_dict[crit_name].record(perfm)

                    _msg += f'[{perfm:.4f}] {crit_unit:s} | '

                _msg = _msg[:-3]
                if if_return_each:
                    msg += _msg + '\n'
                pbar.set_description(_msg)

            if if_iqa:
                timer_wo_iqam.record(time_wo_iqa)

            if img_save_folder is not None:  # save im
                im = tensor2im(torch.squeeze(im_out, 0))
                save_path = img_save_folder / (str(im_name) + '.png')
                cv2.imwrite(str(save_path), im)

            pbar.update()
            test_data = data_fetcher.next()
        pbar.close()

        if not if_tar_only:
            for crit_name in self.crit_lst:
                crit_unit = self.crit_lst[crit_name]['unit']
                crit_if_focus = self.crit_lst[crit_name]['if_focus']

                ave_perfm = recorder_dict[crit_name].get_ave()
                msg += f'{crit_name} | [{ave_perfm:.4f}] {crit_unit} | '

                write_dict_lst.append(dict(tag=f'{crit_name} (val)', scalar=ave_perfm))

                if crit_if_focus:
                    report_dict = dict(ave_perfm=ave_perfm, lsb=self.crit_lst[crit_name]['fn'].lsb)

        ave_fps = 1. / timer.get_ave_inter()
        msg += f'ave. fps | [{ave_fps:.1f}]'

        if if_iqa:
            ave_time_wo_iqam = timer_wo_iqam.get_ave()
            fps_wo_iqam = 1. / ave_time_wo_iqam
            msg += f' | ave. fps wo. IQAM | [{fps_wo_iqam:.1f}]'

        if if_train:
            assert report_dict is not None
            return msg.rstrip(), write_dict_lst, report_dict
        else:
            return msg.rstrip()
