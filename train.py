import argparse
import math
import shutil
from pathlib import Path

import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

import time
import dataset
import algorithm
from utils import mkdir_archived, dict2str, DistSampler, create_dataloader, CPUPrefetcher, init_dist, set_random_seed, \
    CUDATimer, create_logger
from utils.individual.plot_curve_from_log import main as plt_curve


def arg2dict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', '-opt', type=str, default='opts/Cresnet.yml', help='path to option YAML file.')
    parser.add_argument('--case', '-case', type=str, default='div2k_jpeg', help='specified case in YAML.')
    parser.add_argument('--note', '-note', type=str, default='hello world!', help='useless; just FYI.')
    parser.add_argument('--delete_archive', '-del', action='store_true', help='delete archived experimental directories.')
    parser.add_argument('--local_rank', type=int, default=0, help='reserved for DDP.')
    args = parser.parse_args()

    with open(args.opt, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)
        opts_dict = opts_dict[args.case]

    return opts_dict, args.delete_archive, args.local_rank


def mkdir_and_create_logger(opts_dict, if_del_arc=False, rank=0, file_name=None):
    """Make log dir (also used when testing) and tensorboard writer."""
    exp_name = opts_dict['algorithm']['exp_name']
    log_dir = Path("exp") / exp_name
    if file_name is not None:
        log_dir = log_dir / file_name

    if_load_ = False
    if_warn_ = False
    if opts_dict['algorithm']['train']['load_state']['if_load']:
        ckp_load_path = opts_dict['algorithm']['train']['load_state']['opts']['ckp_load_path']
        if ckp_load_path is None:
            ckp_load_path = Path('exp') / opts_dict['algorithm']['exp_name'] / 'ckp_last.pt'

        if Path(ckp_load_path).exists():
            if_load_ = True
            log_dir = Path("exp") / exp_name / ckp_load_path.split('/')[-2]
        else:
            if_warn_ = True

    if not if_load_:
        if_mkdir_ = True
    else:
        if opts_dict['algorithm']['train']['load_state']['opts']['if_keep_dir']:
            if_mkdir_ = False
        else:
            if_mkdir_ = True

    if if_mkdir_ and rank == 0:
        mkdir_archived(log_dir, if_del_arc=if_del_arc)

    time.sleep(0.5) # To wait for making directory of logfile
    log_path = log_dir / "log_train.log"
    logger = create_logger(log_path, rank=rank, mode='a')

    tb_writer = SummaryWriter(log_dir) if rank == 0 else None

    ckp_save_path_pre = log_dir / 'ckp_'
    return logger, tb_writer, ckp_save_path_pre, if_warn_, log_path


def create_data_fetcher(if_train=False, if_baseline=False, seed=None, num_gpu=None, rank=None, ds_type=None, ds_type_base=None, ds_opts=None,
                        enlarge_ratio=None, nworker_pg=None, bs_pg=None):
    """Define data-set, data-sampler, data-loader and CPU-based data-fetcher."""
    if if_baseline:
        ds_cls = getattr(dataset, ds_type_base)
    else:
        ds_cls = getattr(dataset, ds_type)
    ds = ds_cls(ds_opts)
    num_samples = len(ds)

    sampler = DistSampler(num_replicas=num_gpu, rank=rank, ratio=enlarge_ratio, ds_size=num_samples) if if_train \
        else None

    loader = create_dataloader(if_train=if_train, seed=seed, rank=rank, num_worker=nworker_pg, batch_size=bs_pg,
                               dataset=ds, sampler=sampler)

    data_fetcher = CPUPrefetcher(loader)
    return num_samples, sampler, data_fetcher


def cal_state(batch_size_per_gpu, num_gpus, num_samples, enlarge_ratio, num_iters, done_num_iters):
    bs_per_epoch_all_gpu = batch_size_per_gpu * num_gpus
    enlarge_num_samples_pe = num_samples * enlarge_ratio
    niter_per_epoch = math.ceil(enlarge_num_samples_pe / bs_per_epoch_all_gpu)  # also batch num
    num_epochs = math.floor(num_iters / niter_per_epoch)
    done_num_epochs = done_num_iters // niter_per_epoch
    done_iter_this_epoch = done_num_iters % niter_per_epoch
    msg = (
        f'data-loader for training\n'
        f'[{num_samples}] training samples in total.\n'
        f'[{num_epochs}] epochs in total, [{done_num_epochs}] epochs finished.\n'
        f'[{num_iters}] iterations in total, [{done_num_iters}] iterations finished.'
    )
    return done_iter_this_epoch, done_num_epochs, msg


def main():
    now_time = str(time.strftime("%m-%d_%H-%M", time.localtime()))
    opts_dict, if_del_arc, rank = arg2dict()

    num_gpu = torch.cuda.device_count()
    log_paras = dict(num_gpu=num_gpu)
    opts_dict.update(log_paras)

    if_dist = True if num_gpu > 1 else False
    if if_dist:
        init_dist(local_rank=rank, backend='nccl')

    torch.backends.cudnn.benchmark = True if opts_dict['algorithm']['train']['if_cudnn'] else False
    torch.backends.cudnn.deterministic = True if not opts_dict['algorithm']['train']['if_cudnn'] else False

    # Create logger

    logger, tb_writer, ckp_save_path_pre, if_warn_, log_path = mkdir_and_create_logger(opts_dict, if_del_arc=if_del_arc,
                                                                             rank=rank, file_name=now_time)

    if if_warn_:
        logger.info('if_load is True, but NO PRE-TRAINED MODEL!')

    # Record hyper-params

    msg = f'hyper parameters\n{dict2str(opts_dict).rstrip()}'  # remove \n from dict2str()
    logger.info(msg)

    # Enlarge niter

    bs_pg = opts_dict['dataset']['train']['bs_pg']
    real_bs_pg = opts_dict['algorithm']['train']['real_bs_pg']
    assert bs_pg <= real_bs_pg and real_bs_pg % bs_pg == 0, 'CHECK bs AND real bs!'
    inter_step = real_bs_pg // bs_pg

    opts_ = opts_dict['algorithm']['train']['niter']
    niter_lst = list(map(int, opts_['niter']))
    niter_name_lst = opts_['name']
    if_manually_stop = True if ('if_manually_stop' in opts_) and opts_['if_manually_stop'] else False
    num_stage = len(niter_lst)
    end_niter_lst = [sum(niter_lst[:is_]) for is_ in range(1, num_stage + 1)]
    niter = end_niter_lst[-1]  # all stages
    niter = math.ceil(niter * inter_step)  # enlarge niter

    # Set random seed for this process

    seed = opts_dict['algorithm']['train']['seed']
    set_random_seed(seed + rank)  # if not set, seeds for numpy.random in each process are the same

    # Create data-fetcher

    opts_dict_ = dict(if_train=True, seed=seed, num_gpu=num_gpu, rank=rank, **opts_dict['dataset']['train'])
    num_samples_train, train_sampler, train_fetcher = create_data_fetcher(**opts_dict_)
    opts_dict_ = dict(if_train=False, **opts_dict['dataset']['val'])
    num_samples_val, _, val_fetcher = create_data_fetcher(**opts_dict_)
    num_samples_val_base, _, val_fetcher_base = create_data_fetcher(if_baseline=True, **opts_dict_)

    # Create algorithm

    alg_cls = getattr(algorithm, opts_dict['algorithm']['name'])
    opts_dict_ = dict(opts_dict=opts_dict['algorithm'], if_train=True, if_dist=if_dist)
    alg = alg_cls(**opts_dict_)
    alg.model.print_module(logger)

    # Calculate epoch num

    enlarge_ratio = opts_dict['dataset']['train']['enlarge_ratio']
    done_niter = alg.done_niter
    opts_dict_ = dict(batch_size_per_gpu=bs_pg, num_gpus=num_gpu, num_samples=num_samples_train,
                      enlarge_ratio=enlarge_ratio, num_iters=niter, done_num_iters=done_niter)
    done_iter_this_epoch, done_num_epochs, msg = cal_state(**opts_dict_)
    logger.info(msg)

    # Create timer

    timer = CUDATimer()
    timer.start_record()

    # Train

    best_val_perfrm = alg.best_val_perfrm

    inter_print = opts_dict['algorithm']['train']['inter_print']
    inter_val = opts_dict['algorithm']['train']['inter_val']
    if_test_baseline = opts_dict['algorithm']['train']['if_test_baseline']
    additional = opts_dict['algorithm']['train']['additional'] if 'additional' in \
                                                                  opts_dict['algorithm']['train'] else dict()

    alg.set_train_mode()
    if_all_over = False
    if_val_end_of_stage = False  # invalid at the start of training
    while True:
        if if_all_over:
            break  # leave the training process

        train_sampler.set_epoch(done_num_epochs)  # shuffle distributed sub-samplers before each epoch
        train_fetcher.reset()
        if done_niter == alg.done_niter:
            if done_iter_this_epoch > 0:
                logger.info(f'skip {done_iter_this_epoch} iteration(s) to meet the pre-trained status...')
                if_verbose = True if rank == 0 else False
                train_fetcher.skip_front(done_iter_this_epoch, verbose=if_verbose)
                logger.info(f'done.')
        train_data = train_fetcher.next()  # fetch the first batch

        while train_data is not None:
            # Validate
            # # done_niter == alg.done_niter == 0, if_test_baseline == True: test baseline (to be recorded at tb as step 0)
            # # done_niter == alg.done_niter != 0, if_keep_dir == False: test baseline and val (to be recorded at tb as step alg.done_niter)
            # # done_niter != alg.done_niter, done_niter % inter_val == 0: val
            # # done_niter != alg.done_niter, if_val_end_of_stage == True: val
            _if_test_baseline = False
            _if_val = False
            if done_niter == alg.done_niter:
                if alg.done_niter == 0 and if_test_baseline:
                    _if_test_baseline = True
                elif alg.done_niter != 0 and not opts_dict['algorithm']['train']['load_state']['opts']['if_keep_dir']:
                    _if_test_baseline = True
                    _if_val = True
            else:
                if (done_niter % inter_val == 0) or if_val_end_of_stage:
                    _if_val = True

            if rank == 0 and (_if_test_baseline or _if_val):
                if _if_test_baseline:
                    msg, tb_write_dict_lst, report_dict = alg.test(val_fetcher_base, num_samples_val, if_baseline=True)
                    logger.info(msg)
                    if done_niter == 0:
                        best_val_perfrm = dict(iter_lst=[0], perfrm=report_dict['ave_perfm'])

                    for tb_write_dict in tb_write_dict_lst:
                        tb_writer.add_scalar(tb_write_dict['tag'], tb_write_dict['scalar'], global_step=0)

                if _if_val:
                    msg, tb_write_dict_lst, report_dict = alg.test(val_fetcher, num_samples_val, if_baseline=False)

                    ckp_save_path = f'{ckp_save_path_pre}{done_niter}.pt'
                    last_ckp_save_path = f'{ckp_save_path_pre}last.pt'
                    msg = f'model is saved at [{ckp_save_path}] and [{last_ckp_save_path}].\n' + msg

                    perfrm = report_dict['ave_perfm']
                    lsb = report_dict['lsb']
                    if best_val_perfrm is None:  # no pre_val
                        best_val_perfrm = dict(iter_lst=[done_niter], perfrm=perfrm)
                        if_save_best = True
                    elif perfrm == best_val_perfrm['perfrm']:
                        best_val_perfrm['iter_lst'].append(done_niter)
                        if_save_best = False
                    else:
                        if_save_best = False
                        if (not lsb) and (perfrm > best_val_perfrm['perfrm']):
                            if_save_best = True
                            best_val_perfrm = dict(iter_lst=[done_niter], perfrm=perfrm)
                        elif lsb and (perfrm < best_val_perfrm['perfrm']):
                            if_save_best = True
                            best_val_perfrm = dict(iter_lst=[done_niter], perfrm=perfrm)
                    msg += f"\nbest iterations: [{best_val_perfrm['iter_lst']}]" \
                           f" | validation performance: [{best_val_perfrm['perfrm']:.4f}]"

                    alg.save_state(
                        ckp_save_path=ckp_save_path, idx_iter=done_niter, best_val_perfrm=best_val_perfrm,
                        if_sched=alg.if_sched
                    )  # save model
                    shutil.copy(ckp_save_path, last_ckp_save_path)  # copy as the last model
                    if if_save_best:
                        best_ckp_save_path = f'{ckp_save_path_pre}first_best.pt'
                        shutil.copy(ckp_save_path, best_ckp_save_path)  # copy as the best model

                    logger.info(msg)

                    for tb_write_dict in tb_write_dict_lst:
                        tb_writer.add_scalar(tb_write_dict['tag'], tb_write_dict['scalar'], done_niter)

                    for criteria in opts_dict['algorithm']['val']['criterion']:
                        plt_curve(log_path, criteria, log_path.parent, if_timestamp=False)

            # Show network structure
            
            if (rank == 0) and \
                    opts_dict['algorithm']['train']['if_show_graph'] and \
                    ((done_niter == inter_val) or (done_niter == alg.done_niter)):
                alg.add_graph(writer=tb_writer, data=train_data['lq'].cuda())
            

            # Determine whether to exit or not
            if done_niter >= niter and (not if_manually_stop):
                if_all_over = True  # no more training after the upper validation
                break  # leave the training data fetcher, but still in the training-validation loop

            # Figure out the current stage

            if_val_end_of_stage = False
            if done_niter < niter:
                stage_now = niter_name_lst[0]
                end_niter_this_stage = 0
                for is_, end_niter in enumerate(end_niter_lst):
                    if done_niter < end_niter:
                        stage_now = niter_name_lst[is_]
                        end_niter_this_stage = end_niter
                        if_val_end_of_stage = True if done_niter == (end_niter - 1) else False
                        break
            else:
                stage_now = niter_name_lst[-1]  # keep using the optim/scheduler of the last stage when training

            # Train one batch/iteration

            alg.set_train_mode()

            if_step = True if (done_niter + 1) % inter_step == 0 else False
            msg, tb_write_dict_lst, im_lst = alg.update_params(
                stage=stage_now,
                data=train_data,
                if_step=if_step,
                inter_step=inter_step,
                additional=additional,
            )

            done_niter += 1

            # Record & Display

            if done_niter % inter_print == 0 or if_val_end_of_stage:
                used_time = timer.record_and_get_inter()
                et = timer.get_sum_inter() / 3600
                timer.start_record()

                if done_niter < niter:
                    eta = used_time / inter_print * (niter - done_niter) / 3600
                    msg = (f'{stage_now} | iter [{done_niter}]/{end_niter_this_stage}/{niter} | '
                           f'eta/et: [{eta:.1f}]/{et:.1f} h | ' + msg)
                else:
                    msg = (f'never stop | iter [{done_niter}]/{niter} | ' + msg)
                logger.info(msg)

                if rank == 0:
                    for tb_write_dict in tb_write_dict_lst:
                        tb_writer.add_scalar(tb_write_dict['tag'], tb_write_dict['scalar'], done_niter)
                    for im_item in im_lst:
                        ims = im_lst[im_item]
                        tb_writer.add_images(im_item, ims, done_niter, dataformats='NCHW')

            train_data = train_fetcher.next()  # fetch the next batch

        # end of this epoch

        done_num_epochs += 1

    # end of all epochs

    if rank == 0:  # only rank0 conduct tests and record the best_val_perfrm
        timer.record_inter()
        tot_time = timer.get_sum_inter() / 3600
        msg = (
            f"best iterations: [{best_val_perfrm['iter_lst']}] | validation performance: [{best_val_perfrm['perfrm']:.4f}]\n"
            f'total time: [{tot_time:.1f}] h'
        )
        logger.info(msg)


if __name__ == '__main__':
    main()
