import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction
import numpy as np
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from IPython import embed

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', default='work_dirs/res.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    # parser.add_argument(
    #     '--eval',
    #     type=str,
    #     nargs='+',
    #     default='mIoU',
    #     help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
    #     ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'aAcc','mFscore'],
        help='evaluation metrics, e.g., "mIoU", "mDice", "mPrecision", "mRecall", "aAcc"')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', default='./xiaorong/CAB',help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if 'None' in args.eval:
        args.eval = None
    if args.eval and args.format_only:

        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        if cfg.data.test.type == 'CityscapesDataset':
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
            ]
            cfg.data.test.pipeline[1].flip = True
        elif cfg.data.test.type == 'ADE20KDataset':
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.75, 0.875, 1.0, 1.125, 1.25
            ]
            cfg.data.test.pipeline[1].flip = True
        else:
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test.pipeline[1].flip = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    efficient_test = True #False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    # if rank == 0:
    #     if args.out:
    #         results = []
    #         for file_path in outputs:
    #             # 读取文件内容（根据你的文件格式，可能是 .json, .npy, .txt 等）
    #             if file_path.endswith('.json'):
    #                 result = mmcv.load(file_path)  # 如果是JSON文件
    #             elif file_path.endswith('.npy'):
    #                 result = np.load(file_path)  # 如果是NumPy文件
    #             else:
    #                 result = file_path  # 或者直接是路径
    #             # print(f"Result shape: {result.shape}, Min: {np.min(result)}, Max: {np.max(result)}")
    #             results.append(result)

    #         # 将所有读取的结果合并后保存
    #         mmcv.dump(results, args.out)
    #         # print(f"Results saved to {args.out}")

    #     kwargs = {} if args.eval_options is None else args.eval_options
    #     if args.format_only:
    #         dataset.format_results(results, **kwargs)  # 使用合并后的结果
    #     if args.eval:
    #         dataset.evaluate(results, args.eval, **kwargs)  # 使用合并后的结果
    if rank == 0:
            # 统一处理结果
            final_results = []
            if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], str):
                # 如果返回的是临时文件路径列表 (efficient_test 为 True)
                for file_path in outputs:
                    if file_path.endswith('.json'):
                        final_results.append(mmcv.load(file_path))
                    elif file_path.endswith('.npy'):
                        final_results.append(np.load(file_path))
                    else:
                        final_results.append(file_path)
            else:
                # 如果返回的就是结果数据本身
                final_results = outputs
    
            # 保存结果文件
            if args.out:
                print(f'\nWriting results to {args.out}')
                mmcv.dump(final_results, args.out)
    
            # 核心评估逻辑：确保这里只调用一次
            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset.format_results(final_results, **kwargs)
            if args.eval:
                # 这里会触发打印表格
                dataset.evaluate(final_results, args.eval, **kwargs)

if __name__ == '__main__':
    main()
