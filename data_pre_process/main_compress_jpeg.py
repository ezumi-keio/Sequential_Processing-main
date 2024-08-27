import sys
from pathlib import Path
import os

import numpy as np
import cv2
import random
random.seed(45)


dataset = sys.argv[1]  # div2k
src_name = sys.argv[2]  # raw
tar_name = sys.argv[3]  # jpeg
quality = sys.argv[4]  # qf, e.g., 10, 20, 30, ...

current_dir = Path(__file__).resolve().parent
#src_im_dir = (current_dir / '..' / '..' / dataset / 'Flickr2K_HR' ).resolve()
src_im_dir = (current_dir / '..' / '..' / 'Urban100' / src_name ).resolve()
#src_im_dir = (current_dir / '..' / '..' / dataset / src_name ).resolve()
if quality == '-1':
    tar_im_dir = (current_dir / '..' / '..' / 'Dataset_synthesized' / tar_name).resolve()
    tar_im_dir_gt = (current_dir / '..' / '..' / 'Dataset_synthesized' / 'raw').resolve()
else:
    tar_im_dir = (current_dir / '..' / '..' / dataset /  ('jpeg4_qf' + quality)).resolve()
    tar_im_dir_gt = (current_dir / '..' / '..' / dataset / tar_name).resolve()
if not tar_im_dir.exists():
    tar_im_dir.mkdir(parents=True)
if not tar_im_dir_gt.exists():
    tar_im_dir_gt.mkdir(parents=True)

src_im_lst = sorted(src_im_dir.glob('*.png'))

tot = len(src_im_lst)
count = 0
for src_im_path in src_im_lst:
    count_subimg = 0
    src_im = cv2.imread(str(src_im_path))
    size_rawimg = src_im.shape
    src_im_name = src_im_path.name
    im_new_path = tar_im_dir / src_im_name

    count += 1
    print(f'{count} / {tot}: {im_new_path}')

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    if int(quality) == -1:
        src_subimg = []
        encode_param_subimg = []
        subimg_num = [ (size_rawimg[0]-512)//256 + 1, (size_rawimg[1]-512)//256 + 1]
        for h in range(subimg_num[0]):
            for w in range(subimg_num[1]):
                src_subimg.append (src_im[ 256*h:256*h+512, 256*w:256*w+512])
                encode_param_subimg.append( [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(0,100)] )

        for subimg in src_subimg:
            name = dataset + '_' + src_im_name.split('.')[0] + '_' + str(count_subimg) \
                + '_qf' + str(encode_param_subimg[count_subimg][1]) + '.png'
            name_gt = dataset + '_' + src_im_name.split('.')[0] + '_' + str(count_subimg) + '.png'
            im_new_path_sub = tar_im_dir / name
            im_new_path_sub_gt = tar_im_dir_gt / name_gt
            encimg = cv2.imencode('.jpg', subimg, encode_param_subimg[count_subimg])[1].tobytes()  # bytes class
            nparr = np.frombuffer(encimg, np.byte)
            img2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imwrite(str(im_new_path_sub_gt), subimg)
            cv2.imwrite(str(im_new_path_sub), img2)
            count_subimg += 1

    else:
        #new_path = 
        #command_ = f'ln -s {src_path} {new_path}'
        #os.system(command_)
        encimg = cv2.imencode('.jpg', src_im, encode_param)[1].tobytes()  # bytes class
        nparr = np.frombuffer(encimg, np.byte)
        img2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #encimg2 = cv2.imencode('.jpg', img2, [int(cv2.IMWRITE_JPEG_QUALITY), int(40)])[1].tobytes()  # bytes class
        #nparr2 = np.frombuffer(encimg2, np.byte)
        #img3 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
        cv2.imwrite(str(im_new_path), img2)
