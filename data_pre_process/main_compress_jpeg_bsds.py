import sys
from pathlib import Path
import os

import numpy as np
import cv2
from scipy.io import loadmat
import random
random.seed(45)


dataset = sys.argv[1]  # div2k
src_name = sys.argv[2]  # raw
tar_name = sys.argv[3]  # jpeg
quality = sys.argv[4]  # qf, e.g., 10, 20, 30, ...

current_dir = Path(__file__).resolve().parent
src_im_dir = (current_dir / '..' / '..' / dataset / src_name / 'images').resolve()
tar_im_dir = (current_dir / '..' / '..' / dataset / src_name / ('jpeg_qf' + quality)).resolve()
if not tar_im_dir.exists():
    tar_im_dir.mkdir(parents=True)

src_im_lst = sorted(src_im_dir.glob('*.jpg'))

tot = len(src_im_lst)
count = 0
for src_im_path in src_im_lst:
    src_im = cv2.imread(str(src_im_path))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    encimg = cv2.imencode('.jpg', src_im, encode_param)[1].tobytes()  # bytes class
    nparr = np.frombuffer(encimg, np.byte)
    img2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    src_im_name = src_im_path.stem + '.jpg'
    im_new_path = tar_im_dir / src_im_name
    print(im_new_path)
    cv2.imwrite(str(im_new_path), img2)
