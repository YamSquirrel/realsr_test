#把图片分成多个256图片块，

from PIL import Image
import numpy as np
import os.path as osp
import glob
import os
import argparse
import yaml

parser = argparse.ArgumentParser(description='create a dataset')
parser.add_argument('--dataset', default='df2k', type=str, help='selecting different datasets')
parser.add_argument('--artifacts', default='', type=str, help='selecting different artifacts type')
parser.add_argument('--cleanup_factor', default=2, type=int, help='downscaling factor for image cleanup')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[4], help='super resolution upscale factor')
opt = parser.parse_args()

# define input and target directories
# with open('./preprocess/paths.yml', 'r') as stream:
#    PATHS = yaml.load(stream)
with open('paths_test.yml', 'r') as stream:
    PATHS = yaml.load(stream,Loader=yaml.FullLoader)

def noise_patch(rgb_img, sp, max_var, min_mean):
    img = rgb_img.convert('L')
    rgb_img = np.array(rgb_img)
    img = np.array(img)
    w, h = img.shape
    collect_patchs = []
    for i in range(0, w - sp + 1, sp):
        for j in range(0, h - sp + 1, sp):
            patch = img[i:i + sp, j:j + sp]
            var_global = np.var(patch)
            mean_global = np.mean(patch)
            #print('i=',i,'j=',j,'sp=',sp,'var_global=',var_global,'mean_global=',mean_global,'max_var=',max_var,'min_mean=',min_mean)#Debug#
            #if var_global < max_var and mean_global > min_mean:
            if True:
                rgb_patch = rgb_img[i:i + sp, j:j + sp, :]
                collect_patchs.append(rgb_patch)
    return collect_patchs


if __name__ == '__main__':

    if opt.dataset == 'df2k':
        img_dir = PATHS[opt.dataset][opt.artifacts]['source']
        noise_dir = PATHS['datasets']['df2k'] + '/Corrupted_noise'
        sp = 256
        max_var = 20
        min_mean = 0
    else:
        img_dir = PATHS[opt.dataset][opt.artifacts]['hr']['train']
        noise_dir = PATHS['datasets']['dped'] + '/DPEDiphone_noise'
        sp = 256
        max_var = 20
        min_mean = 50
    
    assert os.path.exists(noise_dir)
    if not os.path.exists(noise_dir):
        os.mkdir(noise_dir)

    img_paths = sorted(glob.glob(osp.join(img_dir, '*.png')))
    print('imge path:',img_paths)#Debug#
    cnt = 0
    for path in img_paths:
        img_name = osp.splitext(osp.basename(path))[0]
        img = Image.open(path).convert('RGB')
        #print('ImageInfo:',img,'ImageName:',img_name)#Debug#
        patchs = noise_patch(img, sp, max_var, min_mean)
        for idx, patch in enumerate(patchs):
            save_path = osp.join(noise_dir, '{}_{:03}.png'.format(img_name, idx))
            cnt += 1
            print('collect:', cnt, save_path)
            Image.fromarray(patch).save(save_path)
            
