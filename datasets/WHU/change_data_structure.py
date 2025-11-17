import os
import numpy as np
import scipy.io as sio
from PIL import Image

def to_unix_path(full_path, base_dir):
    """Return path relative to base_dir with forward slashes.
    e.g. full_path='D:\\...\\SPML_data_root\\WHU\\JPEGImages\\a.png',
         base_dir='D:\\...\\SPML_data_root' ->
         'WHU/JPEGImages/a.png'
    """
    rel = os.path.relpath(full_path, base_dir)
    # use os.path.sep for portability, then normalize to '/'
    return rel.replace(os.path.sep, '/')

if __name__ == '__main__':
    source_root = r'D:\ZTB\Dataset\Potsdam_binary'
    target_root_root = r'D:\ZTB\Dataset\SPML_data_root'
    target_root = r'D:\ZTB\Dataset\SPML_data_root\Potsdam'

    target_image_dir = os.path.join(target_root, 'JPEGImages')
    target_scribble_dir = os.path.join(target_root,'scribble')
    target_hed_dir = os.path.join(target_root,'hed')
    target_semantic_dir = os.path.join(target_root,'segcls')
    target_edge_dir = os.path.join(target_root,'edge')
    target_gray_dir = os.path.join(target_root,'gray')
    os.makedirs(target_image_dir, exist_ok=True)    
    os.makedirs(target_scribble_dir, exist_ok=True)
    os.makedirs(target_hed_dir, exist_ok=True)
    os.makedirs(target_semantic_dir, exist_ok=True)
    os.makedirs(target_edge_dir, exist_ok=True)
    os.makedirs(target_gray_dir, exist_ok=True)

    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(source_root, split)
        img_this_split = os.path.join(split_dir, 'image')
        hed_this_split = os.path.join(split_dir, 'UCM_label')
        edge_this_split = os.path.join(split_dir, 'edge_MuGE')
        label_this_split = os.path.join(split_dir, 'label')
        scribble_this_split = os.path.join(split_dir, 'scribble')
        gray_this_split = os.path.join(split_dir, 'blur_image')
        mat_files = [f for f in os.listdir(split_dir) if f.endswith('.mat')]

        if os.path.exists(os.path.join(target_root, f'{split}.txt')):
            os.remove(os.path.join(target_root, f'{split}.txt'))

        for img in os.listdir(img_this_split):
            print(f'Processing {split} image: {img}')
            if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.tif'):
                base_name = os.path.splitext(img)[0]
                img_path = os.path.join(img_this_split, img)
                target_img_path = os.path.join(target_image_dir, f'{split}_{base_name}.png')
                Image.open(img_path).convert('RGB').save(target_img_path)
            semantic_label_path = os.path.join(label_this_split, f'{base_name}.png')
            semantic_label = np.array(Image.open(semantic_label_path).convert('L'))
            semantic_label[semantic_label==255] = 1  # Convert 255 to 1 for building class
            target_semantic_label = Image.fromarray(semantic_label.astype(np.uint8))
            target_semantic_label_path = os.path.join(target_semantic_dir, f'{split}_{base_name}.png')
            target_semantic_label.save(target_semantic_label_path)
            
            hed_path = os.path.join(hed_this_split, f'{base_name}.mat')
            hed_data = sio.loadmat(hed_path)['labels'].astype(np.uint8)
            hed_image = Image.fromarray(hed_data).convert('L')
            target_hed_path = os.path.join(target_hed_dir, f'{split}_{base_name}.png')
            hed_image.save(target_hed_path)

            with open(os.path.join(target_root, f'{split}.txt'), 'a') as f:
                f.write(to_unix_path(target_img_path, target_root_root) + ' ')
                if not split == 'train':
                    f.write(to_unix_path(target_semantic_label_path, target_root_root) + ' ')
                    f.write(to_unix_path(target_hed_path, target_root_root) + '\n')

            if split == 'train':
                scribble_path = os.path.join(scribble_this_split, f'{base_name}.png')
                scribble = np.array(Image.open(scribble_path).convert('L'))
                scribble[scribble==255] = 1
                scribble[scribble==128] = 2 
                target_scribble = Image.fromarray(scribble.astype(np.uint8))
                target_scribble_path = os.path.join(target_scribble_dir, f'{split}_{base_name}.png')
                target_scribble.save(target_scribble_path)  

                edge_path = os.path.join(edge_this_split, f'{base_name}.png')
                edge_image = Image.open(edge_path).convert('L')
                target_edge_path = os.path.join(target_edge_dir, f'{split}_{base_name}.png')
                edge_image.save(target_edge_path)

                gray_path = os.path.join(gray_this_split, f'{base_name}.png')
                gray_image = Image.open(gray_path).convert('L')
                target_gray_path = os.path.join(target_gray_dir, f'{split}_{base_name}.png')
                gray_image.save(target_gray_path)

                with open(os.path.join(target_root, f'{split}.txt'), 'a') as f:
                    f.write(to_unix_path(target_scribble_path, target_root_root) + ' ')
                    f.write(to_unix_path(target_hed_path, target_root_root) + ' ')
                    f.write(to_unix_path(target_edge_path, target_root_root) + ' ')
                    f.write(to_unix_path(target_gray_path, target_root_root) + '\n')