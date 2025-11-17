import os 

with open(r"D:\ZTB\Dataset\SPML_data_root\Potsdam\train.txt", 'r') as f:
    with open(r"D:\ZTB\Dataset\SPML_data_root\Potsdam\panoptic_train.txt", 'w') as fw:
        lines = f.readlines()
        for l in range(len(lines)):
            lines[l] = lines[l].strip('\n')
            parts = lines[l].split(' ')
            img_path = parts[0]
            scribble_path = parts[1]
            panoptic_label_path = scribble_path.split('/')[0]+'/segcls/'+scribble_path.split('/')[2]
            fw.write(img_path + ' '  + panoptic_label_path + ' ' + parts[2]+ ' '+ parts[3] + ' '+parts[4] +'\n')