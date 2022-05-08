import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from PIL import Image


def create_dior(root, dest):
    splits_path = os.path.join(root, 'ImageSets', 'Main')
    annotations_path = os.path.join(root, 'Annotations', 'Horizontal Bounding Boxes')
    trainval_path = os.path.join(root, 'JPEGImages-trainval-002', 'JPEGImages-trainval')
    test_path = os.path.join(root, 'JPEGImages-test-001', 'JPEGImages-test')

    splits = {}
    with open(splits_path + '/train.txt') as f:
        for prefix in f.read().splitlines():
            splits[prefix] = 'train'
    
    with open(splits_path + '/val.txt') as f:
        for prefix in f.read().splitlines():
            splits[prefix] = 'val'
    
    with open(splits_path + '/test.txt') as f:
        for prefix in f.read().splitlines():
            splits[prefix] = 'test'
    
    annotations = defaultdict(list)
    
    for annotation in sorted(os.listdir(annotations_path)):
        file_name = os.path.join(annotations_path, annotation)
        tree = ET.parse(file_name)
        for obj in tree.findall('object'):
            

            bbox = obj.find('bndbox')
            if bbox is None:
                bbox = obj.find('robndbox')

            xmin = int(bbox.find('xmin').text)
            xmax = int(bbox.find('xmax').text)
            ymin = int(bbox.find('ymin').text)
            ymax = int(bbox.find('ymax').text)

            if abs(xmax - xmin) >= 128 and abs(ymax - ymin) >= 128:
                bbox = [xmin, ymin, xmax, ymax]
                class_name = obj.find('name').text
                if class_name != 'vehicle':
                    file_name = tree.find('filename').text
                    prefix = file_name.split('.')[0]
                    if splits[prefix] != 'test':
                        file_name = os.path.join(trainval_path, file_name)
                    else:
                        file_name = os.path.join(test_path, file_name)
                    annotations[file_name].append({'class': class_name, 'bbox': bbox, 'prefix': prefix})

    os.makedirs(dest + '/train', exist_ok=True)
    os.makedirs(dest + '/val', exist_ok=True)
    os.makedirs(dest + '/test', exist_ok=True)
    for file_name in annotations.keys():
        image = Image.open(file_name)
        for i, obj in enumerate(annotations[file_name]):
            o = image.crop(obj['bbox'])
            new_file_name = obj['prefix'] + '_' + obj['class'] + '_' + str(i) + file_name[-4:]
            new_file_name = os.path.join(dest, splits[obj['prefix']], new_file_name)
            o.save(new_file_name)
