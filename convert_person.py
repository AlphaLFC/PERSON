# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 22:39:30 2016

@author: super
"""


import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import xml.etree.ElementTree as ET
import fortranformat as ff
from shutil import copyfile
from PIL import Image


CLASSES = ['head', 'hat'] + \
          ['top_'+str(cls) for cls in xrange(6)] + \
          ['down_'+str(cls) for cls in xrange(5)] + \
          ['shoes_'+str(cls) for cls in xrange(5)] + \
          ['bag_'+str(cls) for cls in xrange(5)]


def convert_gender_hairlen_annotation(xml_file):
    filename = os.path.join('TRAIN',
                            'ANNOTATIONS_TRAIN', xml_file)
    tree = ET.parse(filename)
    gender = tree.find('gender')
    hairlen = tree.find('hairlength')
    gender_path = 'TRAIN/genders'
    hairlen_path = 'TRAIN/hairs'
    if not os.path.exists(gender_path):
        os.mkdir(gender_path)
        os.mkdir(os.path.join(gender_path, '0'))
        os.mkdir(os.path.join(gender_path, '1'))
        os.mkdir(os.path.join(gender_path, '2'))
    if not os.path.exists(hairlen_path):
        os.mkdir(hairlen_path)
        os.mkdir(os.path.join(hairlen_path, '0'))
        os.mkdir(os.path.join(hairlen_path, '1'))
        os.mkdir(os.path.join(hairlen_path, '2'))
    copyfile(os.path.join('TRAIN',
                          'IMAGES_TRAIN',
                          tree.find('filename').text),
             os.path.join(gender_path, gender.text,
                          tree.find('filename').text))
    copyfile(os.path.join('TRAIN',
                          'IMAGES_TRAIN',
                          tree.find('filename').text),
             os.path.join(hairlen_path, hairlen.text,
                          tree.find('filename').text))


def convert_bbox_annotation(xml_file, sets='TRAIN'):
    filename = os.path.join(sets,
                            'ANNOTATIONS_'+sets, xml_file)
    tree = ET.parse(filename)
    subs = tree.findall('subcomponent')
    new_fmt_annos = []
    img_infos = []
    for idx, sub in enumerate(subs):
        if sub.find('name').text == 'bag':
            cls = 'bag'
            for id_num in list(sub)[1:]:
                category_text = id_num.find('category').text
                color_text = id_num.find('color').text
                if category_text != 'NULL':
                    fine_cls = sub.find('name').text + \
                               ' ' + category_text + \
                               ' ' + color_text
                    img_infos.append(fine_cls)
                    bbox = id_num.find('bndbox')
                    x1 = float(bbox.find('xmin').text)
                    y1 = float(bbox.find('ymin').text)
                    x2 = float(bbox.find('xmax').text)
                    y2 = float(bbox.find('ymax').text)
                    if x1 < x2 and y1 < y2:
                        new_fmt_annos.append([x1, y1, x2, y2, cls])
        else:
            if sub.find('category') is not None and \
                    sub.find('category').text != 'NULL':
                fine_cls = sub.find('name').text + \
                           ' ' + sub.find('category').text + \
                           ' ' + sub.find('color').text
            else:
                fine_cls = sub.find('name').text
            cls = sub.find('name').text
            img_infos.append(fine_cls)
            bbox = sub.find('bndbox')
            if bbox is not None:
                if bbox.find('xmin').text != 'NULL':
                    x1 = float(bbox.find('xmin').text)
                    y1 = float(bbox.find('ymin').text)
                    x2 = float(bbox.find('xmax').text)
                    y2 = float(bbox.find('ymax').text)
                    if x1 < x2 and y1 < y2:
                        new_fmt_annos.append([x1, y1, x2, y2, cls])
            if sub.find('name').text == 'shoes':
                if sub.find('xmin_l').text != 'NULL':
                    x1_l = float(sub.find('xmin_l').text)
                    y1_l = float(sub.find('ymin_l').text)
                    x2_l = float(sub.find('xmax_l').text)
                    y2_l = float(sub.find('ymax_l').text)
                    if x1_l < x2_l and y1_l < y2_l:
                        new_fmt_annos.append([x1_l, y1_l, x2_l, y2_l, cls])
                if sub.find('xmin_r').text != 'NULL':
                    x1_r = float(sub.find('xmin_r').text)
                    y1_r = float(sub.find('ymin_r').text)
                    x2_r = float(sub.find('xmax_r').text)
                    y2_r = float(sub.find('ymax_r').text)
                    if x1_r < x2_r and y1_r < y2_r:
                        new_fmt_annos.append([x1_r, y1_r, x2_r, y2_r, cls])
    out_format = ff.FortranRecordWriter('(4(I5, 1X), A11)')
    if not os.path.exists(sets+'/bndboxes'):
        os.mkdir(sets+'/bndboxes')
    if not os.path.exists(sets+'/infos'):
        os.mkdir(sets+'/infos')
    label_file = open(os.path.join(sets,
                                   'bndboxes',
                                   xml_file.split('.')[0]+'.txt'), 'w')
    info_file = open(os.path.join(sets,
                                  'infos',
                                  xml_file.split('.')[0]+'.info'), 'w')
    for label in new_fmt_annos:
        out_string = out_format.write(label)
        print>>label_file, out_string
    for info in img_infos:
        print>>info_file, info
    label_file.close()
    info_file.close()


def crop_subcomponent_by_category_and_color(xml_file):
    filename = os.path.join('TRAIN',
                            'ANNOTATIONS_TRAIN', xml_file)
    imagename = os.path.join('TRAIN',
                             'IMAGES_TRAIN', xml_file.split('.')[0]+'.jpg')
    tree = ET.parse(filename)
    im = Image.open(imagename)
    subs = tree.findall('subcomponent')
    sub_path = os.path.join('TRAIN', 'subcomponets')
    color_path = os.path.join('TRAIN', 'colors')
    if not os.path.exists(sub_path):
        os.mkdir(sub_path)
    if not os.path.exists(color_path):
        os.mkdir(color_path)
    for idx, sub in enumerate(subs):
        cls = sub.find('name').text
        cls_path = os.path.join(sub_path, cls)
        if cls == 'bag':
            for id_num in list(sub)[1:]:
                id_name = str(id_num).split(' ')[1].replace('_', '').replace("'", '')
                category_text = id_num.find('category').text
                color_text = id_num.find('color').text
                if category_text != 'NULL':
                    fine_cls = sub.find('name').text + \
                               '_' + id_name + \
                               '_' + category_text + \
                               '_' + color_text
                    bbox = id_num.find('bndbox')
                    x1 = int(bbox.find('xmin').text)
                    y1 = int(bbox.find('ymin').text)
                    x2 = int(bbox.find('xmax').text)
                    y2 = int(bbox.find('ymax').text)
                    bboxcrop = (x1, y1, x2, y2)
                    if x1 < x2 < im.size[0] and y1 < y2 < im.size[1]:
                        im_crop = im.crop(bboxcrop)
                        im_crop_flip = im_crop.transpose(Image.FLIP_LEFT_RIGHT)
                        cls_sub_path = os.path.join(cls_path, category_text)
                        color_sub_path = os.path.join(color_path, color_text)
                        if not os.path.exists(cls_path):
                            os.mkdir(cls_path)
                        if not os.path.exists(cls_sub_path):
                            os.mkdir(cls_sub_path)
                        if not os.path.exists(color_sub_path):
                            os.mkdir(color_sub_path)
                        cls_fname = os.path.join(cls_sub_path,
                                                 xml_file.split('.')[0] +
                                                 '_' + fine_cls + '.jpg')
                        color_fname = os.path.join(color_sub_path,
                                                   xml_file.split('.')[0] +
                                                   '_' + fine_cls + '.jpg')
                        cls_fname_flip = os.path.join(cls_sub_path,
                                xml_file.split('.')[0] + '_flipped' +
                                '_' + fine_cls + '.jpg')
                        color_fname_flip = os.path.join(color_sub_path,
                                xml_file.split('.')[0] + '_flipped' +
                                '_' + fine_cls + '.jpg')
                        im_crop.save(cls_fname)
                        im_crop.save(color_fname)
                        im_crop_flip.save(cls_fname_flip)
                        im_crop_flip.save(color_fname_flip)
        elif cls == 'shoes':
            category_text = sub.find('category').text
            color_text = sub.find('color').text
            if category_text != 'NULL':
                fine_cls_l = sub.find('name').text + \
                             '_l_' + category_text + \
                             '_' + color_text
                fine_cls_r = sub.find('name').text + \
                             '_r_' + category_text + \
                             '_' + color_text
                if sub.find('xmin_l').text != 'NULL':
                    x1_l = int(sub.find('xmin_l').text)
                    y1_l = int(sub.find('ymin_l').text)
                    x2_l = int(sub.find('xmax_l').text)
                    y2_l = int(sub.find('ymax_l').text)
                    bboxcrop_l = (x1_l, y1_l, x2_l, y2_l)
                    if x1_l < x2_l < im.size[0] and y1_l < y2_l < im.size[1]:
                        im_crop_l = im.crop(bboxcrop_l)
                        im_crop_flip_l = im_crop_l.transpose(Image.FLIP_LEFT_RIGHT)
                        cls_sub_path = os.path.join(cls_path, category_text)
                        color_sub_path = os.path.join(color_path, color_text)
                        if not os.path.exists(cls_path):
                                os.mkdir(cls_path)
                        if not os.path.exists(cls_sub_path):
                            os.mkdir(cls_sub_path)
                        if not os.path.exists(color_sub_path):
                            os.mkdir(color_sub_path)
                        cls_fname_l = os.path.join(cls_sub_path,
                                                   xml_file.split('.')[0] +
                                                   '_' + fine_cls_l + '.jpg')
                        color_fname_l = os.path.join(color_sub_path,
                                                     xml_file.split('.')[0] +
                                                     '_' + fine_cls_l + '.jpg')
                        cls_fname_flip_l = os.path.join(cls_sub_path,
                                xml_file.split('.')[0] + '_flipped' +
                                '_' + fine_cls_l + '.jpg')
                        color_fname_flip_l = os.path.join(color_sub_path,
                                xml_file.split('.')[0] + '_flipped' +
                                '_' + fine_cls_l + '.jpg')
                        im_crop_l.save(cls_fname_l)
                        im_crop_l.save(color_fname_l)
                        im_crop_flip_l.save(cls_fname_flip_l)
                        im_crop_flip_l.save(color_fname_flip_l)
                if sub.find('xmin_r').text != 'NULL':
                    x1_r = int(sub.find('xmin_r').text)
                    y1_r = int(sub.find('ymin_r').text)
                    x2_r = int(sub.find('xmax_r').text)
                    y2_r = int(sub.find('ymax_r').text)
                    bboxcrop_r = (x1_r, y1_r, x2_r, y2_r)
                    if x1_r < x2_r < im.size[0] and y1_r < y2_r < im.size[1]:
                        im_crop_r = im.crop(bboxcrop_r)
                        im_crop_flip_r = im_crop_r.transpose(Image.FLIP_LEFT_RIGHT)
                        cls_sub_path = os.path.join(cls_path, category_text)
                        color_sub_path = os.path.join(color_path, color_text)
                        if not os.path.exists(cls_path):
                                os.mkdir(cls_path)
                        if not os.path.exists(cls_sub_path):
                            os.mkdir(cls_sub_path)
                        if not os.path.exists(color_sub_path):
                            os.mkdir(color_sub_path)
                        cls_fname_r = os.path.join(cls_sub_path,
                                                   xml_file.split('.')[0] +
                                                   '_' + fine_cls_r + '.jpg')
                        color_fname_r = os.path.join(color_sub_path,
                                                     xml_file.split('.')[0] +
                                                     '_' + fine_cls_r + '.jpg')
                        cls_fname_flip_r = os.path.join(cls_sub_path,
                                xml_file.split('.')[0] + '_flipped' +
                                '_' + fine_cls_r + '.jpg')
                        color_fname_flip_r = os.path.join(color_sub_path,
                                xml_file.split('.')[0] + '_flipped' +
                                '_' + fine_cls_r + '.jpg')
                        im_crop_r.save(cls_fname_r)
                        im_crop_r.save(color_fname_r)
                        im_crop_flip_r.save(cls_fname_flip_r)
                        im_crop_flip_r.save(color_fname_flip_r)
        elif sub.find('category') is not None:
            category_text = sub.find('category').text
            color_text = sub.find('color').text
            if category_text != 'NULL':
                fine_cls = sub.find('name').text + \
                           '_' + category_text + \
                           '_' + color_text
                bbox = sub.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                bboxcrop = (x1, y1, x2, y2)
                if x1 < x2 < im.size[0] and y1 < y2 < im.size[1]:
                    im_crop = im.crop(bboxcrop)
                    im_crop_flip = im_crop.transpose(Image.FLIP_LEFT_RIGHT)
                    cls_sub_path = os.path.join(cls_path, category_text)
                    color_sub_path = os.path.join(color_path, color_text)
                    if not os.path.exists(cls_path):
                            os.mkdir(cls_path)
                    if not os.path.exists(cls_sub_path):
                        os.mkdir(cls_sub_path)
                    if not os.path.exists(color_sub_path):
                        os.mkdir(color_sub_path)
                    cls_fname = os.path.join(cls_sub_path,
                                             xml_file.split('.')[0] +
                                             '_' + fine_cls + '.jpg')
                    color_fname = os.path.join(color_sub_path,
                                               xml_file.split('.')[0] +
                                               '_' + fine_cls + '.jpg')
                    cls_fname_flip = os.path.join(cls_sub_path,
                                xml_file.split('.')[0] + '_flipped' +
                                '_' + fine_cls + '.jpg')
                    color_fname_flip = os.path.join(color_sub_path,
                                xml_file.split('.')[0] + '_flipped' +
                                '_' + fine_cls + '.jpg')
                    im_crop.save(cls_fname)
                    im_crop.save(color_fname)
                    im_crop_flip.save(cls_fname_flip)
                    im_crop_flip.save(color_fname_flip)

if __name__ == '__main__':
    train_lists = os.listdir('TRAIN/ANNOTATIONS_TRAIN/')
    for train_list in train_lists[14919:]:
        # convert_bbox_annotation(train_list)
        try:
            crop_subcomponent_by_category_and_color(train_list)
        except:
            print train_list