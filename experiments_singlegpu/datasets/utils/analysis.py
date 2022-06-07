import os
import csv
import json

import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Union
from PIL import Image
from experiments_singlegpu.datasets.SocialProfilePictures import SocialProfilePictures
from experiments_singlegpu.datasets.EMOTIC_custom import EMOTIC
from experiments_singlegpu.datasets.SUN397_custom import SUN397
from shapely.geometry import Polygon
import math
import matplotlib.pyplot as plt
import numpy as np


class DatasetTriplet(Dataset):
    """
        Extends Dataset 
        It will return image, target and indices of the images. 
        This is useful to investigate when using data loader with suffle and order of metadata is lost
    """
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(os.path.join(self.root, self.metadata[idx]['img_folder'], self.metadata[idx]['img_name']))

        if img.mode == "1" or img.mode == "L" or img.mode == "P" or img.mode == "RGBA": # if gray-scale image convert into rgb
            img = img.convert('RGB')

        target = self.targets[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target, idx


def calculate_EMOTIC_people_perc_from_SPP(dataset_folder, save_folder, wrong_predictions_file, use_yolo=False, adjust_ground_truth=False):

    if use_yolo:
        assert adjust_ground_truth == False, "If you use yolo, you don't need to adjust bboxes"
    
    wrong_images_paths = {}
    wrong_images_people_perc_total = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}
    wrong_images_people_perc_selfie = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}
    wrong_images_people_perc_scenes = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}
    wrong_images_people_perc_mscoco_scenes = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}
    wrong_images_people_perc_mscoco_selfies = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}
    wrong_images_people_perc_emodb_scenes = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}
    wrong_images_people_perc_emodb_selfies = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}
    
    counters = {"mscoco_all": 0, "mscoco_scenes": 0, "mscoco_selfies": 0, "emodb_all": 0, "emodb_scenes": 0, "emodb_selfies": 0,}

    scenes = ["shopping_and_dining", "workplace", "home_or_hotel",
                    "transportation", "sports_and_leisure", "cultural",
                    "water_ice_snow", "mountains_hills_desert_sky",
                    "forest_field_jungle", "man-made_elements",
                    "transportation", "cultural_or_historical_building_place",
                    "sportsfields_parks_leisure_spaces", "industrial_and_construction",
                    "houses_cabins_gardens_and_farms", "commercial_buildings"]

    with open(wrong_predictions_file, "r") as f:
        lines = f.readlines()

        for line in lines:
            img_path = line.split(":")[1].split(",")[0].replace("'","").strip()
            
            folder = ""
            wrong_prediction = ""
            
            
            if "mscoco" in line:
                folder = "mscoco"
                if "selfie" in line:
                    counters["mscoco_selfies"] += 1
                    counters["mscoco_all"] += 1
                    wrong_prediction = "selfie"
                    wrong_images_paths[img_path] = {"folder": folder, "wrong_prediction": wrong_prediction}
                else:
                    for s in scenes:
                        if s in line:
                            counters["mscoco_scenes"] += 1
                            counters["mscoco_all"] += 1
                            wrong_prediction = "scenes"
                            wrong_images_paths[img_path] = {"folder": folder, "wrong_prediction": wrong_prediction}
                            break
            if "emodb" in line:
                folder = "emodb"
                if "selfie" in line:
                    counters["emodb_selfies"] += 1
                    counters["emodb_all"] += 1
                    wrong_prediction = "selfie"
                    wrong_images_paths[img_path] = {"folder": folder, "wrong_prediction": wrong_prediction}
                else:
                    for s in scenes:
                        if s in line:
                            counters["emodb_scenes"] += 1
                            counters["emodb_all"] += 1
                            wrong_prediction = "scenes"
                            wrong_images_paths[img_path] = {"folder": folder, "wrong_prediction": wrong_prediction}
                            break
            

    print(counters)

    dataset = EMOTIC(dataset_folder, split=["train", "test"], aspect_ratio_threshold=2.33, dim_threshold=225)

    print("Dataset is big: {}".format(len(dataset)))
    # print(dataset.metadata[0])
    # print(dataset.metadata[0]['target']['extra_annotations'])
    # extra_annotations = dataset.metadata[0]['target']['extra_annotations']

    # print(len(extra_annotations))
    # print(extra_annotations[0]['bbox'])

    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True) if use_yolo else None
    overflow_bboxes = 0
    overflow_bboxes_images = []
    images_people_perc_total = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}
    images_people_perc_mscoco = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}
    images_people_perc_emodb = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}
    images_with_new_g_truth = {}
    images_with_people_perc = {}
    wrong_predictions_with_people_perc = {}

    for i, (img, target) in enumerate(dataset):
        W, H = img.size
        # print(W)
        # print(H)
        img_area = W*H
        bboxes = []
        img_path = 'EMOTIC/data/' + dataset.metadata[i]['img_folder'] + "/" + dataset.metadata[i]['img_name']

        if use_yolo:
            
            bboxes_annotations = dataset.metadata[i]['target']['extra_annotations']
            images_with_new_g_truth[img_path] = {"image_dims": {"w": W, "h": H}, "original_bboxes": [], "adjusted_bboxes": []}
            for j in range(len(bboxes_annotations)):
                # print(extra_annotations[j]['bbox'])
                
                x0 = float(bboxes_annotations[j]['bbox'][0])
                y0 = float(bboxes_annotations[j]['bbox'][1])
                w = float(bboxes_annotations[j]['bbox'][2])
                h = float(bboxes_annotations[j]['bbox'][3])
                # print(x0, y0, w, h)
                
                images_with_new_g_truth[img_path]["original_bboxes"].append({"x0": x0, "y0": y0, "w": w, "h": h})
            
            results = yolo(img)
            results.xyxy[0]  
            predicted_bboxes = results.pandas().xyxy[0]
            for j, r in enumerate(predicted_bboxes['name']):
                if r == 'person':
                    
                    x0 = round(float(predicted_bboxes['xmin'][j]), 1)
                    y0 = round(float(predicted_bboxes['ymin'][j]), 1)
                    x1 = round(float(predicted_bboxes['xmax'][j]), 1)
                    y1 = round(float(predicted_bboxes['ymax'][j]), 1)
                    
                    images_with_new_g_truth[img_path]["adjusted_bboxes"].append({"x0": x0, "y0": y0, "w": x1-x0, "h": y1-y0})

                    bbox = Polygon([(x0,y0), (x1, y0), (x1, y1), (x0, y1)])
                    bboxes.append(bbox)
        else:
            bboxes_annotations = dataset.metadata[i]['target']['extra_annotations']
            # print(dataset.metadata[i]['img_folder']+"/"+dataset.metadata[i]['img_name'])
            images_with_new_g_truth[img_path] = {"image_dims": {"w": W, "h": H}, "original_bboxes": [], "adjusted_bboxes": []}
            for j in range(len(bboxes_annotations)):
                # print(extra_annotations[j]['bbox'])
                
                x0 = float(bboxes_annotations[j]['bbox'][0])
                y0 = float(bboxes_annotations[j]['bbox'][1])
                w = float(bboxes_annotations[j]['bbox'][2])
                h = float(bboxes_annotations[j]['bbox'][3])
                # print(x0, y0, w, h)
                
                images_with_new_g_truth[img_path]["original_bboxes"].append({"x0": x0, "y0": y0, "w": w, "h": h})
                if adjust_ground_truth:
                    x0 = abs(x0)
                    y0 = abs(y0)
                    if x0+w > W:
                        w = float(W-x0)
                    if y0+h > H:
                        h = float(H-y0)
                    images_with_new_g_truth[img_path]["adjusted_bboxes"].append({"x0": x0, "y0": y0, "w": w, "h": h})

                bbox = Polygon([(x0,y0), (x0+w, y0), (x0+w, y0+h), (x0, y0+h)])
                bboxes.append(bbox)
        
       
        people_area = Polygon([(0,0), (0,0), (0,0), (0,0),])
        
        for b in bboxes:
            people_area = people_area.union(b)

        people_area = people_area.area

        # if people_area > img_area:
        #     overflow_bboxes += 1
            
        #     # if 'mscoco' in dataset.metadata[i]['img_folder']:
        #     #     print(dataset.metadata[i]['img_folder']+"/"+dataset.metadata[i]['img_name'])
        #     #     print(W)
        #     #     print(H)
        #     #     for j in range(len(extra_annotations)):
        #     #         print(extra_annotations[j]['bbox'])
        #     #     print(people_area)
        #     #     print(bboxes[0].bounds)
        #     #     exit()
            
        #     people_area = img_area
        #     overflow_bboxes_images.append(img_path)


        perc = math.ceil(math.ceil(people_area/img_area*100) / 10) * 10
        
        if perc > 100:
            perc = 110
        if perc not in images_people_perc_total:
            images_people_perc_total[perc] = 1
        else:
            images_people_perc_total[perc] += 1
        

        if 'mscoco' in img_path:
            images_people_perc_mscoco[perc] += 1
        elif 'emodb' in img_path:
            images_people_perc_emodb[perc] += 1
        
        if img_path in wrong_images_paths.keys():
            wrong_predictions_with_people_perc[img_path] = {"people_percentage": perc, "wrong_prediction": wrong_images_paths[img_path]["wrong_prediction"]}
            wrong_images_people_perc_total[perc] += 1
            if wrong_images_paths[img_path]["wrong_prediction"] == "selfie":
                wrong_images_people_perc_selfie[perc] += 1
                if "mscoco" in img_path:
                    wrong_images_people_perc_mscoco_selfies[perc] += 1
                elif "emodb" in img_path:
                    wrong_images_people_perc_emodb_selfies[perc] += 1
            elif wrong_images_paths[img_path]["wrong_prediction"] == "scenes":
                wrong_images_people_perc_scenes[perc] += 1
                if "mscoco" in img_path:
                    wrong_images_people_perc_mscoco_scenes[perc] += 1
                elif "emodb" in img_path:
                    wrong_images_people_perc_emodb_scenes[perc] += 1
        images_with_people_perc[img_path] = {"img_area": W*H, "people_area": people_area, "people_percentage": perc}
    
    with open('{}/images_with_people_perc{}{}.json'.format(save_folder, "_with_yolo" if use_yolo else "", "_adjusted_g_truth" if adjust_ground_truth else ""), 'w') as outfile:
        json.dump(images_with_people_perc, outfile)

    with open('{}/wrong_predictions_with_people_perc{}{}.json'.format(save_folder, "_with_yolo" if use_yolo else "", "_adjusted_g_truth" if adjust_ground_truth else ""), 'w') as outfile:
        json.dump(wrong_predictions_with_people_perc, outfile)

    if adjust_ground_truth or use_yolo:
        with open('{}/images_with_new_bboxes{}.json'.format(save_folder, "_with_yolo" if use_yolo else "",), 'w') as outfile:
            json.dump(images_with_new_g_truth, outfile)
        

    images_people_perc_total_sorted = {}
    for key, value in sorted(images_people_perc_total.items(), key=lambda item: item[0]):
        images_people_perc_total_sorted[key] = value
    images_people_perc_mscoco_sorted = {}
    for key, value in sorted(images_people_perc_mscoco.items(), key=lambda item: item[0]):
        images_people_perc_mscoco_sorted[key] = value
    images_people_perc_emodb_sorted = {}
    for key, value in sorted(images_people_perc_emodb.items(), key=lambda item: item[0]):
        images_people_perc_emodb_sorted[key] = value
    
    # print(images_people_ratios_total_sorted)
    # plt.figure('Ratio people/image')
    # fig, (ax0, ax1) = plt.subplots(2,1, figsize=(13, 10))
    # plot_total, = ax0.plot(images_people_perc_total_sorted.keys(), images_people_perc_total_sorted.values(), label='mscoco + emodb', color='green')
    # ax0.fill_between(images_people_perc_total_sorted.keys(), images_people_perc_total_sorted.values(), color='green')    
    # plot_mscoco, = ax1.plot(images_people_perc_mscoco_sorted.keys(), images_people_perc_mscoco_sorted.values(), label='mscoco', color='yellow')
    # ax1.fill_between(images_people_perc_mscoco_sorted.keys(), images_people_perc_mscoco_sorted.values(), color='yellow')
    # plot_emodb, = ax1.plot(images_people_perc_emodb_sorted.keys(), images_people_perc_emodb_sorted.values(),label='emodb', color='blue')
    # ax1.fill_between(images_people_perc_emodb_sorted.keys(), images_people_perc_emodb_sorted.values(), color='blue')

    # ax0.legend(loc='upper right')
    # ax1.legend(loc='upper right')
    # ax0.set_xlabel('percentage [%]')
    # ax0.set_ylabel('# of images')
    # ax1.set_xlabel('percentage [%]')
    # ax1.set_ylabel('# of images')
    # ax0.set_title('Area occupied by people in the image')
    # plt.savefig("./nonselfie_perc_people_area_image_area.svg")
    # plt.close()

    plt.figure('Ratio people/image bar chart')
    fig, (ax0, ax1) = plt.subplots(2,1, figsize=(13, 10))

    width0 = 8.0
    rect = ax0.bar(images_people_perc_total_sorted.keys(), 
                    images_people_perc_total_sorted.values(), width=width0, label='mscoco + emodb', color='green')
    ax0.bar_label(rect, padding=3)
    ax0.set_xticks(list(images_people_perc_total_sorted.keys()))
    if len(list(images_people_perc_total_sorted.keys())) > 11:
        ax0.set_xticklabels(['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '>100'] )
    else:
        ax0.set_xticklabels(list(images_people_perc_total_sorted.keys()))
    ax0.set_title('Area occupied by people in images')
    ax0.legend(loc='upper left')
    ax0.set_xlabel('percentage of area occupied by people [%]')
    ax0.set_ylabel('# of images')

    width1 = 0.4
    x = np.arange(len(list(images_people_perc_mscoco_sorted.keys())))
    rect_mscoco = ax1.bar(x - width1/2, images_people_perc_mscoco_sorted.values(), width1, label='mscoco', color='yellow')
    ax1.bar_label(rect_mscoco, padding=3)
    x = np.arange(len(list(images_people_perc_emodb_sorted.keys())))
    rect_emodb = ax1.bar(x + width1/2, images_people_perc_emodb_sorted.values(), width1, label='emodb', color='blue')
    ax1.bar_label(rect_emodb, padding=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(images_people_perc_total_sorted.keys()))
    ax1.set_title('Area occupied by people in images')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('percentage of area occupied by people [%]')
    ax1.set_ylabel('# of images')
    
    bottom, top = ax0.get_ylim()
    ax0.set_ylim([bottom, top + top*0.2])
    bottom, top = ax1.get_ylim()
    ax1.set_ylim([bottom, top + top*0.2])
    fig.subplots_adjust(hspace=0.3)
    plt.savefig("{}/bar_chart_nonselfie_perc_people_area{}{}.svg".format(save_folder, "_with_yolo" if use_yolo else "", "_adjusted_g_truth" if adjust_ground_truth else ""))
    plt.close()

    if use_yolo == False and adjust_ground_truth == False:
        # with original g truth only extract first analysis
        exit()

    plt.figure('Ratio people/image for wrong predictions bar chart')
    fig, (ax0, ax1, ax2) = plt.subplots(3,1, figsize=(13, 15))

    width0 = 8.0
    rect = ax0.bar(wrong_images_people_perc_total.keys(), 
                    wrong_images_people_perc_total.values(), width=width0, label='selfie + scenes', color='orange')
    ax0.bar_label(rect, padding=3)
    ax0.set_xticks(list(wrong_images_people_perc_total.keys()))
    ax0.set_xticklabels(list(wrong_images_people_perc_total.keys()))
    ax0.set_title('Area occupied by people in wrong prediction images')
    ax0.legend(loc='upper left')
    ax0.set_xlabel('percentage of area occupied by people [%]')
    ax0.set_ylabel('# of images')

    width1 = 0.4
    x = np.arange(len(list(wrong_images_people_perc_scenes.keys())))
    rect_mscoco = ax1.bar(x - width1/2, wrong_images_people_perc_scenes.values(), width1, label='scenes', color='yellow')
    ax1.bar_label(rect_mscoco, padding=3)
    x = np.arange(len(list(wrong_images_people_perc_selfie.keys())))
    rect_emodb = ax1.bar(x + width1/2, wrong_images_people_perc_selfie.values(), width1, label='selfie', color='red')
    ax1.bar_label(rect_emodb, padding=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(wrong_images_people_perc_selfie.keys()))
    ax1.set_title('Area occupied by people in wrong prediction images')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('percentage of area occupied by people [%]')
    ax1.set_ylabel('# of images')

    width2 = width1/2
    x = np.arange(len(list(wrong_images_people_perc_mscoco_scenes.keys())))
    rect_mscoco = ax2.bar(x - width1 * 3/4 , wrong_images_people_perc_mscoco_scenes.values(), width2, label='mscoco', color='green')
    ax2.bar_label(rect_mscoco, padding=3)
    x = np.arange(len(list(wrong_images_people_perc_emodb_scenes.keys())))
    rect_mscoco = ax2.bar(x - width1/4, wrong_images_people_perc_emodb_scenes.values(), width2, label='emodb', color='blue')
    ax2.bar_label(rect_mscoco, padding=3)
    

    x = np.arange(len(list(wrong_images_people_perc_mscoco_selfies.keys())))
    rect_mscoco = ax2.bar(x + width1/4, wrong_images_people_perc_mscoco_selfies.values(), width2, color='green')
    ax2.bar_label(rect_mscoco, padding=3)
    x = np.arange(len(list(wrong_images_people_perc_emodb_selfies.keys())))
    rect_mscoco = ax2.bar(x + width1* 3/4, wrong_images_people_perc_emodb_selfies.values(), width2, color='blue')
    ax2.bar_label(rect_mscoco, padding=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(wrong_images_people_perc_selfie.keys()))
    ax2.set_title('Area occupied by people in wrong prediction images')
    ax2.legend(loc='upper left')
    ax2.set_xlabel('percentage of area occupied by people [%]')
    ax2.set_ylabel('# of images')
    
    
    bottom, top = ax0.get_ylim()
    ax0.set_ylim([bottom, top + top*0.2])
    bottom, top = ax1.get_ylim()
    ax1.set_ylim([bottom, top + top*0.2])
    bottom, top = ax2.get_ylim()
    ax2.set_ylim([bottom, top + top*0.2])
    fig.subplots_adjust(hspace=0.3)
    plt.savefig("{}/bar_chart_nonselfie_wrong_predictions_perc_people_area{}{}.svg".format(save_folder, "_with_yolo" if use_yolo else "", "_adjusted_g_truth" if adjust_ground_truth else ""))
    plt.close()

    
    plt.figure('Ratio people/image for wrong predictions weighted on total bar chart')
    fig, (ax0, ax1, ax2) = plt.subplots(3,1, figsize=(13, 15))

    wrong_images_people_perc_total_weighted = {}
    for key in wrong_images_people_perc_total.keys():
        wrong_images_people_perc_total_weighted[key] = round(wrong_images_people_perc_total[key] / len(dataset) * 100, 2)
    width0 = 8.0
    rect = ax0.bar(wrong_images_people_perc_total_weighted.keys(), 
                    wrong_images_people_perc_total_weighted.values(), width=width0, label='selfie + scenes', color='orange')
    ax0.bar_label(rect, padding=3)
    ax0.set_xticks(list(wrong_images_people_perc_total_weighted.keys()))
    ax0.set_xticklabels(list(wrong_images_people_perc_total_weighted.keys()))
    ax0.set_title('Area occupied by people in wrong predictions images')
    ax0.legend(loc='upper left')
    ax0.set_xlabel('percentage of area occupied by people [%]')
    ax0.set_ylabel('percentage of images [%]')

    wrong_images_people_perc_scenes_weighted = {}
    for key in wrong_images_people_perc_scenes.keys():
        wrong_images_people_perc_scenes_weighted[key] = round(wrong_images_people_perc_scenes[key] / len(dataset) * 100, 2)
    wrong_images_people_perc_selfie_weighted = {}
    for key in wrong_images_people_perc_selfie.keys():
        wrong_images_people_perc_selfie_weighted[key] = round(wrong_images_people_perc_selfie[key] / len(dataset) * 100, 2)

    width1 = 0.4
    x = np.arange(len(list(wrong_images_people_perc_scenes_weighted.keys())))
    rect_mscoco = ax1.bar(x - width1/2, wrong_images_people_perc_scenes_weighted.values(), width1, label='scenes', color='yellow')
    ax1.bar_label(rect_mscoco, padding=3)
    x = np.arange(len(list(wrong_images_people_perc_selfie_weighted.keys())))
    rect_emodb = ax1.bar(x + width1/2, wrong_images_people_perc_selfie_weighted.values(), width1, label='selfie', color='red')
    ax1.bar_label(rect_emodb, padding=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(wrong_images_people_perc_selfie_weighted.keys()))
    ax1.set_title('Area occupied by people in wrong predictions images')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('percentage of area occupied by people [%]')
    ax1.set_ylabel('percentage of images [%]')


    wrong_images_people_perc_mscoco_scenes_weighted = {}
    for key in wrong_images_people_perc_mscoco_scenes.keys():
        wrong_images_people_perc_mscoco_scenes_weighted[key] = round(wrong_images_people_perc_mscoco_scenes[key] / len(dataset) * 100, 2)
    wrong_images_people_perc_mscoco_selfies_weighted = {}
    for key in wrong_images_people_perc_mscoco_selfies.keys():
        wrong_images_people_perc_mscoco_selfies_weighted[key] = round(wrong_images_people_perc_mscoco_selfies[key] / len(dataset) * 100, 2)

    wrong_images_people_perc_emodb_scenes_weighted = {}
    for key in wrong_images_people_perc_emodb_scenes.keys():
        wrong_images_people_perc_emodb_scenes_weighted[key] = round(wrong_images_people_perc_emodb_scenes[key] / len(dataset) * 100, 2)
    wrong_images_people_perc_emodb_selfies_weighted = {}
    for key in wrong_images_people_perc_emodb_selfies.keys():
        wrong_images_people_perc_emodb_selfies_weighted[key] = round(wrong_images_people_perc_emodb_selfies[key] / len(dataset) * 100, 2)

    width2 = width1/2
    x = np.arange(len(list(wrong_images_people_perc_mscoco_scenes_weighted.keys())))
    rect_mscoco = ax2.bar(x - width1 * 3/4 , wrong_images_people_perc_mscoco_scenes_weighted.values(), width2, label='mscoco', color='green')
    ax2.bar_label(rect_mscoco, padding=3)
    x = np.arange(len(list(wrong_images_people_perc_emodb_scenes_weighted.keys())))
    rect_mscoco = ax2.bar(x - width1/4, wrong_images_people_perc_emodb_scenes_weighted.values(), width2, label='emodb', color='blue')
    ax2.bar_label(rect_mscoco, padding=3)

    x = np.arange(len(list(wrong_images_people_perc_mscoco_selfies_weighted.keys())))
    rect_mscoco = ax2.bar(x + width1/4, wrong_images_people_perc_mscoco_selfies_weighted.values(), width2, color='green')
    ax2.bar_label(rect_mscoco, padding=3)
    x = np.arange(len(list(wrong_images_people_perc_emodb_selfies_weighted.keys())))
    rect_mscoco = ax2.bar(x + width1* 3/4, wrong_images_people_perc_emodb_selfies_weighted.values(), width2, color='blue')
    ax2.bar_label(rect_mscoco, padding=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(wrong_images_people_perc_selfie_weighted.keys()))
    ax2.set_title('Area occupied by people in wrong prediction images')
    ax2.legend(loc='upper left')
    ax2.set_xlabel('percentage of area occupied by people [%]')
    ax2.set_ylabel('percentage of images [%]')
    
    
    bottom, top = ax0.get_ylim()
    ax0.set_ylim([bottom, top + top*0.2])
    bottom, top = ax1.get_ylim()
    ax1.set_ylim([bottom, top + top*0.2])
    bottom, top = ax2.get_ylim()
    ax2.set_ylim([bottom, top + top*0.2])
    fig.subplots_adjust(hspace=0.3)
    plt.savefig("{}/bar_chart_nonselfie_wrong_predictions_perc_people_area_weighted{}{}.svg".format(save_folder, "_with_yolo" if use_yolo else "", "_adjusted_g_truth" if adjust_ground_truth else ""))
    plt.close()

    plt.figure('Ratio people/image for wrong predictions weighted on every percentage bar chart')
    fig, (ax0, ax1, ax2) = plt.subplots(3,1, figsize=(13, 15))

    wrong_images_people_perc_total_weighted = {}
    for key in wrong_images_people_perc_total.keys():
        wrong_images_people_perc_total_weighted[key] = round(wrong_images_people_perc_total[key] / max(images_people_perc_total[key], 1) * 100, 2)
    width0 = 8.0
    rect = ax0.bar(wrong_images_people_perc_total_weighted.keys(), 
                    wrong_images_people_perc_total_weighted.values(), width=width0, label='selfie + scenes', color='orange')
    ax0.bar_label(rect, padding=3)
    ax0.set_xticks(list(wrong_images_people_perc_total_weighted.keys()))
    ax0.set_xticklabels(list(wrong_images_people_perc_total_weighted.keys()))
    ax0.set_title('Area occupied by people in wrong predictions images')
    ax0.legend(loc='upper left')
    ax0.set_xlabel('percentage of area occupied by people [%]')
    ax0.set_ylabel('percentage of images [%]')

    wrong_images_people_perc_scenes_weighted = {}
    for key in wrong_images_people_perc_scenes.keys():
        wrong_images_people_perc_scenes_weighted[key] = round(wrong_images_people_perc_scenes[key] / max(images_people_perc_total[key], 1) * 100, 2)
    wrong_images_people_perc_selfie_weighted = {}
    for key in wrong_images_people_perc_selfie.keys():
        wrong_images_people_perc_selfie_weighted[key] = round(wrong_images_people_perc_selfie[key] / max(images_people_perc_total[key], 1) * 100, 2)

    width1 = 0.4
    x = np.arange(len(list(wrong_images_people_perc_scenes_weighted.keys())))
    rect_mscoco = ax1.bar(x - width1/2, wrong_images_people_perc_scenes_weighted.values(), width1, label='scenes', color='yellow')
    ax1.bar_label(rect_mscoco, padding=3)
    x = np.arange(len(list(wrong_images_people_perc_selfie_weighted.keys())))
    rect_emodb = ax1.bar(x + width1/2, wrong_images_people_perc_selfie_weighted.values(), width1, label='selfie', color='red')
    ax1.bar_label(rect_emodb, padding=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(wrong_images_people_perc_selfie_weighted.keys()))
    ax1.set_title('Area occupied by people in wrong predictions images')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('percentage of area occupied by people [%]')
    ax1.set_ylabel('percentage of images [%]')


    wrong_images_people_perc_mscoco_scenes_weighted = {}
    for key in wrong_images_people_perc_mscoco_scenes.keys():
        wrong_images_people_perc_mscoco_scenes_weighted[key] = round(wrong_images_people_perc_mscoco_scenes[key] / max(images_people_perc_total[key], 1) * 100, 2)
    wrong_images_people_perc_mscoco_selfies_weighted = {}
    for key in wrong_images_people_perc_mscoco_selfies.keys():
        wrong_images_people_perc_mscoco_selfies_weighted[key] = round(wrong_images_people_perc_mscoco_selfies[key] / max(images_people_perc_total[key], 1) * 100, 2)

    wrong_images_people_perc_emodb_scenes_weighted = {}
    for key in wrong_images_people_perc_emodb_scenes.keys():
        wrong_images_people_perc_emodb_scenes_weighted[key] = round(wrong_images_people_perc_emodb_scenes[key] / max(images_people_perc_total[key], 1) * 100, 2)
    wrong_images_people_perc_emodb_selfies_weighted = {}
    for key in wrong_images_people_perc_emodb_selfies.keys():
        wrong_images_people_perc_emodb_selfies_weighted[key] = round(wrong_images_people_perc_emodb_selfies[key] / max(images_people_perc_total[key], 1) * 100, 2)

    width2 = width1/2
    x = np.arange(len(list(wrong_images_people_perc_mscoco_scenes_weighted.keys())))
    rect_mscoco = ax2.bar(x - width1 * 3/4 , wrong_images_people_perc_mscoco_scenes_weighted.values(), width2, label='mscoco', color='green')
    ax2.bar_label(rect_mscoco, padding=3)
    x = np.arange(len(list(wrong_images_people_perc_emodb_scenes_weighted.keys())))
    rect_mscoco = ax2.bar(x - width1/4, wrong_images_people_perc_emodb_scenes_weighted.values(), width2, label='emodb', color='blue')
    ax2.bar_label(rect_mscoco, padding=3)

    x = np.arange(len(list(wrong_images_people_perc_mscoco_selfies_weighted.keys())))
    rect_mscoco = ax2.bar(x + width1/4, wrong_images_people_perc_mscoco_selfies_weighted.values(), width2, color='green')
    ax2.bar_label(rect_mscoco, padding=3)
    x = np.arange(len(list(wrong_images_people_perc_emodb_selfies_weighted.keys())))
    rect_mscoco = ax2.bar(x + width1* 3/4, wrong_images_people_perc_emodb_selfies_weighted.values(), width2, color='blue')
    ax2.bar_label(rect_mscoco, padding=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(wrong_images_people_perc_selfie_weighted.keys()))
    ax2.set_title('Area occupied by people in wrong prediction images')
    ax2.legend(loc='upper left')
    ax2.set_xlabel('percentage of area occupied by people [%]')
    ax2.set_ylabel('percentage of images [%]')
    
    
    bottom, top = ax0.get_ylim()
    ax0.set_ylim([bottom, top + top*0.2])
    bottom, top = ax1.get_ylim()
    ax1.set_ylim([bottom, top + top*0.2])
    bottom, top = ax2.get_ylim()
    ax2.set_ylim([bottom, top + top*0.2])
    fig.subplots_adjust(hspace=0.3)
    plt.savefig("{}/bar_chart_nonselfie_wrong_predictions_perc_people_area_weighted_on_single_perc{}{}.svg".format(save_folder, "_with_yolo" if use_yolo else "", "_adjusted_g_truth" if adjust_ground_truth else ""))
    plt.close()


    with open("{}/statistics{}{}.txt".format(save_folder, "_with_yolo" if use_yolo else "", "_adjusted_g_truth" if adjust_ground_truth else ""), "w") as stats:
        stats.write("There are {} images where bboxes are bigger than the image itself\n".format(overflow_bboxes))
        for img in overflow_bboxes_images:
            stats.write("\t{}\n".format(img))


def calculate_SUN397_people_perc_from_SPP(dataset_folder, save_folder):
    
    dataset = SUN397(root=dataset_folder, split=['train', 'test'], aspect_ratio_threshold=2.33, dim_threshold=225)

    print("Dataset has {} images".format(len(dataset)))

    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

    overflow_bboxes = 0
    images_people_perc_total = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}
    images_with_people_perc = {}

    for i, (img, target) in enumerate(dataset):
        W, H = img.size
        # print(W)
        # print(H)
        img_area = W*H
        img_path = dataset.metadata[i]['img_folder'] + "/" + dataset.metadata[i]['img_name']

        results = yolo(img)

        # Results
        # results.print()
        # results.save()  # or .show()

        results.xyxy[0]  # img1 predictions (tensor)
        # print(dataset.metadata[i])
        # print(results.pandas().xyxy[0])
        predicted_bboxes = results.pandas().xyxy[0]
        bboxes = []
        for j, r in enumerate(predicted_bboxes['name']):
            if r == 'person':
                
                x0 = round(float(predicted_bboxes['xmin'][j]), 2)
                y0 = round(float(predicted_bboxes['ymin'][j]), 2)
                x1 = round(float(predicted_bboxes['xmax'][j]), 2)
                y1 = round(float(predicted_bboxes['ymax'][j]), 2)
                # print(x0)
                # print(y0)
                # print(x1)
                # print(y1)
                bbox = Polygon([(x0,y0), (x1, y0), (x1, y1), (x0, y1)])
                bboxes.append(bbox)
        # if results.pandas().xyxy[0]['name'] == 'person':
        #     print("yaoo")
        people_area = Polygon([(0,0), (0,0), (0,0), (0,0),])
    
        for b in bboxes:
            people_area = people_area.union(b)
            # print(people_area.area)

        people_area = people_area.area
        if people_area > img_area:
            overflow_bboxes += 1
            people_area = img_area
        
        perc = math.ceil(math.ceil(people_area/img_area*100) / 10) * 10

        images_people_perc_total[perc] += 1
        images_with_people_perc[img_path] = {"img_area": W*H, "people_area": people_area, "people_percentage": perc}


    with open('{}/SUN397_images_with_people_perc.json'.format(save_folder), 'w') as outfile:
        json.dump(images_with_people_perc, outfile)

    plt.figure('Ratio people/image bar chart')
    fig, ax0 = plt.subplots(figsize=(13, 5))

    width0 = 8.0
    print(overflow_bboxes)
    rect = ax0.bar(images_people_perc_total.keys(), 
                    images_people_perc_total.values(), width=width0, color='green')
    ax0.bar_label(rect, padding=3)
    ax0.set_xticks(list(images_people_perc_total.keys()))
    ax0.set_xticklabels(list(images_people_perc_total.keys()))
    ax0.set_title('Area occupied by people in the image')
    ax0.set_xlabel('percentage of area occupied by people [%]')
    ax0.set_ylabel('# of images')

    bottom, top = ax0.get_ylim()
    ax0.set_ylim([bottom, top + top*0.2])
    plt.savefig("{}/bar_chart_SUN397_perc_people_area.svg".format(save_folder))
    plt.close()


def calculate_scenes_people_perc_from_SPP(spp_folder, sun_folder, save_folder, wrong_predictions_folder):

    wrong_images_paths = {}
    wrong_images_people_perc_total = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}
    wrong_images_people_perc_selfie = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}
    wrong_images_people_perc_nonselfie = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}

    images_with_people_perc = {}
    wrong_predictions_with_people_perc = {}

    scenes = ["shopping_and_dining", "workplace", "home_or_hotel",
            "transportation", "sports_and_leisure", "cultural",
            "water_ice_snow", "mountains_hills_desert_sky",
            "forest_field_jungle", "man-made_elements",
            "transportation", "cultural_or_historical_building_place",
            "sportsfields_parks_leisure_spaces", "industrial_and_construction",
            "houses_cabins_gardens_and_farms", "commercial_buildings"]

    for scene in scenes:
        with open("{}/{}_false_positives.txt".format(wrong_predictions_folder, scene), "r") as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                if i > 0:
                    img_path = line.split(":")[1].split(",")[0].replace("'","").strip()
                    if "nonselfie" in line:
                        wrong_images_paths[img_path] = {"wrong_prediction": "nonselfie"}
                    elif "selfie" in line:
                        wrong_images_paths[img_path] = {"wrong_prediction": "selfie"}
    
    dataset = SocialProfilePictures(root=spp_folder, split=['train', 'test', 'val'], aspect_ratio_threshold=2.33, dim_threshold=225)

    print("Dataset has {} images".format(len(dataset)))

    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

    overflow_bboxes = 0
    images_people_perc_total = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}

    for i, (img, target) in enumerate(dataset):

        if dataset.metadata[i]['target']['level0'] == 'scenes':
            # print('yaoo')
            W, H = img.size
            # print(W)
            # print(H)
            img_area = W*H
            img_path = dataset.metadata[i]['img_folder'] + "/" + dataset.metadata[i]['img_name']
            

            results = yolo(img)

            # Results
            # results.print()
            # results.save()  # or .show()

            results.xyxy[0]  # img1 predictions (tensor)
            # print(dataset.metadata[i])
            # print(results.pandas().xyxy[0])
            predicted_bboxes = results.pandas().xyxy[0]
            bboxes = []
            for j, r in enumerate(predicted_bboxes['name']):
                if r == 'person':
                    
                    x0 = round(float(predicted_bboxes['xmin'][j]), 2)
                    y0 = round(float(predicted_bboxes['ymin'][j]), 2)
                    x1 = round(float(predicted_bboxes['xmax'][j]), 2)
                    y1 = round(float(predicted_bboxes['ymax'][j]), 2)
                    # print(x0)
                    # print(y0)
                    # print(x1)
                    # print(y1)
                    bbox = Polygon([(x0,y0), (x1, y0), (x1, y1), (x0, y1)])
                    bboxes.append(bbox)
            # if results.pandas().xyxy[0]['name'] == 'person':
            #     print("yaoo")
            people_area = Polygon([(0,0), (0,0), (0,0), (0,0),])
        
            for b in bboxes:
                people_area = people_area.union(b)
                # print(people_area.area)

            people_area = people_area.area
            if people_area > img_area:
                overflow_bboxes += 1
                people_area = img_area
            
            perc = math.ceil(math.ceil(people_area/img_area*100) / 10) * 10

            images_people_perc_total[perc] += 1
            images_with_people_perc[img_path] = {"img_area": W*H, "people_area": people_area, "people_percentage": perc}
            
            if img_path in wrong_images_paths.keys():
                wrong_predictions_with_people_perc[img_path] = {"people_percentage": perc, "wrong_prediction": wrong_images_paths[img_path]["wrong_prediction"]}
                wrong_images_people_perc_total[perc] += 1
                if wrong_images_paths[img_path]["wrong_prediction"] == "selfie":
                    wrong_images_people_perc_selfie[perc] += 1
                elif wrong_images_paths[img_path]["wrong_prediction"] == "nonselfie":
                    wrong_images_people_perc_nonselfie[perc] += 1


    with open('{}/images_with_people_perc.json'.format(save_folder), 'w') as outfile:
        json.dump(images_with_people_perc, outfile)

    with open('{}/wrong_predictions_with_people_perc.json'.format(save_folder), 'w') as outfile:
        json.dump(wrong_predictions_with_people_perc, outfile) 

    plt.figure('Ratio people/image bar chart')
    fig, ax0 = plt.subplots(figsize=(13, 5))

    width0 = 8.0
    print(overflow_bboxes)
    rect = ax0.bar(images_people_perc_total.keys(), 
                    images_people_perc_total.values(), width=width0, color='green')
    ax0.bar_label(rect, padding=3)
    ax0.set_xticks(list(images_people_perc_total.keys()))
    ax0.set_xticklabels(list(images_people_perc_total.keys()))
    ax0.set_title('Area occupied by people in the image')
    ax0.set_xlabel('percentage of area occupied by people [%]')
    ax0.set_ylabel('# of images')

    bottom, top = ax0.get_ylim()
    ax0.set_ylim([bottom, top + top*0.2])
    plt.savefig("{}/bar_chart_scenes_perc_people_area.svg".format(save_folder))
    plt.close()

    plt.figure('Ratio people/image bar chart for wrong predictions')
    fig, (ax0, ax1) = plt.subplots(2,1, figsize=(13, 10))

    width0 = 8.0
    rect = ax0.bar(wrong_images_people_perc_total.keys(), 
                    wrong_images_people_perc_total.values(), width=width0, label='selfie + nonselfie', color='green')
    ax0.bar_label(rect, padding=3)
    ax0.set_xticks(list(wrong_images_people_perc_total.keys()))
    ax0.set_xticklabels(list(wrong_images_people_perc_total.keys()))
    ax0.set_title('Area occupied by people in images')
    ax0.legend(loc='upper left')
    ax0.set_xlabel('percentage of area occupied by people [%]')
    ax0.set_ylabel('# of images')

    width1 = 0.4
    x = np.arange(len(list(wrong_images_people_perc_selfie.keys())))
    rect_mscoco = ax1.bar(x - width1/2, wrong_images_people_perc_selfie.values(), width1, label='selfie', color='yellow')
    ax1.bar_label(rect_mscoco, padding=3)
    x = np.arange(len(list(wrong_images_people_perc_nonselfie.keys())))
    rect_emodb = ax1.bar(x + width1/2, wrong_images_people_perc_nonselfie.values(), width1, label='nonselfie', color='blue')
    ax1.bar_label(rect_emodb, padding=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(wrong_images_people_perc_total.keys()))
    ax1.set_title('Area occupied by people in images')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('percentage of area occupied by people [%]')
    ax1.set_ylabel('# of images')
    
    bottom, top = ax0.get_ylim()
    ax0.set_ylim([bottom, top + top*0.2])
    bottom, top = ax1.get_ylim()
    ax1.set_ylim([bottom, top + top*0.2])
    fig.subplots_adjust(hspace=0.3)
    plt.savefig("{}/bar_chart_scenes_wrong_predictions_perc_people_area.svg".format(save_folder))
    plt.close()


    plt.figure('Ratio people/image bar chart for wrong predictions weighted on total')
    fig, (ax0, ax1) = plt.subplots(2,1, figsize=(13, 10))

    wrong_images_people_perc_total_weighted = {}
    for key in wrong_images_people_perc_total.keys():
        wrong_images_people_perc_total_weighted[key] = round(wrong_images_people_perc_total[key] / len(list(images_with_people_perc.keys())) * 100, 2)
    width0 = 8.0
    rect = ax0.bar(wrong_images_people_perc_total_weighted.keys(), 
                    wrong_images_people_perc_total_weighted.values(), width=width0, label='selfie + nonselfie', color='green')
    ax0.bar_label(rect, padding=3)
    ax0.set_xticks(list(wrong_images_people_perc_total_weighted.keys()))
    ax0.set_xticklabels(list(wrong_images_people_perc_total_weighted.keys()))
    ax0.set_title('Area occupied by people in images')
    ax0.legend(loc='upper left')
    ax0.set_xlabel('percentage of area occupied by people [%]')
    ax0.set_ylabel('percentage of images [%]')

    wrong_images_people_perc_selfie_weighted = {}
    for key in wrong_images_people_perc_selfie.keys():
        wrong_images_people_perc_selfie_weighted[key] = round(wrong_images_people_perc_selfie[key] / len(list(images_with_people_perc.keys())) * 100, 2)
    wrong_images_people_perc_nonselfie_weighted = {}
    for key in wrong_images_people_perc_nonselfie.keys():
        wrong_images_people_perc_nonselfie_weighted[key] = round(wrong_images_people_perc_nonselfie[key] / len(list(images_with_people_perc.keys())) * 100, 2)
    width1 = 0.4
    x = np.arange(len(list(wrong_images_people_perc_selfie_weighted.keys())))
    rect_mscoco = ax1.bar(x - width1/2, wrong_images_people_perc_selfie_weighted.values(), width1, label='selfie', color='yellow')
    ax1.bar_label(rect_mscoco, padding=3)
    x = np.arange(len(list(wrong_images_people_perc_nonselfie_weighted.keys())))
    rect_emodb = ax1.bar(x + width1/2, wrong_images_people_perc_nonselfie_weighted.values(), width1, label='nonselfie', color='blue')
    ax1.bar_label(rect_emodb, padding=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(wrong_images_people_perc_total_weighted.keys()))
    ax1.set_title('Area occupied by people in images')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('percentage of area occupied by people [%]')
    ax1.set_ylabel('percentage of images [%]')
    
    bottom, top = ax0.get_ylim()
    ax0.set_ylim([bottom, top + top*0.2])
    bottom, top = ax1.get_ylim()
    ax1.set_ylim([bottom, top + top*0.2])
    fig.subplots_adjust(hspace=0.3)
    plt.savefig("{}/bar_chart_scenes_wrong_predictions_perc_people_area_weighted.svg".format(save_folder))
    plt.close()


    plt.figure('Ratio people/image bar chart for wrong predictions weighted on single perc')
    fig, (ax0, ax1) = plt.subplots(2,1, figsize=(13, 10))

    wrong_images_people_perc_total_weighted = {}
    for key in wrong_images_people_perc_total.keys():
        wrong_images_people_perc_total_weighted[key] = round(wrong_images_people_perc_total[key] / max(images_people_perc_total[key], 1) * 100, 2)
    width0 = 8.0
    rect = ax0.bar(wrong_images_people_perc_total_weighted.keys(), 
                    wrong_images_people_perc_total_weighted.values(), width=width0, label='selfie + nonselfie', color='green')
    ax0.bar_label(rect, padding=3)
    ax0.set_xticks(list(wrong_images_people_perc_total_weighted.keys()))
    ax0.set_xticklabels(list(wrong_images_people_perc_total_weighted.keys()))
    ax0.set_title('Area occupied by people in images')
    ax0.legend(loc='upper left')
    ax0.set_xlabel('percentage of area occupied by people [%]')
    ax0.set_ylabel('percentage of images [%]')

    wrong_images_people_perc_selfie_weighted = {}
    for key in wrong_images_people_perc_selfie.keys():
        wrong_images_people_perc_selfie_weighted[key] = round(wrong_images_people_perc_selfie[key] / max(images_people_perc_total[key], 1) * 100, 2)
    wrong_images_people_perc_nonselfie_weighted = {}
    for key in wrong_images_people_perc_nonselfie.keys():
        wrong_images_people_perc_nonselfie_weighted[key] = round(wrong_images_people_perc_nonselfie[key] / max(images_people_perc_total[key], 1) * 100, 2)
    width1 = 0.4
    x = np.arange(len(list(wrong_images_people_perc_selfie_weighted.keys())))
    rect_mscoco = ax1.bar(x - width1/2, wrong_images_people_perc_selfie_weighted.values(), width1, label='selfie', color='yellow')
    ax1.bar_label(rect_mscoco, padding=3)
    x = np.arange(len(list(wrong_images_people_perc_nonselfie_weighted.keys())))
    rect_emodb = ax1.bar(x + width1/2, wrong_images_people_perc_nonselfie_weighted.values(), width1, label='nonselfie', color='blue')
    ax1.bar_label(rect_emodb, padding=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(wrong_images_people_perc_total_weighted.keys()))
    ax1.set_title('Area occupied by people in images')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('percentage of area occupied by people [%]')
    ax1.set_ylabel('percentage of images [%]')
    
    bottom, top = ax0.get_ylim()
    ax0.set_ylim([bottom, top + top*0.2])
    bottom, top = ax1.get_ylim()
    ax1.set_ylim([bottom, top + top*0.2])
    fig.subplots_adjust(hspace=0.3)
    plt.savefig("{}/bar_chart_scenes_wrong_predictions_perc_people_area_weighted_on_single_perc.svg".format(save_folder))
    plt.close()


def calculate_scenes_false_positives_for_hierarchy_classes(dataset_folder, save_folder, wrong_predictions_folder):
    scenes = ["shopping_and_dining", "workplace", "home_or_hotel",
                    "transportation", "sports_and_leisure", "cultural",
                    "water_ice_snow", "mountains_hills_desert_sky",
                    "forest_field_jungle", "man-made_elements",
                    "transportation", "cultural_or_historical_building_place",
                    "sportsfields_parks_leisure_spaces", "industrial_and_construction",
                    "houses_cabins_gardens_and_farms", "commercial_buildings"]

    wrong_predictions_level_0 = { 'people': 0, 'scenes': 0, 'other': 0}
    wrong_predictions_level_1 = {'selfie': 0, 'nonselfie': 0, 'scenes': 0, 'pet': 0, 'cartoon': 0, 'art': 0}
    for scene in scenes:
        with open("{}/{}_false_positives.txt".format(wrong_predictions_folder, scene), "r") as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                if i > 0:
                    img_path = line.split(":")[1].split(",")[0].replace("'","").strip()
                    wrong_prediction = line.split("wrong_prediction")[1]
                    if "nonselfie" in wrong_prediction:
                        wrong_predictions_level_0['people'] += 1
                        wrong_predictions_level_1['nonselfie'] +=1
                    elif "selfie" in wrong_prediction:
                        wrong_predictions_level_0['people'] += 1
                        wrong_predictions_level_1['selfie'] += 1            
                    elif ('cat' in wrong_prediction or 'dog' in wrong_prediction or 'cartoon' in wrong_prediction 
                        or 'drawings' in wrong_prediction or 'engraving' in wrong_prediction or 
                        'iconography' in wrong_prediction or 'painting' in wrong_prediction or 'sculpture' in wrong_prediction):
                        wrong_predictions_level_0['other'] += 1
                        if 'cat' in wrong_prediction or 'dog' in wrong_prediction:
                            wrong_predictions_level_1['pet'] += 1
                        elif 'cartoon' in wrong_prediction:
                            wrong_predictions_level_1['cartoon'] += 1
                        else:
                            wrong_predictions_level_1['art'] += 1
                    else:
                        wrong_predictions_level_0['scenes'] += 1
                        wrong_predictions_level_1['scenes'] += 1
    print(wrong_predictions_level_0)
    print(wrong_predictions_level_1)

def calculate_people_false_positives_for_hierarchy_classes(dataset_folder, save_folder, wrong_predictions_folder):

    scenes = ["shopping_and_dining", "workplace", "home_or_hotel",
                    "transportation", "sports_and_leisure", "cultural",
                    "water_ice_snow", "mountains_hills_desert_sky",
                    "forest_field_jungle", "man-made_elements",
                    "transportation", "cultural_or_historical_building_place",
                    "sportsfields_parks_leisure_spaces", "industrial_and_construction",
                    "houses_cabins_gardens_and_farms", "commercial_buildings"]
    
    wrong_predictions_level_0 = { 'people': 0, 'scenes': 0, 'other': 0}
    wrong_predictions_selfie_level_1 = {'selfie': 0, 'nonselfie': 0, 'scenes': 0, 'pet': 0, 'cartoon': 0, 'art': 0}
    wrong_predictions_nonselfie_level_1 = {'selfie': 0, 'nonselfie': 0, 'scenes': 0, 'pet': 0, 'cartoon': 0, 'art': 0}
    with open("{}/selfie_false_positives.txt".format(wrong_predictions_folder), "r") as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if i > 0:
                wrong_prediction = line.split("wrong_prediction")[1]
                if "nonselfie" in wrong_prediction:
                    wrong_predictions_level_0['people'] += 1
                    wrong_predictions_selfie_level_1['nonselfie'] +=1       
                elif ('cat' in wrong_prediction or 'dog' in wrong_prediction or 'cartoon' in wrong_prediction 
                    or 'drawings' in wrong_prediction or 'engraving' in wrong_prediction or 
                    'iconography' in wrong_prediction or 'painting' in wrong_prediction or 'sculpture' in wrong_prediction):
                    wrong_predictions_level_0['other'] += 1
                    if 'cat' in wrong_prediction or 'dog' in wrong_prediction:
                        wrong_predictions_selfie_level_1['pet'] += 1
                    elif 'cartoon' in wrong_prediction:
                        wrong_predictions_selfie_level_1['cartoon'] += 1
                    else:
                        wrong_predictions_selfie_level_1['art'] += 1
                else:
                    wrong_predictions_level_0['scenes'] += 1
                    wrong_predictions_selfie_level_1['scenes'] += 1
    
    with open("{}/nonselfie_false_positives.txt".format(wrong_predictions_folder), "r") as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if i > 0:
                wrong_prediction = line.split("wrong_prediction")[1]
                if "selfie" in wrong_prediction:
                    wrong_predictions_level_0['people'] += 1
                    wrong_predictions_nonselfie_level_1['selfie'] +=1       
                elif ('cat' in wrong_prediction or 'dog' in wrong_prediction or 'cartoon' in wrong_prediction 
                    or 'drawings' in wrong_prediction or 'engraving' in wrong_prediction or 
                    'iconography' in wrong_prediction or 'painting' in wrong_prediction or 'sculpture' in wrong_prediction):
                    wrong_predictions_level_0['other'] += 1
                    if 'cat' in wrong_prediction or 'dog' in wrong_prediction:
                        wrong_predictions_nonselfie_level_1['pet'] += 1
                    elif 'cartoon' in wrong_prediction:
                        wrong_predictions_nonselfie_level_1['cartoon'] += 1
                    else:
                        wrong_predictions_nonselfie_level_1['art'] += 1
                else:
                    wrong_predictions_level_0['scenes'] += 1
                    wrong_predictions_nonselfie_level_1['scenes'] += 1
    print(wrong_predictions_level_0)
    print(wrong_predictions_selfie_level_1)
    print(wrong_predictions_nonselfie_level_1)


def calculate_scenes_false_positives_for_hierarchy_classes_v3(wrong_predictions_folder):
    scenes = ["shopping_and_dining", "workplace", "home_or_hotel",
                    "transportation", "sports_and_leisure", "cultural",
                    "water_ice_snow", "mountains_hills_desert_sky",
                    "forest_field_jungle", "man-made_elements",
                    "transportation", "cultural_or_historical_building_place",
                    "sportsfields_parks_leisure_spaces", "industrial_and_construction",
                    "houses_cabins_gardens_and_farms", "commercial_buildings"]

    wrong_predictions_level_0 = { 'people': 0, 'scenes': 0, 'other': 0}
    wrong_predictions_level_1 = {'people': 0, 'scenes': 0, 'pet': 0, 'cartoon': 0, 'art': 0}
    for scene in scenes:
        with open("{}/{}_false_positives.txt".format(wrong_predictions_folder, scene), "r") as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                if i > 0:
                    img_path = line.split(":")[1].split(",")[0].replace("'","").strip()
                    wrong_prediction = line.split("wrong_prediction")[1]
                    if "people" in wrong_prediction:
                        wrong_predictions_level_0['people'] += 1
                        wrong_predictions_level_1['people'] +=1          
                    elif ('cat' in wrong_prediction or 'dog' in wrong_prediction or 'cartoon' in wrong_prediction 
                        or 'drawings' in wrong_prediction or 'engraving' in wrong_prediction or 
                        'iconography' in wrong_prediction or 'painting' in wrong_prediction or 'sculpture' in wrong_prediction):
                        wrong_predictions_level_0['other'] += 1
                        if 'cat' in wrong_prediction or 'dog' in wrong_prediction:
                            wrong_predictions_level_1['pet'] += 1
                        elif 'cartoon' in wrong_prediction:
                            wrong_predictions_level_1['cartoon'] += 1
                        else:
                            wrong_predictions_level_1['art'] += 1
                    else:
                        wrong_predictions_level_0['scenes'] += 1
                        wrong_predictions_level_1['scenes'] += 1
    print(wrong_predictions_level_0)
    print(wrong_predictions_level_1)

def calculate_people_false_positives_for_hierarchy_classes(dataset_folder, save_folder, wrong_predictions_folder):

    scenes = ["shopping_and_dining", "workplace", "home_or_hotel",
                    "transportation", "sports_and_leisure", "cultural",
                    "water_ice_snow", "mountains_hills_desert_sky",
                    "forest_field_jungle", "man-made_elements",
                    "transportation", "cultural_or_historical_building_place",
                    "sportsfields_parks_leisure_spaces", "industrial_and_construction",
                    "houses_cabins_gardens_and_farms", "commercial_buildings"]
    
    wrong_predictions_level_0 = { 'people': 0, 'scenes': 0, 'other': 0}
    wrong_predictions_selfie_level_1 = {'selfie': 0, 'nonselfie': 0, 'scenes': 0, 'pet': 0, 'cartoon': 0, 'art': 0}
    wrong_predictions_nonselfie_level_1 = {'selfie': 0, 'nonselfie': 0, 'scenes': 0, 'pet': 0, 'cartoon': 0, 'art': 0}
    with open("{}/selfie_false_positives.txt".format(wrong_predictions_folder), "r") as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if i > 0:
                wrong_prediction = line.split("wrong_prediction")[1]
                if "nonselfie" in wrong_prediction:
                    wrong_predictions_level_0['people'] += 1
                    wrong_predictions_selfie_level_1['nonselfie'] +=1       
                elif ('cat' in wrong_prediction or 'dog' in wrong_prediction or 'cartoon' in wrong_prediction 
                    or 'drawings' in wrong_prediction or 'engraving' in wrong_prediction or 
                    'iconography' in wrong_prediction or 'painting' in wrong_prediction or 'sculpture' in wrong_prediction):
                    wrong_predictions_level_0['other'] += 1
                    if 'cat' in wrong_prediction or 'dog' in wrong_prediction:
                        wrong_predictions_selfie_level_1['pet'] += 1
                    elif 'cartoon' in wrong_prediction:
                        wrong_predictions_selfie_level_1['cartoon'] += 1
                    else:
                        wrong_predictions_selfie_level_1['art'] += 1
                else:
                    wrong_predictions_level_0['scenes'] += 1
                    wrong_predictions_selfie_level_1['scenes'] += 1
    
    with open("{}/nonselfie_false_positives.txt".format(wrong_predictions_folder), "r") as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if i > 0:
                wrong_prediction = line.split("wrong_prediction")[1]
                if "selfie" in wrong_prediction:
                    wrong_predictions_level_0['people'] += 1
                    wrong_predictions_nonselfie_level_1['selfie'] +=1       
                elif ('cat' in wrong_prediction or 'dog' in wrong_prediction or 'cartoon' in wrong_prediction 
                    or 'drawings' in wrong_prediction or 'engraving' in wrong_prediction or 
                    'iconography' in wrong_prediction or 'painting' in wrong_prediction or 'sculpture' in wrong_prediction):
                    wrong_predictions_level_0['other'] += 1
                    if 'cat' in wrong_prediction or 'dog' in wrong_prediction:
                        wrong_predictions_nonselfie_level_1['pet'] += 1
                    elif 'cartoon' in wrong_prediction:
                        wrong_predictions_nonselfie_level_1['cartoon'] += 1
                    else:
                        wrong_predictions_nonselfie_level_1['art'] += 1
                else:
                    wrong_predictions_level_0['scenes'] += 1
                    wrong_predictions_nonselfie_level_1['scenes'] += 1
    print(wrong_predictions_level_0)
    print(wrong_predictions_selfie_level_1)
    print(wrong_predictions_nonselfie_level_1)


def calculate_people_false_positives_for_hierarchy_classes_v3(dataset_folder, save_folder, wrong_predictions_folder):

    scenes = ["shopping_and_dining", "workplace", "home_or_hotel",
                    "transportation", "sports_and_leisure", "cultural",
                    "water_ice_snow", "mountains_hills_desert_sky",
                    "forest_field_jungle", "man-made_elements",
                    "transportation", "cultural_or_historical_building_place",
                    "sportsfields_parks_leisure_spaces", "industrial_and_construction",
                    "houses_cabins_gardens_and_farms", "commercial_buildings"]
    
    wrong_predictions_level_0 = { 'people': 0, 'scenes': 0, 'other': 0}
    wrong_predictions_level_1 = {'people': 0, 'scenes': 0, 'pet': 0, 'cartoon': 0, 'art': 0}
    with open("{}/people_false_positives.txt".format(wrong_predictions_folder), "r") as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if i > 0:
                wrong_prediction = line.split("wrong_prediction")[1]
                if ('cat' in wrong_prediction or 'dog' in wrong_prediction or 'cartoon' in wrong_prediction 
                    or 'drawings' in wrong_prediction or 'engraving' in wrong_prediction or 
                    'iconography' in wrong_prediction or 'painting' in wrong_prediction or 'sculpture' in wrong_prediction):
                    wrong_predictions_level_0['other'] += 1
                    if 'cat' in wrong_prediction or 'dog' in wrong_prediction:
                        wrong_predictions_level_1['pet'] += 1
                    elif 'cartoon' in wrong_prediction:
                        wrong_predictions_level_1['cartoon'] += 1
                    else:
                        wrong_predictions_level_1['art'] += 1
                else:
                    wrong_predictions_level_0['scenes'] += 1
                    wrong_predictions_level_1['scenes'] += 1
    
    print(wrong_predictions_level_0)
    print(wrong_predictions_level_1)