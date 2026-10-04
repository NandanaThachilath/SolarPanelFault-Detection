import os
import cv2
import numpy as np
import albumentations as A
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm
import math

# Define class names and their indices
CLASS_NAMES = {
    0: 'hot_spot',
    1: 'scratch',
    2: 'no_electricity',
    3: 'black_border',
    4: 'broken'
}

# Configuration
INPUT_IMAGE_DIR = r"C:\Users\Admin\Documents\D14 PV Multi Defect\D14 PV Multi Defect\PV-Multi-Defect-main\PV-Multi-Defect-main\JPEGImages"
INPUT_ANNOTATION_DIR = r"C:\Users\Admin\Documents\D14 PV Multi Defect\D14 PV Multi Defect\PV-Multi-Defect-main\PV-Multi-Defect-main\Annotations"
OUTPUT_DIR = r"C:\Users\Admin\Documents\PV_Multi_Defect_Augmented"

# Output subdirectories
OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_DIR, "JPEGImages")
OUTPUT_ANNOT_DIR = os.path.join(OUTPUT_DIR, "Annotations")
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_ANNOT_DIR, exist_ok=True)

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.2),
    A.Resize(512, 640)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def parse_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    objects = []
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        objects.append({
            'class': class_name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return filename, width, height, objects

def create_xml(filename, width, height, objects, output_dir):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = filename
    ET.SubElement(root, "folder").text = "JPEGImages"
    
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    
    for obj in objects:
        obj_elem = ET.SubElement(root, "object")
        ET.SubElement(obj_elem, "name").text = obj['class']
        ET.SubElement(obj_elem, "pose").text = "Unspecified"
        ET.SubElement(obj_elem, "truncated").text = "0"
        ET.SubElement(obj_elem, "difficult").text = "0"
        
        bbox = ET.SubElement(obj_elem, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(round(obj['bbox'][0]))
        ET.SubElement(bbox, "ymin").text = str(round(obj['bbox'][1]))
        ET.SubElement(bbox, "xmax").text = str(round(obj['bbox'][2]))
        ET.SubElement(bbox, "ymax").text = str(round(obj['bbox'][3]))
    
    tree = ET.ElementTree(root)
    xml_filename = os.path.splitext(filename)[0] + ".xml"
    tree.write(os.path.join(output_dir, xml_filename))

def clip_bbox(bbox, width, height):
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, min(xmin, width))
    ymin = max(0, min(ymin, height))
    xmax = max(0, min(xmax, width))
    ymax = max(0, min(ymax, height))
    
    if xmin >= xmax or ymin >= ymax:
        return None
    
    return [xmin, ymin, xmax, ymax]

def count_instances_per_class(objects):
    counts = defaultdict(int)
    for obj in objects:
        counts[obj['class']] += 1
    return counts

def get_class_stats(annotation_dir):
    class_image_counts = {name: 0 for name in CLASS_NAMES.values()}
    class_instance_counts = {name: 0 for name in CLASS_NAMES.values()}
    total_images = 0
    
    print("\n📊 Collecting dataset statistics...")
    xml_files = [f for f in os.listdir(annotation_dir) if f.endswith('.xml')]
    
    for xml_file in tqdm(xml_files, desc="Processing XML files"):
        xml_path = os.path.join(annotation_dir, xml_file)
        try:
            _, _, _, objects = parse_annotation(xml_path)
            image_classes = set(obj['class'] for obj in objects)
            
            for class_name in image_classes:
                if class_name in CLASS_NAMES.values():
                    class_image_counts[class_name] += 1
            
            instance_counts = count_instances_per_class(objects)
            for class_name, count in instance_counts.items():
                if class_name in CLASS_NAMES.values():
                    class_instance_counts[class_name] += count
            
            total_images += 1
        except Exception as e:
            print(f"⚠️ Error processing {xml_file}: {str(e)}")
    
    return class_image_counts, class_instance_counts, total_images

def main():
    # Get original stats
    orig_image_counts, orig_instance_counts, orig_total_images = get_class_stats(INPUT_ANNOTATION_DIR)
    
    # Collect all valid images with annotations
    valid_images = []
    print("\n🔍 Collecting valid images with annotations...")
    xml_files = [f for f in os.listdir(INPUT_ANNOTATION_DIR) if f.endswith('.xml')]
    
    for xml_file in tqdm(xml_files, desc="Processing annotations"):
        xml_path = os.path.join(INPUT_ANNOTATION_DIR, xml_file)
        try:
            filename, width, height, objects = parse_annotation(xml_path)
            image_path = os.path.join(INPUT_IMAGE_DIR, filename)
            
            if not os.path.exists(image_path):
                continue
                
            valid_images.append({
                'image_path': image_path,
                'xml_path': xml_path,
                'filename': filename,
                'width': width,
                'height': height,
                'objects': objects
            })
        except Exception as e:
            print(f"⚠️ Error processing {xml_file}: {str(e)}")
    
    # Organize images by class
    class_images = {class_name: [] for class_name in CLASS_NAMES.values()}
    
    for img_data in valid_images:
        image_classes = set(obj['class'] for obj in img_data['objects'])
        for class_name in image_classes:
            if class_name in CLASS_NAMES.values():
                class_images[class_name].append(img_data)
    
    # Phase 1: Balance classes
    print("\n⚖️ Balancing classes...")
    output_images = set()
    augmentation_count = 0
    
    # Set custom targets - hotspot reduced to 2500
    class_targets = {
        'hot_spot': 2500,  # Changed to 2500
        'scratch': 3000,
        'no_electricity': 1500,
        'black_border': 1500,
        'broken': 1500
    }
    
    for class_name in CLASS_NAMES.values():
        class_list = class_images[class_name]
        target = class_targets[class_name]
        num_needed = target - len(class_list)
        
        print(f"\nProcessing {class_name}:")
        print(f"  Original: {len(class_list)} images")
        print(f"  Target: {target} images")
        print(f"  Needed: {max(0, num_needed)} augmentations")
        
        # Copy existing images
        for img_data in class_list:
            if img_data['filename'] not in output_images:
                # Copy image
                src_img = img_data['image_path']
                dst_img = os.path.join(OUTPUT_IMAGE_DIR, img_data['filename'])
                shutil.copy2(src_img, dst_img)
                
                # Create XML annotation
                create_xml(
                    img_data['filename'],
                    img_data['width'],
                    img_data['height'],
                    img_data['objects'],
                    OUTPUT_ANNOT_DIR
                )
                output_images.add(img_data['filename'])
        
        # Augment if needed
        if num_needed > 0:
            print(f"  Augmenting {num_needed} images...")
            for i in tqdm(range(num_needed), desc=f"Augmenting {class_name}"):
                # Select random source image from this class
                src_data = random.choice(class_list)
                image = cv2.imread(src_data['image_path'])
                
                if image is None:
                    continue
                
                # Prepare data for augmentation
                bboxes = []
                class_labels = []
                for obj in src_data['objects']:
                    bboxes.append(obj['bbox'])
                    class_labels.append(obj['class'])
                
                # Apply augmentation
                try:
                    augmented = transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_class_labels = augmented['class_labels']
                except Exception as e:
                    print(f"⚠️ Augmentation error: {str(e)}")
                    continue
                
                # Process bounding boxes
                new_objects = []
                height, width = aug_image.shape[:2]
                for bbox, class_label in zip(aug_bboxes, aug_class_labels):
                    clipped_bbox = clip_bbox(bbox, width, height)
                    if clipped_bbox:
                        new_objects.append({
                            'class': class_label,
                            'bbox': clipped_bbox
                        })
                
                # Skip if no valid objects
                if not new_objects:
                    continue
                
                # Create new filename
                new_filename = f"aug_{augmentation_count}_{src_data['filename']}"
                augmentation_count += 1
                
                # Save augmented image
                new_image_path = os.path.join(OUTPUT_IMAGE_DIR, new_filename)
                cv2.imwrite(new_image_path, aug_image)
                
                # Create XML annotation
                create_xml(
                    new_filename,
                    width,
                    height,
                    new_objects,
                    OUTPUT_ANNOT_DIR
                )
                output_images.add(new_filename)
    
    # Final statistics
    print("\n📊 Final Dataset Statistics:")
    final_image_counts, final_instance_counts, final_total_images = get_class_stats(OUTPUT_ANNOT_DIR)
    
    print(f"\n{'Class':<15} | {'Original Images':>15} | {'Final Images':>15} | {'Original Instances':>15} | {'Final Instances':>15}")
    print("-" * 80)
    for class_name in CLASS_NAMES.values():
        print(f"{class_name:<15} | {orig_image_counts[class_name]:>15} | {final_image_counts[class_name]:>15} | {orig_instance_counts[class_name]:>15} | {final_instance_counts[class_name]:>15}")
    
    print("\n✅ Dataset augmentation complete!")

if __name__ == "__main__":
    main()