import json
import os
import requests
from pycocotools.coco import COCO
from tqdm import tqdm

def get_coco_categories():
    """Get all available COCO categories"""
    # Download annotations temporarily to get categories
    url = 'http://images.cocodataset.org/annotations/instances_train2017.json'
    response = requests.get(url)
    data = json.loads(response.content)
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    return categories

def download_coco_subset_by_categories(
    categories,
    year="2017",
    set_name="train",
    output_dir="coco_subset"
):
    """
    Download COCO dataset images for specific categories.
    
    Args:
        categories: List of category names (e.g., ['person', 'dog', 'car'])
        year: Dataset year (2014 or 2017)
        set_name: Dataset split (train or val)
        output_dir: Directory to save the dataset
    """
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    ann_dir = os.path.join(output_dir, 'annotations')
    img_dir = os.path.join(output_dir, f'{set_name}{year}')
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # Download and load annotations
    ann_file = f'instances_{set_name}{year}.json'
    ann_url = f'http://images.cocodataset.org/annotations/{ann_file}'
    ann_path = os.path.join(ann_dir, ann_file)
    
    if not os.path.exists(ann_path):
        print(f"Downloading annotations from {ann_url}")
        response = requests.get(ann_url)
        with open(ann_path, 'wb') as f:
            f.write(response.content)
    
    coco = COCO(ann_path)
    
    # Get category IDs for requested categories
    cat_ids = []
    for category in categories:
        cat_id = coco.getCatIds(catNms=[category])
        if cat_id:
            cat_ids.extend(cat_id)
        else:
            print(f"Warning: Category '{category}' not found in COCO dataset")
    
    if not cat_ids:
        raise ValueError("None of the specified categories were found in COCO dataset")
    
    # Get image IDs containing the specified categories
    img_ids = []
    for cat_id in cat_ids:
        img_ids.extend(coco.getImgIds(catIds=[cat_id]))
    img_ids = list(set(img_ids))  # Remove duplicates
    
    # Create subset annotations
    subset_anns = {
        'info': coco.dataset['info'],
        'licenses': coco.dataset['licenses'],
        'categories': [cat for cat in coco.dataset['categories'] if cat['id'] in cat_ids],
        'images': [],
        'annotations': []
    }
    
    # Download images and collect annotations
    print(f"Downloading {len(img_ids)} images containing specified categories...")
    for img_id in tqdm(img_ids):
        # Get image info and annotations
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        
        if not anns:  # Skip images with no annotations for specified categories
            continue
        
        # Download image
        img_url = f"http://images.cocodataset.org/{set_name}{year}/{img_info['file_name']}"
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            try:
                response = requests.get(img_url)
                response.raise_for_status()
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                    
                # Only add to annotations if image download successful
                subset_anns['images'].append(img_info)
                subset_anns['annotations'].extend(anns)
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {img_info['file_name']}: {e}")
                continue
    
    # Save subset annotations
    subset_ann_path = os.path.join(ann_dir, f'instances_subset_{set_name}{year}.json')
    with open(subset_ann_path, 'w') as f:
        json.dump(subset_anns, f)
    
    print(f"\nDataset summary:")
    print(f"Downloaded {len(subset_anns['images'])} images")
    print(f"Saved {len(subset_anns['annotations'])} annotations")
    print(f"Categories included: {[cat['name'] for cat in subset_anns['categories']]}")
    print(f"Dataset saved to {output_dir}")
    
    return output_dir

def get_dataset_stats(ann_file):
    """Print basic statistics about the dataset"""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nDetailed dataset statistics:")
    print(f"Number of images: {len(data['images'])}")
    print(f"Number of annotations: {len(data['annotations'])}")
    print(f"Number of categories: {len(data['categories'])}")
    
    # Count instances per category
    category_counts = {}
    for ann in data['annotations']:
        cat_id = ann['category_id']
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
    
    print("\nInstances per category:")
    for cat in data['categories']:
        count = category_counts.get(cat['id'], 0)
        print(f"{cat['name']}: {count}")



if __name__ == "__main__":
    # First, you can see all available COCO categories
    categories = get_coco_categories()
    print("Available categories:", list(categories.values()))

# Then download images for your chosen categories
    selected_categories = ['person', 'car', 'dog', 'cat', 'horse', 'bird', 'bicycle', 'boat', 'traffic light', 'motorcycle']
    output_dir = download_coco_subset_by_categories(
        categories=selected_categories,
        year="2017",
        set_name="train"
    )

# Get statistics about the downloaded subset
    ann_file = os.path.join(output_dir, 'annotations', 'instances_subset_train2017.json')
    get_dataset_stats(ann_file)
