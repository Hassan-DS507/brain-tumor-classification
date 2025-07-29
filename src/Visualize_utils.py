from collections import Counter
import matplotlib.pyplot as plt
def plot_class_distribution(labels, title):
    counter = Counter(labels)
    classes = list(counter.keys())
    counts = list(counter.values())

    plt.figure(figsize=(10, 5))
    bars = plt.bar(classes, counts, color='skyblue')
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of Images")

    # Add counts on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5, f'{int(height)}', ha='center')

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict

def plot_random_images_by_class(X, y, class_names=None, samples_per_class=5, figsize=(15, 10)):
    """
    X: array-like of images
    y: labels (list or array)
    class_names: list of class names (optional)
    samples_per_class: number of images to show per class
    figsize: size of the figure
    """
    y = np.array(y)
    class_to_images = defaultdict(list)
    

    for img, label in zip(X, y):
        class_to_images[label].append(img)
    
    classes = sorted(class_to_images.keys())
    num_classes = len(classes)
    

    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=figsize)
    fig.suptitle('Random Images per Class', fontsize=18)

    for row, cls in enumerate(classes):
        imgs = random.sample(class_to_images[cls], min(samples_per_class, len(class_to_images[cls])))
        for col in range(samples_per_class):
            ax = axes[row, col] if num_classes > 1 else axes[col]
            if col < len(imgs):
                img = imgs[col]
                ax.imshow(img.astype(np.uint8))
                ax.axis('off')
                if col == 0:
                    name = class_names[cls] if class_names else str(cls)
                    ax.set_title(f"Class {name}", fontsize=10)
            else:
                ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
