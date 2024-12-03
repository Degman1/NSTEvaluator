import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools


def create_image_table_from_dirs(directories, row_labels, output_file, label_offset=-0.02, flip_axes=False):
    """
    Generates a table of images with labeled rows or columns, pulling images from directories,
    and saves the figure. Automatically repeats images in directories with fewer files.
    
    Args:
        directories (list): List of directories, each containing images for one row or column.
        row_labels (list): List of row (or column) labels.
        output_file (str): Path where the figure should be saved (e.g., 'output.png').
        label_offset (float): Offset for row or column labels.
        flip_axes (bool): If True, flip rows and columns (transpose the table).
    """
    # Load all images from each directory and repeat as needed
    images = []
    max_images = 0

    for directory in directories:
        image_files = sorted(
            [os.path.join(directory, file) for file in os.listdir(directory)
             if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        )
        max_images = max(max_images, len(image_files))  # Track the maximum number of images
        images.append(image_files)

    # Ensure all rows/columns have the same number of images by repeating images as necessary
    for i in range(len(images)):
        if len(images[i]) < max_images:
            images[i] = list(itertools.islice(itertools.cycle(images[i]), max_images))

    n_rows = len(images)
    n_cols = max_images

    if flip_axes:
        # Flip rows and columns
        images = list(zip(*images))  # Transpose the list of lists
        n_rows, n_cols = n_cols, n_rows

    assert len(row_labels) == (n_rows if not flip_axes else n_cols), \
        "Number of labels must match the number of rows or columns after flipping."

    # Create the figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    # If there is only one row or column, axes might be 1D, so ensure it's 2D
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    # Plot each image in its respective cell
    for i in range(n_rows):
        for j in range(n_cols):
            img = mpimg.imread(images[i][j])  # Read the image
            axes[i][j].imshow(img)
            axes[i][j].axis("off")  # Hide axes

    # Add row or column labels
    if not flip_axes:
        # Add row labels to the left side of each row
        for i in range(n_rows):
            fig.text(label_offset, (n_rows - i - 0.5) / n_rows, row_labels[i], va='center', rotation='vertical', fontsize=30)
    else:
        # Add column labels at the top of each column
        for j in range(n_cols):
            fig.text((j + 0.5) / n_cols, 1 + label_offset, row_labels[j], ha='center', rotation='horizontal', fontsize=30)

    # Adjust layout to ensure proper spacing
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.05)  # Adjust margins to make space for labels
    plt.tight_layout()

    # Save the figure to the specified file path
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to {output_file}")


if __name__ == "__main__":
    directories = [
        "images/output_exp1/preprocessed_content",
        "images/exp12_base_style_images_cropped",
        "images/output_exp1/preprocessed_style",
        "images/output_exp1/Linear",
        "images/output_exp1/ArbitraryMultiAdaptation_alpha60",
        "images/output_exp1/ArbitraryMultiAdaptation_alpha90",
        "images/output_exp1/StyleID",
    ]
    row_labels = ["Content", "Baseline", "Style", "Linear", "AMA60", "AMA90", "SID"]
    output_file = "images/exp1_table.png"  # Path where the figure will be saved
    
    create_image_table_from_dirs(directories, row_labels, output_file)
    
    directories = [
        "images/output_exp2/preprocessed_content",
        "images/exp12_base_style_images_cropped",
        "images/output_exp2/preprocessed_style",
        "images/output_exp2/Linear",
        "images/output_exp2/ArbitraryMultiAdaptation_alpha60",
        "images/output_exp2/ArbitraryMultiAdaptation_alpha90",
        "images/output_exp2/StyleID",
    ]
    row_labels = ["Content", "Baseline", "Style", "Linear", "AMA60", "AMA90", "SID"]
    output_file = "images/exp2_table.png"  # Path where the figure will be saved
    
    create_image_table_from_dirs(directories, row_labels, output_file)
    
    directories = [
        "images/output_exp3/preprocessed_content",
        "images/exp3_base_style_images_cropped",
        "images/output_exp3/preprocessed_style",
        "images/output_exp3/Linear",
        "images/output_exp3/ArbitraryMultiAdaptation_alpha60",
        "images/output_exp3/ArbitraryMultiAdaptation_alpha90",
        "images/output_exp3/StyleID",
    ]
    row_labels = ["Content", "Baseline", "Style", "Linear", "AMA60", "AMA90", "SID"]
    output_file = "images/exp3_table.png"  # Path where the figure will be saved
    
    create_image_table_from_dirs(directories, row_labels, output_file, label_offset=-0.04)
    
    directories = [
        "images/output_exp4/preprocessed_content",
        "images/output_exp4/preprocessed_style",
        "images/output_exp4/Linear",
        "images/output_exp4/ArbitraryMultiAdaptation_alpha60",
        "images/output_exp4/ArbitraryMultiAdaptation_alpha90",
        "images/output_exp4/StyleID",
    ]
    row_labels = ["Content", "Style", "Linear", "AMA60", "AMA90", "SID"]
    output_file = "images/exp4_table.png"  # Path where the figure will be saved

    create_image_table_from_dirs(directories, row_labels, output_file, label_offset=-0.035)
