"""Script to check a folder full of images and see if there are any 
that TensorFlow cannot read. If there are, the paths to these "bad" 
images will be written to a CSV file in the target folder.

Intended as an optional diagnostic if the program crashes while 
generating class scores, as this can be due to corrupted or incomplete
image files.

Run from the command line (activate the r-reticulate environment first):

conda activate r-reticulate
python [path to CheckImageFiles.py] [path to target dir]

By default this script will look for images in the directory tree rooted at
the folder the script is in. Optionally, you can provide the path to the 
target directory as a command-line argument.
"""

import multiprocessing as mp
import os
import sys
from tensorflow.keras.preprocessing.image import load_img


def checkImageFile(image_path):
    """Verify that tensorflow can load an image file."""
    try:
        img = load_img(image_path)
        return True
    except:
        return False


class imageChecker(mp.Process):
    """Worker process to check for bad image files.
    """
    def __init__(self, in_queue, bad_queue):
        mp.Process.__init__(self)
        self.in_queue = in_queue
        self.bad_queue = bad_queue
        
    def run(self):
        while True:
            img_path = self.in_queue.get()
            img_loads = checkImageFile(img_path)
            if not img_loads:
                self.bad_queue.put(img_path)
            self.in_queue.task_done()


def checkImages(top_dir, n_workers=0):
    """Check all the .png images in a folder.
    
    Arguments:
    - top_dir: the directory containing images to be checked
    - n_workers: number of worker processes to use. Defaults to number
    of available cores.
    
    Returns:
    - bad_imgs: a sorted list of paths of images that could not be 
    loaded for any reason
    """
    if n_workers == 0:
        n_workers = mp.cpu_count()
    else:
        n_workers = min(mp.cpu_count, n_workers)

    # pngs = pycnet.file.findFiles(top_dir, ".png")
    
    pngs = []
    for root, dirs, files in os.walk(top_dir):
        for file in files:
            if file[-4:] == ".png":
                pngs.append(os.path.join(root, file))
    pngs.sort()
    
    print("\nFound {0} PNG files under {1}.\n".format(len(pngs), top_dir))
    print("Checking images... ", end='')
    
    img_queue, bad_img_queue = mp.JoinableQueue(), mp.Queue()
    for i in pngs:
        img_queue.put(i)
        
    for j in range(n_workers):
        worker = imageChecker(img_queue, bad_img_queue)
        worker.daemon = True
        worker.start()
        
    img_queue.join()
    
    print("done.")
    
    n_bad_imgs = bad_img_queue.qsize()
    bad_imgs = []

    if n_bad_imgs > 0:
        print("\n{0} images could not be loaded.".format(n_bad_imgs))        
        while bad_img_queue.qsize() > 0:
            bad_imgs.append(bad_img_queue.get())
        with open(os.path.join(top_dir, "Bad_Images.csv"), 'w') as outfile:
            outfile.write("Path\n")
            outfile.write('\n'.join(sorted(bad_imgs)))
    else:
        print("\nAll images loaded successfully. No errors detected.")
    
    return sorted(bad_imgs)


def main():
    try:
        indir = sys.argv[1]
    except:
        indir = sys.path[0]

    bad_images = checkImages(indir)


if __name__ == "__main__":
    main()