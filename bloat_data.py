import glob
import os
import random
import shutil
from math import floor

# Path to directory where the downloaded data is
DATA_DIR = ""
# Path to directory where new dataset should be created
TARGET_DIR = ""

# Name of the sub-directory containing real images in DATA_DIR
REAL_DIR = "real"
# Name of the sub-directory containing fake images in DATA_DIR
FAKE_DIR = "df"
# Amount of data that should be put in the test set
PROP = 0.10


def create_target_dirs():
    r_train = os.path.join(TARGET_DIR, "train", REAL_DIR)
    f_train = os.path.join(TARGET_DIR, "train", FAKE_DIR)

    os.makedirs(r_train)
    os.makedirs(f_train)

    r_test = os.path.join(TARGET_DIR, "test", REAL_DIR)
    f_test = os.path.join(TARGET_DIR, "test", FAKE_DIR)

    os.makedirs(r_test)
    os.makedirs(f_test)


def main():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"{DATA_DIR} does not exist")

    if not os.path.isdir(DATA_DIR):
        raise NotADirectoryError(f"{DATA_DIR} is not a directory")

    shutil.rmtree(TARGET_DIR, ignore_errors=True)

    path = os.path.join(DATA_DIR, "**", "*.jpg")
    imgs = glob.glob(path, recursive=True)

    num_imgs = len(imgs)

    test_size = floor(len(imgs) * PROP)
    train_size = num_imgs - test_size

    while True:
        print(f"Creating directories at {TARGET_DIR}....")
        create_target_dirs()
        selected = random.sample(imgs, test_size)

        train_r_len = len(
            [1 for img in imgs if REAL_DIR in img and img not in selected]
        )
        train_f_len = train_size - train_r_len
        test_r_len = len([1 for img in selected if REAL_DIR in img])
        test_f_len = test_size - test_r_len

        print(f"Found {num_imgs} images in {DATA_DIR}....")
        print(f"Copying {train_size} files to {TARGET_DIR}/train/....")
        print(f"Copying {test_size} files to {TARGET_DIR}/test/....")
        print(
            f"The training set will have {train_r_len} real images and {train_f_len} fake images...."
        )
        print(
            f"The test set will have {test_r_len} real images and {test_f_len} fake images..."
        )

        for img in imgs:
            leaf_dir = REAL_DIR if REAL_DIR in img else FAKE_DIR
            intr_dir = "test" if img in selected else "train"
            dst = os.path.join(TARGET_DIR, intr_dir, leaf_dir)
            shutil.copy(img, dst)

        prompt = input("Try again? [Yy/Nn] ")
        if prompt in ["n", "N"]:
            print("Exiting....")
            break

        print("\n\nTrying again....")
        print(f"Deleting {TARGET_DIR}....\n")
        shutil.rmtree(TARGET_DIR)


if __name__ == "__main__":
    main()
