import os

def rename_tests_imgs(folder_path):
    imgs = os.listdir(folder_path)
    for i in range(len(imgs)):
        new = folder_path + '/{:06}.jpg'.format(i)
        imgs[i] = folder_path + '/' + imgs[i]
        os.rename(imgs[i], new)

if __name__ == '__main__':
    img_folder = './tests'
    rename_tests_imgs(img_folder)