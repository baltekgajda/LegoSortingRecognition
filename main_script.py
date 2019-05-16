from multiprocessing import freeze_support

from data_loader import load_data, show_random_images

if __name__ == '__main__':
    freeze_support()
    data_loader = load_data()
    show_random_images(data_loader)
