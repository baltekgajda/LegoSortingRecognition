from multiprocessing import freeze_support

from data_loader import load_data, show_random_images

if __name__ == '__main__':
    freeze_support()
    train_loader, valid_loader, test_loader = load_data()
    show_random_images(train_loader)
