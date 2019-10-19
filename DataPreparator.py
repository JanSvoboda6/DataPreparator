import matplotlib.pyplot as plt
import pydicom
import os
from shutil import copy2


def prepare_data(unprepared_data_path, prepared_data_path):
    """
    Prepare data for further processing

    :param unprepared_data_path: input directory
    :param prepared_data_path:   output directory
    """
    if not os.path.exists(unprepared_data_path):
        raise FileNotFoundError('Input directory does not exist!')
    if not os.path.exists(prepared_data_path):
        raise FileNotFoundError('Output directory does not exist!')

    num_of_unprepared_images = len(os.listdir(unprepared_data_path))
    print('Number of unprepared images in input directory: ' + str(num_of_unprepared_images))
    num_of_prepared_images = len(os.listdir(prepared_data_path))
    print('Number of already prepared images in output directory: ' + str(num_of_prepared_images))
    print('''To exit script type 'exit'.\n''')
    # Plot setup
    plt.ion()
    plt.show()
    plt.gray()
    for i in range(1, num_of_unprepared_images):
        image_number = str(i).zfill(4)
        filename = os.path.abspath(unprepared_data_path + '/IMG' + image_number + '.dcm')
        image = pydicom.dcmread(filename)
        plt.imshow(image.pixel_array)
        plt.pause(0.001)
        confirmation = input('Current image: ' + image_number + '\t\tIs image relevant?')
        # If current image is relevant, create copy of it in directory for prepared data
        if confirmation == 'y':
            num_of_prepared_images += 1
            destination = os.path.abspath(prepared_data_path + '/' + str(num_of_prepared_images) + '.dcm')
            copy2(filename, destination)
        elif confirmation == 'exit':
            return
        plt.clf()


if __name__ == '__main__':
    init_directory = input('Enter input directory: ')
    out_directory = input('Enter output directory: ')
    prepare_data(init_directory,out_directory)
    #'C:/Users/svobo/Desktop/BachThesis/DICOMPOKUS4'
    #'C:/Users/svobo/Desktop/BachThesis/PreparedData'