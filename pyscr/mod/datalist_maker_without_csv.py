import glob
import os

def makeDataListWithoutCsv(dir_list):
    target_extension = ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']
    data_list = []
    for dir in dir_list:
        file_list = glob.glob(dir + '/**', recursive=True)
        for file_path in file_list:
            extension = file_path.split('.')[-1]
            if extension in target_extension:
                data_list.append([0, 0, 1, file_path])
    return data_list


def test():
    dir_list = [os.environ['HOME'] + '/dataset/img_align_celeba']
    data_list = makeDataListWithoutCsv(dir_list)
    ## debug
    # print(data_list)
    print("len(data_list) = ", len(data_list))
    print("example0: ", data_list[0][:3], data_list[0][3:])
    print("example1: ", data_list[1][:3], data_list[1][3:])

if __name__ == '__main__':
    test()