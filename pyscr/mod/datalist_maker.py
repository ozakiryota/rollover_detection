import csv
import os

def makeDataList(dir_list, csv_name):
    data_list = []
    for dir in dir_list:
        csv_path = os.path.join(dir, csv_name)
        with open(csv_path) as data_csv:
            reader = csv.reader(data_csv)
            for row in reader:
                row[3] = os.path.join(dir, row[3])
                data_list.append(row)
    return data_list


def test():
    dir_list = [os.environ['HOME'] + "/dataset/rollover_detection/airsim/sample"]
    csv_name = "imu_camera.csv"
    data_list = makeDataList(dir_list, csv_name)
    ## debug
    # print(data_list)
    print("len(data_list) = ", len(data_list))
    print("example0: ", data_list[0][:3], data_list[0][3:])
    print("example1: ", data_list[1][:3], data_list[1][3:])

if __name__ == '__main__':
    test()