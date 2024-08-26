import os
import json
import base64
from path_src import detections_path, pings_path

os.makedirs(pings_path + "img/", exist_ok=True)

def read_detections(values,max):
    # list of all json files and sort (after time)
    list_file_json = sorted(os.listdir(detections_path))
    list_file_json = list_file_json
    index = 0
    n_images = 0
    # read one file json
    one_file_name = list_file_json[values]
    json_path = detections_path + one_file_name
    print(json_path)
    with open(json_path) as json_file:
        json_load = json.load(json_file)
    # loop through the elements in the selected file
    for detection in json_load['detections']:
        img = detection["frame_content"]
        name = detection["id"]
        # is detection visible on api.credo.science
        visible = detection["visible"]
        # if Yes continue, but only for max img
        if visible is True and n_images < max:
            print(visible)
            image = img.encode('ascii')
            # save img as png
            adres = pings_path + "img/" + str(name) + ".png"
            with open(adres, "wb") as fh:
                fh.write(base64.decodebytes(image))
            n_images += 1


def main():
    read_detections(1,200)


if __name__ == '__main__':
    main()
