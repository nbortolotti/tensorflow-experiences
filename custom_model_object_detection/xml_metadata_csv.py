"""
Usage:
  python xml_metadata_csv.py

# validate dependencies todos
# validate fix parameters todos
"""

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text), member[0].text,
                     int(member[1][0].text),
                     int(member[1][1].text),
                     int(member[1][2].text),
                     int(member[1][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'FOLDER/ANNOTATIONFOLDER')  # todo: change the annotation folder
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('CSVMETADATAFILE.csv', index=None)  # todo: change the output file name
    print('successfully converted xml metadata to csv')


main()