import xml.dom.minidom as xmldom

# 1:pic 2'caption', 3'paragraph', 4'heading'
class xml_reader():
    def __init__(self, path):
        self.cat2lable = {'pic':0, 'caption':1, 'paragraph':2, 'heading':3}
        self.cat2tag_name = {'pic':'GraphicRegion', 'caption':'TextRegion',
                             'paragraph':'TextRegion', 'heading':'TextRegion'}
        self.path = path
        self.xml_file = self.get_xml_file()
        self.pic_annotation = self.get_annotation('pic')
        self.caption_annotation = self.get_annotation('caption')
        self.paragraph_annotation = self.get_annotation('paragraph')
        self.heading_annotation = self.get_annotation('heading')
        self.annotations = self.pic_annotation + self.heading_annotation + \
                           self.caption_annotation + self.paragraph_annotation


    def get_xml_file(self):
        return (xmldom.parse(self.path)).documentElement

    def get_annotation(self, cat):
        annotations = []
        if cat == 'pic':
            node_list = self.xml_file.getElementsByTagName('GraphicRegion')
        else:
            node_list = [x for x in self.xml_file.getElementsByTagName(self.cat2tag_name[cat])
                             if x.getAttribute('type')==cat]
        if len(node_list) == 0:
            return []
        for node in node_list:
            points_location = node.getElementsByTagName('Coords')[0].getAttribute('points')
            points_location = [[int(y) for y in x.split(',')] for x in points_location.split(' ')]
            # print(points_location) 调试点
            left_top_x = points_location[-1][0]
            left_top_y = points_location[-1][1]
            height = points_location[2][1] - points_location[0][1]
            width = points_location[2][0] - points_location[0][0]
            area = height * width
            annotations.append({"bbox": [left_top_x, left_top_y, width, height],
                                "area": area,
                                "iscrowd": 0,
                                "category_id": self.cat2lable[cat]})

        return annotations


    # def get_pic_annotation(self):
    #     if len(self.xml_file.getElementsByTagName('GraphicRegion')) == 0:
    #         return None
    #     annotations = []
    #     pic_node_list = self.xml_file.getElementsByTagName('GraphicRegion')
    #     for pic_node in pic_node_list:
    #         points_location = pic_node.getElementsByTagName('Coords')[0].getAttribute('points')
    #         points_location = [[int(y) for y in x.split(',')] for x in points_location.split(' ')]
    #         # print(points_location) 调试点
    #         left_top_x = points_location[-1][0]
    #         left_top_y = points_location[-1][1]
    #         height = points_location[2][1] - points_location[0][1]
    #         width = points_location[2][0] - points_location[0][0]
    #         area = height * width
    #         annotations.append({"bbox": [left_top_x, left_top_y, width, height],
    #                              "area":area,
    #                              "iscrowd": 0,
    #                              "category_id": 1})
    #
    #     return annotations


x = xml_reader('/home/wenjun_sun/code/dlu_baseline/data/18690715_1-0001.xml')

print(x.annotations)
# a = x.xml_file.getElementsByTagName('TextRegion')
# b = []
# for temp in a:
#     b.append(temp.getAttribute('type'))
# print(set(b))

