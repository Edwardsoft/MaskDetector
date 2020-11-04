
from xml.dom.minidom import parse
import os
def readxml():
	figure = []  #图像路径
	position = []  #人脸位置
	val = []  #标签
	for root, dirs, files in os.walk("C:\Users\639\maskDectorData"):
		for file in files:
			if os.path.splitext(file)[1] == ".xml":
				domTree = parse("C:\Users\639\maskDectorData\\" + file)

				# 文档根元素
				rootNode = domTree.documentElement
				# 所有顾客
				peoples = rootNode.getElementsByTagName("object")
				for people in peoples:
					figure.append("C:\Users\639\maskDectorData\\" + file)
					type = people.getElementsByTagName('name')[0].firstChild.data
					if type == "face":
						val.append(0)
					else:
						val.append(1)
					bd = people.getElementsByTagName('bndbox')[0]
					print(file)
					pos = [];
					xmin = bd.getElementsByTagName('xmin')[0].firstChild.data
					ymin = bd.getElementsByTagName('ymin')[0].firstChild.data
					xmax = bd.getElementsByTagName('xmax')[0].firstChild.data
					ymax = bd.getElementsByTagName('ymax')[0].firstChild.data
					pos.append(xmin)
					pos.append(xmax)
					pos.append(ymin)
					pos.append(ymax)
					position.append(pos)
	print(len(position))
	print(len(val))
	print(len(figure))
	return position, val, figure
if __name__ == '__main__':
	readxml()
