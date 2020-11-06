
from xml.dom.minidom import parse
import os
def readxml():
	figure = []  #图像路径
	position = []  #人脸位置
	val = []  #标签
	img_realname = [] # 图像真名
	for root, dirs, files in os.walk("D:\\maskDectorData\\train"):
		for file in files:
			if os.path.splitext(file)[1] == ".xml":
				domTree = parse("D:\\maskDectorData\\train\\" + file)

				# 文档根元素
				rootNode = domTree.documentElement
				# 所有顾客
				peoples = rootNode.getElementsByTagName("object")
				i = 1
				for people in peoples:
					figure.append("D:\\maskDectorData\\train\\" + os.path.splitext(file)[0] + ".jpg")
					img_realname.append(os.path.splitext(file)[0] + "_" + str(i) + ".jpg")
					i = i + 1
					type = people.getElementsByTagName('name')[0].firstChild.data
					if type == "face":
						val.append(0)
					else:
						val.append(1)
					bd = people.getElementsByTagName('bndbox')[0]
					print(file)
					pos = []
					xmin = int(bd.getElementsByTagName('xmin')[0].firstChild.data)
					ymin = int(bd.getElementsByTagName('ymin')[0].firstChild.data)
					xmax = int(bd.getElementsByTagName('xmax')[0].firstChild.data)
					ymax = int(bd.getElementsByTagName('ymax')[0].firstChild.data)
					pos.append(xmin)
					pos.append(xmax)
					pos.append(ymin)
					pos.append(ymax)
					position.append(pos)
	print(len(position))
	print(len(val))
	print(len(figure))
	return position, val, figure, img_realname
if __name__ == '__main__':
	readxml()
