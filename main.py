from sklearn.cluster import KMeans as m
import os
import numpy as np
from scipy import ndimage
import sklearn.preprocessing as pr

class Feature_detection:
    def __init__(self):
        pass
    def end(self):
        self.high_width()
        self.quad()
        self.find_sum()
        self.find_rathers()
        self.find_rathers()
        self.f.append(self.h_w)
        return self.f
    def set_obj(self,obj):
        self.obj = obj
    def high_width(self):
        self.h_w = self.obj.x/self.obj.y
    def half_up_down(self,img):
        up = img[:len(img)//2,:]
        down = img[len(img)//2:,:]
        return up,down
    def half_right_left(self,img):
        right = img[:,len(img[0])//2:]
        left = img[:,:len(img[0])//2]
        return left,right
    def quad(self):
        ###################
        #     #     #     #
        #  1  #  2  #  3  #
        #     #     #     #
        ###################
        #     #     #     #
        #  4  #  5  #  6  #
        #     #     #     #
        ###################
        #     #     #     #
        #  7  #  8  #  9  #
        #     #     #     #
        ###################
        self.f = []
        self.f.append(self.obj.img[:len(self.obj.img)//3,:len(self.obj.img[0])//3])
        self.f.append(self.obj.img[:len(self.obj.img)//3,len(self.obj.img[0])//3:(len(self.obj.img[0])//3)*2])
        self.f.append(self.obj.img[:len(self.obj.img)//3,(len(self.obj.img[0])//3)*2:len(self.obj.img)])

        self.f.append(self.obj.img[len(self.obj.img)//3:(len(self.obj.img)//3)*2,:len(self.obj.img[0])//3])
        self.f.append(self.obj.img[len(self.obj.img)//3:(len(self.obj.img)//3)*2,len(self.obj.img[0])//3:(len(self.obj.img[0])//3)*2])
        self.f.append(self.obj.img[len(self.obj.img)//3:(len(self.obj.img)//3)*2,(len(self.obj.img[0])//3)*2:len(self.obj.img[0])])

        self.f.append(self.obj.img[(len(self.obj.img)//3)*2:len(self.obj.img),:len(self.obj.img[0])//3])
        self.f.append(self.obj.img[(len(self.obj.img)//3)*2:len(self.obj.img),len(self.obj.img[0])//3:(len(self.obj.img[0])//3)*2])
        self.f.append(self.obj.img[(len(self.obj.img)//3)*2:len(self.obj.img),(len(self.obj.img[0])//3)*2:len(self.obj.img[0])])
    def find_sum(self):
        self.fu = []
        for i in self.f:
            self.fu.append(sum(i.flat))
    def find_rathers(self):
        self.f = []
        for i in range(len(self.fu)):
            for j in range(i + 1, len(self.fu)):
                self.f.append(self.fu[i]/self.fu[j])
class make_ready:
    def __init__(self):
        pass
    def set_image(self,img):
        self.img = img
    def get_point(self):
        return self.point
    def get_image(self):
        return self.img
    def binarize(self, t):
        self.img = pr.binarize(self.img,threshold=t, copy=True)
    def rgb2gray(self):
        self.img = np.dot(self.img[..., :3], [0.299, 0.587, 0.114])
    def find_point(self):
        point = []
        w, h = self.img.shape[:2]
        for x in range(w):
            for y in range(h):
                d = self.img[x, y]
                if d == 0:
                    point.append([y, x])
        self.point = np.array(point)
    def clean(self):
        self.img = ndimage.gaussian_filter(self.img, sigma=1)
    def crop(self):
        x_start = min(self.point[:,0])
        x_end = max(self.point[:, 0])
        y_start = min(self.point[:,1])
        y_end = max(self.point[:, 1])
        self.x = x_end-x_start
        self.y = y_end- y_start
        self.img = self.img[y_start:y_end,x_start:x_end]
    def end(self):
        self.rgb2gray()
        self.clean()
        self.binarize(1)
        self.find_point()
        self.crop()
class reader:
    def __init__(self):
        self.address = self.place()
        self.first = self.address
    def place(self):
        'read where are you now'
        self.address = os.getcwd()
        return self.address
    def go(self,address):
        'go to the address that you choose'
        os.chdir(address)
        self.place()
    def read(self):
        'read files in your place'
        files = os.listdir(self.place())
        return files
    def open(self,foldername):
        self.address = os.path.join(self.address,foldername)
        self.go(self.address)
    def back(self):
        self.address = self.address.split('\\')
        del self.address[len(self.address)-1]
        self.go('\\'.join(self.address))
        self.place()
    def read_file_name(self):
        names = [f for f in os.listdir(self.address) if os.path.isfile(os.path.join(self.address, f))]
        names_no_hidden = [f for f in names if f[0] != '.']
        names_with_dot = [f for f in names_no_hidden if '.' in f]
        self.files = names_with_dot
    def read_folder_name(self):
        self.read_file_name()
        self.folder = []
        all = self.read()
        for i in all:
            if i not in self.files:
                self.folder.append(i)
    def end(self):
        self.read_folder_name()
        self.folder.remove('.idea')
    def back_all(self):
        self.go(self.first)
class avg:
    def __init__(self):
        self.ender = []
    def set_feature(self,feature):
        self.feature = feature
    def avg_kmeans(self):
        for i in self.feature:
            kmeans = m(1).fit(i)
            self.ender.append(kmeans.cluster_centers_)
    def end(self):
        return(np.array(self.ender))
class final:
    def __init__(self):
        self.all_feature = []
        self.short = []
    def set_obj_make_ready(self,obj):
        self.ready = obj
    def set_obj_reader(self,obj):
        self.reader = obj
    def set_obj_detection(self,obj):
        self.detection = obj
    def do(self):
        for j in self.reader.folder:
            self.reader.back_all()
            self.reader.open(j)
            inside = self.reader.read()
            for i in inside:
                self.short.append(self.maker(i))
            self.all_feature.append(self.short)
            self.short =[]
    def maker(self,img_name):
        img = ndimage.imread(img_name)
        self.ready.set_image(img)
        self.ready.end()
        self.detection.set_obj(self.ready)
        return b.end()
class weight:
    def __init__(self):
        self.weights = []
    def set_feature(self,feature):
        self.feature = feature
    def main(self):
        space = []
        for i in range(len(self.feature[0])):
            all = self.feature[:,i]
            avg = sum(all)/len(self.feature)
            for j in all:
                space.append(abs(avg-j))
            self.weights.append(sum(space)/len(space))
    def give_weights(self):
        return(np.array(self.weights))
class finish:
    def __init__(self):
        pass
    def set(self,all):
        self.all = all
    def set_feature(self,feature):
        self.feature = feature
    def find_diffrent(self):
        space = []
        self.diffrent = []
        for i, j in enumerate(self.feature):
            for k, h in enumerate(self.all[:, :, i]):
                space.append(abs(j - h))
            self.diffrent.append((space))
            space = []
    def find_smallest(self):
        self.smallest = []
        for i in self.diffrent:
            x = min(i)
            for j in range(len(i)):
                if i[j] == x:
                    self.smallest.append(j)
                    break
    def do_weigh(self):
        self.end = []
        a = max(self.smallest)
        for i in range(a+1):
            self.end.append([])
        for i, j in enumerate(self.smallest):
            self.end[j].append(self.w.flat[i-1])
    def set_w(self,w):
        self.w = w
###########################################
c = reader()
a = make_ready()
b = Feature_detection()
d = avg()
e = weight()
f = final()
g = finish()
f.set_obj_detection(b)
f.set_obj_make_ready(a)
f.set_obj_reader(c)
c.end()
f.do()
d.set_feature(f.all_feature)
d.avg_kmeans()
e.set_feature(d.end())
e.main()
w = e.give_weights()
g.set_w(w)
g.set(d.end())
c.back_all()
while(1):
    gg = input('enter: ')
    x = f.maker(gg)
    g.set_feature(x)
    g.find_diffrent()
    g.find_smallest()
    g.do_weigh()
    t = []
    for i in g.end:
        t.append(sum(i))
    k = max(t)
    jj = 0
    kk = sum(t)
    kk = 100/kk
    for i in range(len(t)):
        if t[i] != 0:
            print(c.folder[i],' = ',str(t[i]*kk)+'%')
            if t[i] == k:
              jj = i

    print('this image is : ',c.folder[jj],'\n\n')