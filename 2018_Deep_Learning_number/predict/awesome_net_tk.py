from tkinter import*
import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from deep_convnet import*
import mnist


def init_network():
    with open("deep_convnet_params.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def set_img(x,y):
    global img
    if x>=560:
        xi=27
    elif x>=0:
        xi=x//20
    if y>=560:
        yi=27
    elif y>=0:
        yi=y//20
    img[0][0][yi][xi]=1

def draw(event):
    global x0,y0
    canvas.create_line(x0,y0,event.x,event.y)
    x0,y0=event.x,event.y
    set_img(x0,y0)
    
def down(event):
    global x0,y0
    x0,y0=event.x,event.y
    
def up(event):
    global x0,y0
    if x0==event.x and y0==event.y:
        canvas.create_line(x0,y0,x0+1,y0+1)
        set_img(x0,y0)

def apply():
    global img, canvas, network, entry
    y = network.predict(img)
    p = np.argmax(y)
    entry.delete(0, END)
    entry.insert(0, str(p))
    canvas.delete("all")
    img_clear(img)
    
def img_clear(img):
    img[0][0]=0
    


img = np.array([[[[0]*28]*28]])
img_clear(img)

network = DeepConvNet()
network.load_params(file_name="deep_convnet_params.pkl")


root=Tk()
root.title("awesome")
root.geometry('560x700')
canvas=Canvas(root,bg="white",width=560,height=560)
entry = Entry(width=5,font = "Helvetica 44 bold",justify=CENTER)

apply_button = Button(text='Apply', command = apply)
delete_button = Button(text='Delete', command=lambda:canvas.delete("all") )
quit_button = Button(text='quit', command = root.destroy)


quit_button.pack(side="bottom")
delete_button.pack(side="bottom")
apply_button.pack(side="bottom")
canvas.bind("<B1-Motion>",draw)
canvas.bind("<Button-1>",down)
canvas.bind("<ButtonRelease-1>",up)
entry.pack()
canvas.pack()
root.mainloop()