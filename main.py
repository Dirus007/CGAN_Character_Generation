import train_model as tm
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
Gen = tm.give_models('Predict')

import tkinter as tk
from PIL import Image, ImageTk

class ImageViewer:
    def __init__(self, root, image_path, title):
        self.root = root
        self.root.title(title)

        #Open the image using PIL
        self.image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(self.image)

        #Create a Canvas widget to display the image
        self.canvas = tk.Canvas(root, width=self.image.width, height=self.image.height)
        self.canvas.pack()

        #Display the image on the canvas
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        #Bind mouse events for zooming and moving
        self.canvas.bind("<ButtonPress-1>", self.start_move)
        self.canvas.bind("<B1-Motion>", self.move)
        self.canvas.bind("<MouseWheel>", self.zoom)

        self.zoom_factor = 1.0
        self.x = 0
        self.y = 0

    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def move(self, event):
        dx = event.x - self.x
        dy = event.y - self.y
        self.canvas.move(self.image_item, dx, dy)
        self.x = event.x
        self.y = event.y

    def zoom(self, event):
        #Get the current zoom factor
        current_zoom = self.zoom_factor

        #Zoom in or out based on mouse wheel movement
        if event.delta > 0:
            self.zoom_factor *= 1.2
        else:
            self.zoom_factor *= 0.8

        #Calculate the scaled image size
        image_width = int(self.image.width * self.zoom_factor)
        image_height = int(self.image.height * self.zoom_factor)

        #Resize the image
        resized_image = self.image.resize((image_width, image_height), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(resized_image)

        #Update the canvas with the new image
        self.canvas.delete(self.image_item)
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        #Adjust the canvas to fit the new image size
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))



while True:
    print('''
        Press 'G' to view Generator Model.
        Press 'D' to view Discriminator Model.
        Press 'C' to view Conditional GAN Model.
        Press 'P' to generate realistic handwritten character.
        Press 'V' to view some training pictures.
        Press any other key to exit.
        \n
''')
    key_press = input('Key Pressed : ')
    key_press = key_press.lower()
    
    if(key_press=='g'):
        root = tk.Tk()
        ImageViewer(root, 'Generator_model.png' , 'Generator_Model')
        root.mainloop()
    
    elif(key_press=='d'):
        root = tk.Tk()
        ImageViewer(root, 'Discriminator_model.png' , 'Discriminator_Model')
        root.mainloop()
    
    elif(key_press=='c'):
        root = tk.Tk()
        ImageViewer(root, 'cGAN_model.png' , 'CGAN_Model')
        root.mainloop()
    
    elif(key_press=='p'):
        character = input('Enter a character : ')
        label = 0
        if(character[0]>='0' and character[0]<='9'):
            label = ord(character[0]) - ord('0')
        else:
            label = ord(character[0]) - ord('a') + 10
        
        random_val = np.random.randn(100)
        random_val = random_val.reshape(1, 100)
        
        with open(tm.GENERATOR_PATH, 'rb') as file1:
            Gen = load_model(file1)
            generated_image = Gen.predict([label,random_val])
            generated_image = (generated_image * 255).astype(np.uint8)  #Convert values to 0-255 range
            pil_image = Image.fromarray(generated_image)
            ImageViewer.show_image(pil_image)
    
    elif(key_press=='v'):
        tm.viewTrainImages(5,5)

    else:
        print('\nExit')
        break

