        # Get the current zoom factor
        current_zoom = self.zoom_factor

        # Zoom in or out based on mouse wheel movement
        if event.delta > 0:
            self.zoom_factor *= 1.2
        else:
            self.zoom_factor *= 0.8

        # Calculate the scaled image size
        image_width = int(self.image.width * self.zoom_factor)
        image_height = int(self.image.height * self.zoom_factor)

        # Resize the image
        resized_image = self.image.resize((image_width, image_height), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(resized_image)

        # Update the canvas with the new image
        self.canvas.delete(self.image_item)
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Adjust the canvas to fit the new image size
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))