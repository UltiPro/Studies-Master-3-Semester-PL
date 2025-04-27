from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import os

train_cats_dir = "cats"

datagen = ImageDataGenerator( rotation_range=40, width_shift_range=0.2,
height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
fill_mode='nearest')

# Operacja importowania modułu zawierającego narzędzia przetwarzające obrazy.
from tensorflow.keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

# Wybieramy obraz do zmodyfikowania.
img_path = fnames[0]

# Wczytujemy obraz i zmieniamy jego rozdzielczość.
img = image.load_img(img_path, target_size=(150, 150))

# Zamieniamy obraz w tablicę Numpy o kształcie (150, 150, 3).
x = image.img_to_array(img)

# Zmieniamy kształt na (1, 150, 150, 3).
x = x.reshape((1,) + x.shape)

# Polecenie .flow() generuje wsady obrazów zmodyfikowanych w sposób losowy.
# Pętla jest wykonywana w nieskończoność, a więc należy ją w pewnym momencie przerwać!
i = 1

for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 5 == 0:
        break
    
plt.show()