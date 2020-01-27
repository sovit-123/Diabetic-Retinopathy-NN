from fastai.vision import *

'''
Severity Levels

0 - 'No_DR',
1 - 'Mild',
2 - 'Moderate',
3 - 'Severe',
4 - 'Proliferate_DR'
'''

classes = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

path = Path('../input/colored_images')
print(path.ls())

''' 
run the following three lines only once to 
delete the images that cannot be opened 
'''
# for c in classes:
#     print(c)
#     verify_images(path/c, delete=True, max_size=500)

''' 
run the above three lines only once to 
delete the images that cannot be opened 
'''

np.random.seed(42)
data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2, 
                                  ds_tfms=get_transforms(), size=224, 
                                  num_workers=4, bs=8).normalize(imagenet_stats)

print(data.classes)

# data.show_batch(rows=3, figsize=(10, 7))

print(data.classes)
print(data.c)
print(f'Number of train images: {len(data.train_ds)}')
print(f'Number of validation images: {len(data.valid_ds)}')

# the model
learn = cnn_learner(data, models.resnet34, metrics = error_rate)
# path = users/.cache/torch

learn.fit_one_cycle(2)

learn.recorder.plot_losses()



