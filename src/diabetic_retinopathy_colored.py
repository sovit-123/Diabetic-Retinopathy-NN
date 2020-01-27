from fastai.vision import *

if __name__ == '__main__':
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
                                    num_workers=4, bs=16).normalize(imagenet_stats)

    print(data.classes)

    data.show_batch(rows=3, figsize=(10, 7))
    plt.show()

    print(data.classes)
    print(data.c)
    print(f'Number of train images: {len(data.train_ds)}')
    print(f'Number of validation images: {len(data.valid_ds)}')

    # the model
    learn = cnn_learner(data, models.resnet34, metrics = [error_rate, accuracy])
    # path = users/.cache/torch

    learn.fit_one_cycle(20)

    learn.recorder.plot_losses()
    plt.show()

    learn.save('../../../models/colored_stage_1')

    learn.unfreeze()

    learn.lr_find()

    learn.recorder.plot()
    plt.show()

    learn.fit_one_cycle(3, max_lr=slice(1e-5, 1e-4))

    learn.save('../../../models/colored_stage_2')

    learn.load('../../../models/colored_stage_2')

    interp = ClassificationInterpretation.from_learner(learn)

    interp.plot_confusion_matrix()

    learn.export('colored_export.pkl')

    defaults.device = torch.device('cpu')

    img = open_image('../input/train_images/ffec9a18a3ce.png')

    # give the correct path here
    learn = load_learner(path) 

    pred_class, pred_idx, outputs = learn.predict(img)
    print(pred_class)