from fastai.vision import *

import argparse

if __name__ == '__main__':
    '''
    Severity Levels

    0 - 'No_DR',
    1 - 'Mild',
    2 - 'Moderate',
    3 - 'Severe',
    4 - 'Proliferate_DR'
    '''

    ap = argparse.ArgumentParser()
    ap.add_argument(
        '-e', '--epochs', type=int, default=10, 
        help='number of epoch to train'
    )
    ap.add_argument(
        '-dp', '--data_path', type=str, 
        help='path to the data'
    )
    ap.add_argument(
        '-mp', '--model_path', type=str,
        help='path for saving the model'
    )
    args = vars(ap.parse_args())

    classes = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

    path = Path(args['data_path'])
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

    # show input images
    # data.show_batch(rows=3, figsize=(10, 7))
    # plt.show()

    print(data.classes)
    print(data.c)
    print(f'Number of train images: {len(data.train_ds)}')
    print(f'Number of validation images: {len(data.valid_ds)}')

    # the model
    learn = cnn_learner(data, models.resnet34, metrics = [error_rate, accuracy])
    # path = users/.cache/torch

    learn.fit_one_cycle(args['epochs'])

    learn.recorder.plot_losses()
    plt.show()

    learn.save(args['model_path'])

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

    img = open_image('./input/train_images/ffec9a18a3ce.png')

    # give the correct path here
    learn = load_learner(path) 

    pred_class, pred_idx, outputs = learn.predict(img)
    print(pred_class)