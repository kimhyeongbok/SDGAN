
import sys
sys.path.append('./source')     
from train import ciagan_exp

r = ciagan_exp.run(config_updates={
    'TRAIN_PARAMS':
        {
            'ARCH_NUM': 'spade_feature_flex',
            'ARCH_SIAM': 'resnet_siam',
            'EPOCH_START': 200, 'EPOCHS_NUM': 501,
            'LEARNING_RATE': 0.0001,
            'FILTER_NUM': 16,

            'ITER_CRITIC': 1,
            'ITER_GENERATOR': 3,
            'ITER_SIAMESE': 1,

            'GAN_TYPE': 'lsgan',  # lsgan wgangp
        },
    'DATA_PARAMS':
        {
            'LABEL_NUM': 1200,
            'BATCH_SIZE': 32,
            'WORKERS_NUM': 4,
            'IMG_SIZE': 128,
        },
    'OUTPUT_PARAMS': {
            'SAVE_EPOCH': 1,
            'SAVE_CHECKPOINT': 100,
            'LOG_ITER': 2,
            'COMMENT': "Something here",
            'EXP_TRY': 'check',
        }
    })


# from train_with_feature_on_Adience import ciagan_exp
# r = ciagan_exp.run(config_updates={
#     'TRAIN_PARAMS':
#         {
#             'ARCH_NUM': 'spade_feature_flex',
#             'ARCH_SIAM': 'resnet_siam',
#             'EPOCH_START': 1000,
#             'EPOCHS_NUM': 1801,
#             'LEARNING_RATE': 0.0001,
#             'FILTER_NUM': 16,
#
#             'ITER_CRITIC': 1,
#             'ITER_GENERATOR': 3,
#             'ITER_SIAMESE': 1,
#              'LAMBDA1':1,
#              'LAMBDA2':1,
#              'LAMBDA3':1,
#             'GAN_TYPE': 'lsgan',  # lsgan wgangp
#         },
#     'DATA_PARAMS':
#         {
#             'LABEL_NUM': 329,
#             'BATCH_SIZE': 32,
#             'WORKERS_NUM': 4,
#             'IMG_SIZE': 128,
#         },
#     'OUTPUT_PARAMS': {
#             'SAVE_EPOCH': 10,
#             'SAVE_CHECKPOINT': 50,
#             'LOG_ITER': 2,
#             'COMMENT': "Something here",
#             'EXP_TRY': 'check',
#         }
#     })
