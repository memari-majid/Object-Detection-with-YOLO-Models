TRAINING_PARAMS = {
    'epochs': 200,
    'imgsz': 1024,
    'batch': 4,
    'device': '0,1',
    'optimizer': 'AdamW',
    'lr0': 0.0005,
    'mosaic': 1.0,
    'scale': 0.5,
    'flipud': 0.5,
    'fliplr': 0.5,
    'augment': True,
    'cos_lr': True,
    'patience': 30,
    'name': 'yolo_small_object_optimized',
    'project': 'WTB_Results_Clean',
    'exist_ok': True
}

# Validation parameters
VAL_PARAMS = {
    'imgsz': 1024,
    'batch': 8,
    'device': '0,1'
} 