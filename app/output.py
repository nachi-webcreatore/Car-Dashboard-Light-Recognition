import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
import itertools
from PIL import Image, ImageOps
import numpy as np
import re
import os


def model_output():
    path = os.getcwd()
    print(path)
    if os.path.isfile('airbags.jpg') is False:
        print('not a filesss')
    image = Image.open('airbags.jpg')

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = tensorflow.keras.models.load_model('model.savedmodel')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array
    model.summary()
    # run the inference: model.predict(data)
    prediction = model.predict(data)
    label_data = [
        "antilockbrakesystemlight",
        "batterywarninglight",
        "brakesystemwarninglights",
        "bulbmonitoringwarning",
        "enginelight",
        "engineoilleveltoolow",
        "foglights",
        "fueltankempty",
        "highbeamflasher",
        "parkingbrake",
        "parkinglights",
        "airbags",
        "asr_manual",
        "break_pad",
        "epc",
        "central_warning",
        "engine_oil_low",
        "elect_stablization",
        "permanent_d_lignt",
        "rear_fog",
        "engine_control",
        "windscreen_washer",
        "cooling_system",
        "transmision_disturbed",
        "windscreen_wiper_fun",
        "tyre_pressure"
    ]
    predict_by_label = str(np.round(prediction)).split(' ')
    for i in range(len(predict_by_label)):
        if predict_by_label[i] == '1.':
            print(label_data[i] + ': ' + predict_by_label[i])

    saved_model_dir = '/content/TFLite'
    tf.saved_model.save(model, saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

    # Creates model info.
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "Car Warning Light Dashboard Image Classifier"
    model_meta.description = ("Identify the most prominent object in the "
                              "image from a set of 26 categories such as "
                              "enginelight, antilockbrakesystemlight, etc")
    model_meta.version = "v1.4"
    model_meta.author = "Webcreatore Digital Solutions LLP"
    model_meta.license = " Not licensed. Contact Developers! "

    # Creates input info.
    input_meta = _metadata_fb.TensorMetadataT()

    # Creates output info.
    output_meta = _metadata_fb.TensorMetadataT()

    input_meta.name = "image"
    input_meta.description = (
        "Input image to be classified. The expected image is {0} x {1}, with "
        "three channels (red, blue, and green) per pixel. Each value in the "
        "tensor is a single byte between 0 and 255. You must pre-process the image.".format(224, 224))
    input_meta.content = _metadata_fb.ContentT()
    input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
    input_meta.content.contentProperties.colorSpace = (
        _metadata_fb.ColorSpaceType.RGB)
    input_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.ImageProperties)
    input_normalization = _metadata_fb.ProcessUnitT()
    input_normalization.optionsType = (
        _metadata_fb.ProcessUnitOptions.NormalizationOptions)
    input_normalization.options = _metadata_fb.NormalizationOptionsT()
    input_normalization.options.mean = [127.5]
    input_normalization.options.std = [127.5]
    input_meta.processUnits = [input_normalization]
    input_stats = _metadata_fb.StatsT()
    input_stats.max = [255]
    input_stats.min = [0]
    input_meta.stats = input_stats

    # Creates output info.
    output_meta.name = "probability"
    output_meta.description = "Probabilities of the 26 labels respectively."
    output_meta.content = _metadata_fb.ContentT()
    output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
    output_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_stats = _metadata_fb.StatsT()
    output_stats.max = [1.0]
    output_stats.min = [0.0]
    output_meta.stats = output_stats
    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = 'E:/imagedata/labels.txt'
    print(label_file.name)
    label_file.description = "Labels for objects that the model can recognize."
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS

    # Creates subgraph info.
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [output_meta]
    model_meta.subgraphMetadata = [subgraph]

    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()

    model_file = 'model.tflite'
    populator = _metadata.MetadataPopulator.with_model_file(model_file)
    populator.load_metadata_buffer(metadata_buf)
    populator.load_associated_files(["labels.txt"])
    populator.populate()
