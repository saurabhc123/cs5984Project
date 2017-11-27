import KaggleDataManager




kdm = KaggleDataManager.KaggleDataManager(" ")
len(kdm.test)
sample = KaggleDataManager.KaggleSample(None)
sample.set(kdm.test[0])
raw_data = sample.get_image_data()
#img = kdm.test[0].get_image_data()