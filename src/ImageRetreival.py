from graphSearch import GraphSearch
from DeepFeatures import DeepFeatures
import os
import glob
class ImageRetrieval():
	def __init__(self,image_folder_name):
		self.gs = GraphSearch()
		self.folder_name=image_folder_name
		self.files = glob.glob(os.path.join(self.folder_name,'**/*.jpg'))
		print("Number of files",len(self.files))
		self.feature_gen=DeepFeatures()
		
	def create_index(self,create_new=False):
		self.feature_store=[]
		if create_new:
			for idx,file_name in enumerate(files):
				image=self.feature_gen.read_image(file_name)
				feature = self.feature_gen.get_feature(image)
				self.feature_store.append(feature.ravel())
			self.gs.create_index(np.array(feature_store,np.float32),
								np.arange(len(files[:10])))
			self.gs.save_index()
		else:
			self.gs.load_index()
	def get_match(self,image_file):
		image = self.feature_gen.read_image(image_file)
		feature = self.feature_gen.get_feature(image)
		return self.gs.knn(feature.ravel())[0][0]
