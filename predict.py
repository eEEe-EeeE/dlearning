from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import pandas as pd
import numpy as np


def recall50(y, y_):
    diff = np.abs(y - y_)
    pre = np.squeeze(1 - np.dot(diff / (y + [5, 3, 3]), [[0.5], [0.25], [0.25]]))
    y_sum = np.sum(y, axis=1)
    y_sum[y_sum > 100] = 100
    sign = np.ones_like(y_sum)
    sign[pre <= 0.8] = 0
    return np.dot(y_sum, sign) / np.sum(y_sum)


def test():
	features = pd.read_csv("dataset/train_features.txt", header=None)
	labels = pd.read_csv("dataset/train_labels.txt", header=None)
	to_predict_features = pd.read_csv("dataset/predict_features.txt", header=None)
	X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2)

	y = np.around(np.array(y_test))

	def join_to_str(L):
		return ','.join(map(str,L)) + '\n'

	def linear_regress():
		model = LinearRegression()
		model.fit(X_train, y_train)
		y_ = np.around(model.predict(X_test))
		y_[y_ < 0] = 0
		res = recall50(y, y_)
		print('linear_regress recall50: ' + str(res))

		predicted_res = np.around(model.predict(to_predict_features))
		predicted_res[predicted_res < 0] = 0
		outfile = open("output_files/linear_regress_predict.txt", "w", encoding="utf-8")
		outfile.writelines(list(map(join_to_str, predicted_res.astype(np.int32).tolist())))
		outfile.close()
	linear_regress()

	def decision_tree():
		model = DecisionTreeRegressor()
		model.fit(X_train, y_train)
		y_ = np.around(model.predict(X_test))
		y_[y_ < 0] = 0
		res = recall50(y, y_)
		print('decision_tree recall50: ' + str(res))

		predicted_res = np.around(model.predict(to_predict_features))
		predicted_res[predicted_res < 0] = 0
		outfile = open("output_files/decision_tree_predict.txt", "w", encoding="utf-8")
		outfile.writelines(list(map(join_to_str, predicted_res.astype(np.int32).tolist())))
		outfile.close()
	decision_tree()

	def random_forest():
		model = RandomForestRegressor()
		model.fit(X_train, y_train)
		y_ = np.around(model.predict(X_test))
		y_[y_ < 0] = 0
		res = recall50(y, y_)
		print('random_forest recall50: ' + str(res))

		predicted_res = np.around(model.predict(to_predict_features))
		predicted_res[predicted_res < 0] = 0
		outfile = open("output_files/random_forest_predict.txt", "w", encoding="utf-8")
		outfile.writelines(list(map(join_to_str, predicted_res.astype(np.int32).tolist())))
		outfile.close()
	random_forest()

	def xgb_model():
		model1 = xgb.XGBRegressor() # 转
		model2 = xgb.XGBRegressor() # 评
		model3 = xgb.XGBRegressor() # 赞
		model1.fit(X_train, y_train[0])
		model2.fit(X_train, y_train[1])
		model3.fit(X_train, y_train[2])
		y_ = np.zeros_like(y_test)
		y_[:,0] = np.around(model1.predict(X_test))
		y_[:,1] = np.around(model2.predict(X_test))
		y_[:,2] = np.around(model3.predict(X_test))
		y_[y_ < 0] = 0
		res = recall50(y, y_)
		print('xgb_model recall50: ' + str(res))

		predicted_res = np.zeros((len(to_predict_features), 3))
		predicted_res[:,0] = np.around(model1.predict(to_predict_features))
		predicted_res[:,1] = np.around(model2.predict(to_predict_features))
		predicted_res[:,2] = np.around(model3.predict(to_predict_features))
		predicted_res[predicted_res < 0] = 0
		outfile = open("output_files/xgb_model_predict.txt", "w", encoding="utf-8")
		outfile.writelines(list(map(join_to_str, predicted_res.astype(np.int32).tolist())))
		outfile.close()
	xgb_model()


if __name__ == '__main__':
	test()

	

