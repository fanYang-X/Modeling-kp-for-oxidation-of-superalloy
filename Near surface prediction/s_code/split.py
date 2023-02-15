from sklearn.model_selection import train_test_split, KFold

class DataSplit:
    def __init__(self, data, label, random_state) -> None:
        self.data = data
        self.label = label
        self.num = 10
        self.random_state = random_state

    def cv(self, train_data, train_label):
        cv_data = {}
        kfold = KFold(n_splits=self.num, random_state=self.random_state, shuffle=True)
        for fold, (train_id, val_id) in enumerate(kfold.split(train_data, train_label)):
            cv_data[fold] = {'train': (train_data[train_id], train_label[train_id]),
                             'val': (train_data[val_id], train_label[val_id])}
        return cv_data


    def train_test(self):
        train_data, test_data, train_label, test_label = train_test_split(self.data, self.label, test_size=0.2, 
                                                                         random_state=self.random_state)
        return train_data, test_data, train_label, test_label

    def get_split(self):
        train_data, test_data, train_label, test_label = self.train_test()
        cv_data = self.cv(train_data, train_label)
        return cv_data, train_data, train_label, test_data, test_label