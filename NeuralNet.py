from sklearn.utils import shuffle
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NN(nn.Module):
    
    def __init__(self, input_len):

      super(NN, self).__init__()
      self.input_len = input_len

      self.fc1 = nn.Linear(input_len, 16)
      self.bn1 = nn.BatchNorm1d(16)
      self.drop1 = nn.Dropout(p=0.6)
      #self.fc2 = nn.Linear(128, 128)
      #self.bn2 = nn.BatchNorm1d(128)
      #self.fc3 = nn.Linear(128, 64)
      #self.bn3 = nn.BatchNorm1d(64)
      #self.fc4 = nn.Linear(64, 32)
      #self.bn4 = nn.BatchNorm1d(32)
      self.fc5 = nn.Linear(16, 8)
      self.bn5 = nn.BatchNorm1d(8)
      self.drop2 = nn.Dropout(p=0.6)
      self.fc6 = nn.Linear(8, 1)

      self.optimizer = optim.Adam(self.parameters(), lr=.03)
      self.metric = nn.SmoothL1Loss()

    def _forward(self, x):

      x = F.relu(self.drop1(self.bn1(self.fc1(x))))
      #x = F.relu(self.bn2(self.fc2(x)))
      #x = F.relu(self.bn3(self.fc3(x)))
      #x = F.relu(self.bn4(self.fc4(x)))
      x = F.relu(self.drop2(self.bn5(self.fc5(x))))
      x = F.relu(self.fc6(x))
      return x

    def _shuffle(self, X, Y):

        new_indices = np.random.choice(len(X), len(X))
        print("shuffle", new_indices)
        X = X[new_indices]
        Y = Y[new_indices]

    def train_model(self, X, Y, X_val, Y_val, batch_size=128, epochs=500, validation_step=20):
      
      iterations = int(len(X) // batch_size)
      for epoch in range(epochs+1):

        avg_loss = 0
        for it in range(iterations):

          batch_indices = np.random.choice(len(X), batch_size)
          batch_X = X[batch_indices]
          batch_Y = Y[batch_indices]
          pred = self._forward(batch_X)

          self.optimizer.zero_grad()
          loss = self.metric(pred, batch_Y)
          avg_loss += loss

          loss.backward()
          self.optimizer.step()
          
        print(avg_loss / iterations)
        avg_loss = 0
        self._shuffle(X, Y)

        if epoch % validation_step == 0:

          print(" \n Validation loss:")
          loss, pred = self.predict(X_val, Y_val)
          print(loss, '\n')

    def predict(self, X_test, Y_test):

      self.eval()
      pred = self._forward(X_test)
      loss = self.metric(pred, Y_test)
      self.train()
      return loss, pred.detach()


