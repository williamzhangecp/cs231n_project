require 'hdf5'
require 'image'
require 'nn'
require 'optim'
require 'os'

-- read data
myFile = hdf5.open('full_raw.h5', 'r')
X_train = myFile:read('train/X_train'):all()
y_train = myFile:read('train/y_train'):all()
X_test = myFile:read('test/X_test'):all()
y_test = myFile:read('test/y_test'):all()
X_val = myFile:read('val/X_val'):all()
y_val = myFile:read('val/y_val'):all()
myFile:close()

print("Done reading data")

-- net params
num_train = 1000
num_val = 500
learning_rate = 0.01
num_epochs = 10
num_hidden = 100
num_filters = 32
filter_size = 5
num_filters2 = 32
filter_size2 = 5
num_classes = 7
reg = 0.005
dropout_prob = 0.4
-- Size of image
H = 48
W = 48

train_set = {}
train_set.data = X_train[{ {1, num_train}, {}, {}, {} }]
train_set.label = y_train[{ {1, num_train}}]
train_set.label = train_set.label:byte()

val_set = {}
val_set.data = X_val[{ {1, num_val}, {}, {}, {} }]
val_set.label = y_val[{ {1, num_val} }]
val_set.label = val_set.label:byte()

-- net
net = nn.Sequential()

-- conv - relu - 2x2 max pool
net:add(nn.SpatialConvolution(1, num_filters, filter_size, filter_size, 1, 1, 2, 2)) -- 1 input image channel, 32 output channels, 5x5 convolution kernel
net:add(nn.Dropout(dropout_prob))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2)) -- max-pooling operation that looks at 2x2 windows and finds the max.

-- conv - relu - 2x2 max pool (Note input image now has 32 channels)
net:add(nn.SpatialConvolution(num_filters, num_filters2, filter_size2, filter_size2, 1, 1, 2, 2)) -- 32 input image channel, 32 output channels, 5x5 convolution kernel
net:add(nn.Dropout(dropout_prob))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))

-- affine - relu
net:add(nn.View(num_filters2*H/4*H/4))  -- reshapes from a 3D tensor to 1D tensor
net:add(nn.Linear(num_filters2*H/4*H/4, num_hidden))  -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())

-- affine - softmax
net:add(nn.Linear(num_hidden, num_classes))
net:add(nn.LogSoftMax())  -- converts the output to a log-probability

criterion = nn.CrossEntropyCriterion()

print("Done setting up conv net")

-- Number of parameters
parameters, gradParams = net:getParameters()
print("Number of paramters:")
print(#parameters)

-- set up a meta-table --> what is this?
setmetatable(train_set,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]}
                end}
);

function train_set:size()
    return self.data:size(1)
end


-- set up solver

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = learning_rate
trainer.weightDecay = reg
trainer.maxIteration = num_epochs

start_time = os.time()

print("Number of epochs:")
print(trainer.maxIteration)

trainer:train(train_set)

print("Done training")

end_time = os.time()
elapsed_time = os.difftime(end_time, start_time)
print("Elapsed time:")
print(elapsed_time)


-- Confusion Matrix
classes = { 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral' }
confusion = optim.ConfusionMatrix(classes)

for i=1,num_val do
    local groundtruth = val_set.label[i]
    local prediction = net:forward(val_set.data[i])
    confusion:add(prediction, groundtruth)
end

print("Validation")
print(confusion)

confusion:zero()

for i=1,num_train do
    local groundtruth = train_set.label[i]
    local prediction = net:forward(train_set.data[i])
    confusion:add(prediction, groundtruth)
end

print("Train")
print(confusion)
