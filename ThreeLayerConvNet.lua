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

train_set = {}
num_train = 5000
train_set.data = X_train[{ {1, num_train}, {}, {}, {} }]
train_set.label = y_train[{ {1, num_train}}]
train_set.label = train_set.label:byte()


test_set = {}
num_test = 500
test_set.data = X_test[{ {1, num_test}, {}, {}, {} }]
test_set.label = y_test[{ {1, num_test} }]
test_set.label = test_set.label:byte()

print(train_set)


num_hidden = 500
num_filters = 32

print(train_set.data:mean())

-- net
net = nn.Sequential()
-- conv - relu - 2x2 max pool
net:add(nn.SpatialConvolution(1, num_filters, 5, 5, 1, 1, 2, 2)) -- 1 input image channel, 32 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.

-- affine - relu
net:add(nn.View(num_filters*24*24))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(num_filters*24*24, num_hidden))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())							-- non-linearity 

-- affine - softmax
net:add(nn.Linear(num_hidden, 7))
net:add(nn.LogSoftMax())					-- converts the output to a log-probability. Useful for classification problems

criterion = nn.ClassNLLCriterion()

print("Done setting up conv net")


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
trainer.learningRate = 0.001
trainer.maxIteration = 5	 -- just do 5 epochs of training.

start_time = os.time()

trainer:train(train_set)

print("Done training")

end_time = os.time()
elapsed_time = os.difftime(end_time, start_time)
print(elapsed_time)

correct = 0
for i=1,num_test do
    local groundtruth = test_set.label[i]
    local prediction = net:forward(test_set.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/num_test .. ' % ')