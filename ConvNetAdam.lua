require 'hdf5'
require 'image'
require 'math'
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

-- separate sets
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


-- hyper-parameters
num_hidden = 200
num_filters = 32

batch_size = 50
l2_reg = 0.001
max_epoch = 5

classes = {'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'}
geometry = {48, 48}


-- setting up neural net
net = nn.Sequential()
-- conv - relu - conv - relu - 2x2 max pool
net:add(nn.SpatialConvolution(1, num_filters, 3, 3, 1, 1, 1, 1)) -- 1 input image channel, 32 output channels, 3x3 convolution kernel
net:add(nn.ReLU())                       -- non-linearity
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.

-- conv - relu - conv - relu - 2x2 max pool
net:add(nn.SpatialConvolution(num_filters, num_filters, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))

-- conv - relu - conv - relu - 2x2 max pool
net:add(nn.SpatialConvolution(num_filters, num_filters, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))


-- affine - relu
net:add(nn.View(num_filters*6*6)) -- reshapes the 3D tensor

--net:add(nn.Linear(num_filters*6*6, num_hidden))             -- fully connected layer (matrix multiplication between input and weights)
--net:add(nn.ReLU())              -- non-linearity

-- affine - softmax
--net:add(nn.Linear(num_hidden, 7))
--net:add(nn.LogSoftMax())          -- converts the output to a log-probability. Useful for classification problems
net:add(nn.Linear(num_filters*6*6, 7))

criterion = nn.CrossEntropyCriterion()

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
confusion = optim.ConfusionMatrix(classes)

local parameters, gradParameters = net:getParameters()

config = {
   learningRate = 1e-3,
   beta1 = 0.9,
   beta2 = 0.999,
   epsilon = 1e-8,
   state = state
}

function train(dataset)
  -- epoch tracker
  epoch = epoch or 1

  -- do one epoch
  print("\nEpoch # " .. epoch .. ' [batchSize = ' .. batch_size .. ']')

  -- shuffle indices
  local shuffled_idx = torch.randperm((#dataset)[1], 'torch.LongTensor')

  --for t = 1, dataset:size(), batch_size do
  for t = 1, num_train, batch_size do

    -- create mini batch
    local inputs = torch.Tensor(batch_size, 1, geometry[1], geometry[2])
    local targets = torch.Tensor(batch_size)
    local k = 1
    for i = t, math.min(t + batch_size - 1, num_train) do
      -- load new sample
      local sample = dataset[shuffled_idx[i]]
      local input = sample[1]:clone()
      local target = y_train[shuffled_idx[i]]
      inputs[k] = input
      targets[k] = target
      k = k+1
    end


    -- create closure to evalute f(X) and df/dX
    local feval = function(x)
      -- just in case:
      collectgarbage()

      -- get new parameters
      if x ~= parameters then
        parameters:copy(x)
      end

      -- reset gradients
      gradParameters:zero()

      -- evaluate function for complete mini batch
      local outputs = net:forward(inputs)
      local f = criterion:forward(outputs, targets)

      -- estimate df/dW
      local df_do = criterion:backward(outputs, targets)
      net:backward(inputs, df_do)

      -- L2 penalty
      if l2_reg ~= 0 then
        -- local:
        local norm = torch.norm

        -- loss:
        f = f + l2_reg * norm(parameters, 2)^2/2

        -- gradients:
        gradParameters:add( parameters:clone():mul(l2_reg) )
      end

      -- update confusion
      for i = 1, batch_size do
        confusion:add(outputs[i], targets[i])
      end

      -- return f and df/dX
      return f, gradParameters
    end


    -- optimize on current mini batch

    state = state or config

    optim.adam(feval, parameters, config, state)

    xlua.progress(t, num_train)

  end
  --print(confusion)
  confusion:updateValids()
  print('\n% mean class accuracy (train set) ' .. confusion.totalValid*100)
  confusion:zero()
  epoch = epoch + 1
end

-- let's train 
for e = 1, max_epoch do
  train(X_train)
end

-- test accuracy
print('\n Test accuracy: \n')
confusion:zero()

for i=1,num_test do
    local groundtruth = test_set.label[i]
    local prediction = net:forward(test_set.data[i])
    confusion:add(prediction, groundtruth)
end

print(confusion)