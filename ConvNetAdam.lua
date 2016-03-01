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

train_set = {}
num_train = 100
train_set.data = X_train[{ {1, num_train}, {}, {}, {} }]
train_set.label = y_train[{ {1, num_train}}]
train_set.label = train_set.label:byte()


test_set = {}
num_test = 500
test_set.data = X_test[{ {1, num_test}, {}, {}, {} }]
test_set.label = y_test[{ {1, num_test} }]
test_set.label = test_set.label:byte()


-- hyper-parameters
num_hidden = 500
num_filters = 32

batch_size = 50
l2_reg = 0

classes = {'1', '2', '3', '4', '5', '6', '7'}
geometry = {48, 48}


-- setting up neural net
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
--net:add(nn.LogSoftMax())					-- converts the output to a log-probability. Useful for classification problems

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

-- state = {}

config = {
   learningRate = 1e-3,
   beta1 = 0.9,
   beta2 = 0.999,
   epsilon = 1e-8,
   state = state
   -- opfunc : function that takes a single input X, and returns f(X) and df/dX
   -- x: initial point
}

--[[
-- callback function for optimization routine
local function loss_and_grad(w)
  assert(w == weights)
  
  -- forward pass
  local scores = net:forward(X_batch)
  local loss = criterion:forward(scores, y_batch)

  -- backward pass: compute gradients
  grad_weights:zero()
  local dscores = criterion:backward(scores, y_batch)
  net:backward(X_batch, dscores) -- local dx = 

  return loss, grad_weights
end
]]--



function train(dataset)
  -- epoch tracker
  epoch = epoch or 1

  -- do one epoch
  print('<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batch_size .. ']')

  --for t = 1, dataset:size(), batch_size do
  for t = 1, num_train, batch_size do
    
    -- create mini batch
    local inputs = torch.Tensor(batch_size, 1, geometry[1], geometry[2])
    local targets = torch.Tensor(batch_size)
    local k = 1
    for i = t, math.min(t + batch_size - 1, num_train) do
      -- load new sample
      local sample = dataset[i]
      local input = sample[1]:clone()
      local target = y_train[i]
      --target = target:squeeze()
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
        f = f + l2_reg + norm(parameters, 2)^2/2

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
  print(confusion)
  print('% mean class accuracy (train set)' .. confusion.totalValid*100)
  confusion:zero()
end

-- let's train bitchez

for e = 1, 5 do
  train(X_train)
end