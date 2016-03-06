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

geometry = {48, 48} -- Size of image
H = geometry[1] -- Height of image
num_classes = 7 -- 7 emotions
classes = { 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral' }

---------------------------------------------------------------

--image.display(X_train[{1, {}, {}, {} }])

-- size of data
num_train = 25000
num_val = 3000

train_set = {}
train_set.data = X_train[{ {1, num_train}, {}, {}, {} }]
train_set.label = y_train[{ {1, num_train}}]
train_set.label = train_set.label:byte()

val_set = {}
val_set.data = X_val[{ {1, num_val}, {}, {}, {} }]
val_set.label = y_val[{ {1, num_val} }]
val_set.label = val_set.label:byte()



---------------------------------------------------------------

-- net params
num_hidden = 300
num_filters = {32, 64, 64}
filter_size = {3,3,3,3}
dropout = {0.1, 0.25, 0.5, 0.5}

-- net
net = nn.Sequential()

-- conv - relu -- max pool
net:add(nn.SpatialConvolution(1, num_filters[1], filter_size[1], filter_size[1], 1, 1, 1, 1)) -- 1 input image channel, 32 output channels, 5x5 convolution kernel
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
--net:add(nn.Dropout(dropout[1]))

-- conv - relu -- max pool
net:add(nn.SpatialConvolution(num_filters[1], num_filters[2], filter_size[2], filter_size[2], 1, 1, 1, 1)) -- 32 input image channel, 32 output channels, 5x5 convolution kernel
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
--net:add(nn.Dropout(dropout[2]))

-- conv - relu - max pool
net:add(nn.SpatialConvolution(num_filters[2], num_filters[3], filter_size[3], filter_size[3], 1, 1, 1, 1)) -- 32 input image channel, 32 output channels, 5x5 convolution kernel
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
--net:add(nn.Dropout(dropout[3]))

-- affine - relu
net:add(nn.View(num_filters[3]*H/8*H/8))  -- reshapes from a 3D tensor to 1D tensor
net:add(nn.Linear(num_filters[3]*H/8*H/8, num_hidden))  -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())

-- affine - softmax
net:add(nn.Linear(num_hidden, num_classes))

criterion = nn.CrossEntropyCriterion()

print("Done setting up conv net")

-- Number of parameters
parameters, gradParams = net:getParameters()
print("Number of parameters:")
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

confusion = optim.ConfusionMatrix(classes)

function val()
      confusion:zero()

      for i=1,num_val do
        local groundtruth = val_set.label[i]
        local prediction = net:forward(val_set.data[i])
        confusion:add(prediction, groundtruth)
      end

      confusion:updateValids()
      print('% Validation accuracy: ' .. confusion.totalValid*100)

end

---------------------------------------------------------------

-- set up solver

local parameters, gradParameters = net:getParameters()

-- math.randomseed( os.time() ); math.random(); math.random(); math.random(); -- Well Lua sucks for random numbers...

loss_history = {}
-- Parameter search
-- regularization params
l2_reg = 0.001
--weight_scale = math.pow( 10, -3 * math.random() - 1.0 )

-- learning params
learning_rate = 0.002
learning_decay = 0.95
num_epochs = 50
batch_size = 50


config = {
   learningRate = learning_rate,
   beta1 = 0.9,
   beta2 = 0.999,
   epsilon = 1e-8,
   state = state
}


-- Reset the net
--[[]
net.modules[1].weight = torch.rand(num_filters, 1, filter_size, filter_size):mul(weight_scale)
net.modules[1].bias = torch.rand(num_filters):mul(weight_scale)
net.modules[4].weight = torch.rand(num_filters2, num_filters, filter_size2, filter_size2):mul(weight_scale)
net.modules[4].bias = torch.rand(num_filters2):mul(weight_scale)
net.modules[8].weight = torch.rand(num_hidden, num_filters2*H/4*H/4):mul(weight_scale)
net.modules[8].bias = torch.rand(num_hidden):mul(weight_scale)
net.modules[10].weight = torch.rand(num_classes, num_hidden):mul(weight_scale)
net.modules[10].bias = torch.rand(num_classes):mul(weight_scale)
]]--

confusion:zero()

function train(dataset)

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

      -- Print loss
      if (t-1) % 5*batch_size == 0 then
          loss_history[#loss_history + 1] = f
      end

      -- return f and df/dX
      return f, gradParameters
    end


    -- optimize on current mini batch

    state = state or config

    optim.adam(feval, parameters, config, state)

    xlua.progress(t, num_train)

  end

  confusion:updateValids()
  print('\n% Train accuracy: ' .. confusion.totalValid*100)
  confusion:zero()
  -- Update learning rate
  config.learningRate = config.learningRate * learning_decay
end

start_time = os.time()

-- let's train
for e = 1, num_epochs do
  print("\nEpoch # " .. e)
  train(X_train)
  val()
end

end_time = os.time()
elapsed_time = os.difftime(end_time, start_time)
print("Elapsed time:")
print(elapsed_time)

print("Loss history:")
print(loss_history)

print(confusion)
--torch.save("deep_net", net)




-- Do some saliency Map stuff over here




