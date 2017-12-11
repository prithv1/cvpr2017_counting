-- Run sequential subitizing for ResNet features

require 'dpnn'
require 'rnn'
require 'torch'
require 'hdf5'
require 'optim'
require 'nn'
require 'eval'
require 'optim_updates'

local matio = require 'matio'
local color = require 'trepl.colorize'
local utils = require 'utils'
local json = require 'json'
local weight_init = require 'weight_init'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options for the context model')
cmd:option('-r', 3, 'Discretization for aso-sub')
cmd:option('-hid', 500, 'units in the hidden layer')
cmd:option('-nhid', 1, 'number of hidden layers')
cmd:option('-bsize', 64, 'batch size while training')
cmd:option('-epochs', 60, 'number of epochs to train for')
cmd:option('-exp', 'new_exp', 'experiment directory')
cmd:option('-lr', 0.01, 'Learning rate for training')
cmd:option('-gpuid', 1, 'GPU to use. Set 0 to use CPU')
cmd:option('-seed', 123, 'Random seed to standardize runs')
cmd:option('-set', 'coco', 'Dataset to work on')
cmd:option('-net', 'resnet-101', 'Network to extract features from')
cmd:option('-rel', 1, 'Whether to add relu after last layer')
cmd:option('-method', 'heuristic', 'Method for initializing the weights')
cmd:option('-estop', 20, 'Number of epochs for early-stopping')
cmd:option('-checkpt', 20, 'Number of epochs after which every model is to be saved')
cmd:option('-wt_dec', 0.01, 'Weight Decay')
cmd:option('-lr_dec', 0.3, 'Learning Rate Decay')
cmd:option('-decay_every', 15, 'Decay Learning Rate every n epochs')
cmd:option('-optimizer', 'rmsprop', 'sgd|rmsprop|adagrad|asgd|adadelta|adam|nag')
cmd:option('-backend', 'nn', 'Use nn|cudnn')
cmd:option('-iter_decay', 0, 'Iteration lr decay')
cmd:option('-loss', 'MSECriterion', 'Loss function to use')
cmd:option('-fix_sequence', 1, 'Fix the sequence ordering of patch features before taking any permutation')
cmd:option('-tr_load_models', 0, 'Train from partially trained models')
opt = cmd:parse(arg)

local exp_dir = 'seqsub_models/' .. opt.exp .. '_seq_sub_' .. opt.net .. '_imwise_' .. opt.set .. '_' .. opt.r .. '_' .. tostring(opt.nhid) .. 'hl_' .. tostring(opt.hid) .. '_' .. opt.optimizer .. '_rel_' .. tostring(opt.rel) .. '_lr_' .. tonumber(opt.lr) .. '_fix_sequence_' .. opt.fix_sequence

paths.mkdir(exp_dir)

local opt_table = opt
local table_json = json.encode(opt_table)
print(exp_dir)
print('-----')
print('-----')
print('-----')
print(table_json)

local file = io.open(exp_dir .. '/exp_config.json', 'w')

if file then
  file:write(table_json)
  io.close(file)
end

if opt.backend == 'cudnn' then
	require 'cudnn'
	cudnn.verbose = true
end

--Set defaults
torch.setdefaulttensortype('torch.DoubleTensor')
torch.manualSeed(opt.seed)

if opt.gpuid + 1 > 0 then
	require 'cutorch'
	require 'cunn'
	cutorch.setDevice(1)
end

local data_dir = '/home/prithv1/compositional_counting/resnet_comparisons/features/' .. opt.set .. '/' .. opt.net .. '_feats/'

if opt.set == 'pascal' then
	list_dir = '/home/prithv1/finetuning_count/VOC/imagelists/'
else
	list_dir = '/home/prithv1/finetuning_count/COCO/imagelists/'
end

local train_files = {}
local val_files = {}
local test_files = {}

local train_dir = list_dir .. 'train.txt'
local val_dir = list_dir .. 'val.txt'
local test_dir = list_dir .. 'test.txt'

local train_file = io.open(train_dir, 'r')
local val_file = io.open(val_dir, 'r')
local test_file = io.open(test_dir, 'r')

if train_file then for line in train_file:lines() do table.insert(train_files, line) end end
if val_file then for line in val_file:lines() do table.insert(val_files, line) end end
if test_file then for line in test_file:lines() do table.insert(test_files, line) end end

print('Image List prepared')

local train_path = data_dir .. 'train_aso-sub_' .. opt.r .. '/'
local val_path = data_dir .. 'val_aso-sub_' .. opt.r .. '/'
local test_path = data_dir .. 'test_aso-sub_' .. opt.r .. '/'

if opt.set == 'pascal' then
	Y_train = matio.load('/home/prithv1/neural_net_exp/count_voc07_train.mat','category_count')
	Y_val = matio.load('/home/prithv1/neural_net_exp/count_voc07_val.mat','category_count')
	Y_test = matio.load('/home/prithv1/neural_net_exp/count_voc07_test.mat','category_count')
	Y_test_gt = matio.load('/home/prithv1/neural_net_exp/count_voc07_test.mat','category_count')
else
	Y_train = matio.load('/home/prithv1/COCO/count_matrices/count_faster_coco_train2014_new.mat','category_count')
	vl = matio.load('/home/prithv1/COCO/count_matrices/count_faster_coco_val2014_new.mat','category_count')
	Y_val = vl:sub(1, 20252)
	Y_test = vl:sub(20253, 40504)
	Y_test_gt = vl:sub(20253,40504)
end

-----------------------
local ncat = Y_train:size(2)
local net_ip_size = 0
if opt.net == 'resnet-101' or opt.net == 'resnet-152' or opt.net == 'resnet-50' or opt.net == 'resnet-200' then
	net_ip_size = 2048
elseif opt.net == 'resnet-18' or opt.net == 'resnet-34' then
	net_ip_size = 512
end
print('Data Loaded..')

local disc_size = opt.r*opt.r

-- Permutation matrices for different directions

local perm1 = torch.eye(disc_size)
local prev_ind = torch.linspace(1, disc_size, disc_size)
local new_ind = torch.zeros(disc_size)
for i=1,disc_size do
  if i%opt.r ~= 0 then
	new_ind[i] = 1 + opt.r*((i%opt.r) - 1) + ((i - (i%opt.r))/opt.r)
  else
	new_ind[i] = disc_size + opt.r*((i%opt.r) - 1) + ((i - (i%opt.r))/opt.r)  
  end
end
local perm2 = torch.zeros(disc_size, disc_size)
for i=1, disc_size do
  perm2[{{prev_ind[i]},{new_ind[i]}}] = 1
end

-- Fix sequence
local mul_mat = torch.eye(disc_size)
if opt.fix_sequence == 1 then
	print('Fixed Sequence')
	local disc = opt.r
	local index_new = torch.linspace(1, disc_size, disc_size)
	local a = 1 + ((2*torch.linspace(1, (disc-1)/2, (disc-1)/2)) - 1)*disc
	local b = 2*torch.linspace(1,(disc-1)/2,(disc-1)/2)*disc
	local d = torch.zeros((disc-1)/2, disc)
	for i=1, (disc-1)/2 do
		index_new[{{a[i],b[i]}}] = torch.linspace(b[i], a[i], disc)
	end
	mul_mat = mul_mat:fill(0)
	for i=1, disc_size do
		mul_mat[{{i},{index_new[i]}}] = 1
	end
end

-- Prepare sequence
local function prepare_sequence(mat, feat, count)
	if feat == 1 then
		temp_data_1 = mat:clone()
		temp_data_2 = mat:clone()
		temp_data_1 = nn.MM():forward({perm1, temp_data_1})
		temp_data_1 = nn.MM():forward({mul_mat, temp_data_1})
		temp_data_2 = nn.MM():forward({perm2, temp_data_2})
		temp_data_2 = nn.MM():forward({mul_mat, temp_data_2})
		return temp_data_1, temp_data_2

	elseif count == 1 then
		temp_data = torch.zeros(disc_size, mat:size(2))
		temp_data = mat:clone()
		return temp_data
	end
end

local inv_1 = nn.MM():forward({perm1, mul_mat})
inv_1 = torch.repeatTensor(inv_1, opt.bsize, 1, 1)
local inv_2 = nn.MM():forward({perm2, mul_mat})
inv_2 = torch.repeatTensor(inv_2, opt.bsize, 1, 1)

-- View orderings
local order = torch.linspace(1, disc_size, disc_size)
order = nn.Unsqueeze(2):forward(order)
print('Permutation 1')
print(nn.MM():forward({perm1, order}))
print('Permutation 2')	
print(nn.MM():forward({perm2, order}))
print('Final Order 1')
local c = nn.MM():forward({mul_mat, perm1})
print(nn.MM():forward({c, order}))
print('Final Order 2')
local d = nn.MM():forward({mul_mat, perm2})
print(nn.MM():forward({d, order}))
print('Inverse Order 1')
print(nn.MM():forward({inv_1[1], nn.MM():forward({c, order})}))
print('Inverse Order 2')
print(nn.MM():forward({inv_2[1], nn.MM():forward({d, order})}))
----------------------------------------------------------------------------------------
--LSTM in all 4 directions
-- Model 1 having LSTM in all 4 directions

-- One end to end model
if opt.tr_load_models == 0 then
	model1 = nn.Sequential()
	c = nn.ConcatTable()
	for i = 1, 2 do
		--bilstm module
		bilstm = nn.Sequential()
		bilstm:add(nn.SelectTable(i))
		bilstm:add(nn.SplitTable(2))
		-- bilstm:add(nn.Sequencer(nn.BatchNormalization(net_ip_size)))
		bilstm:add(nn.Sequencer(nn.ReLU()))
		bilstm:add(nn.Sequencer(nn.Linear(net_ip_size, opt.hid)))
		-- bilstm:add(nn.Sequencer(nn.BatchNormalization(opt.hid)))
		bilstm:add(nn.Sequencer(nn.ReLU()))
		for j=1, opt.nhid do
			local lstm_sz = opt.hid*torch.pow(2, j-1)
			bilstm:add(nn.BiSequencer(nn.LSTM(lstm_sz, lstm_sz, disc_size), nn.LSTM(lstm_sz, lstm_sz, disc_size), nn.JoinTable(1,1)))
		end
		c:add(bilstm)
	end
	model1:add(c)
	model1:add(nn.FlattenTable())

	model2 = nn.Sequential()
	-- model2:add(nn.Sequencer(nn.BatchNormalization(opt.hid*torch.pow(2, opt.nhid + 1))))
	model2:add(nn.Sequencer(nn.ReLU()))
	model2:add(nn.Sequencer(nn.Linear(opt.hid*torch.pow(2, opt.nhid + 1), ncat)))
	if opt.rel == 1 then
		model2:add(nn.Sequencer(nn.ReLU()))
	end
	model1 = weight_init(model1, opt.method)
	model2 = weight_init(model2, opt.method)
else
	print('Loading Pretrained model')
	model1 = torch.load(exp_dir .. '/counting_best_1.t7')
	model2 = torch.load(exp_dir .. '/counting_best_2.t7')
end

criterion = nn.SequencerCriterion(nn[opt.loss]())
if opt.gpuid + 1 > 0 then
	model1:cuda()
	model2:cuda()
	criterion:cuda()
end

if opt.backend == 'cudnn' then
	require 'cudnn'
	cudnn.convert(model1, cudnn)
	cudnn.convert(model2, cudnn)
end

print(model1)
print(model2)

local ct = nn.Container()
ct:add(model1)
ct:add(model2)
local param, gparam = ct:getParameters()

--Optimizer
optimState = {
  learningRate = opt.lr,
  weightDecay = opt.wt_dec,
  learningRateDecay = opt.lr_dec
}

ep_step = opt.decay_every

local tr_loss = {}
local vl_loss = {}
local tr_loss_iter = {}
local logger = optim.Logger(exp_dir .. '/counting_seq_sub_epoch.log')
logger:setNames{'Training Error', 'Validation Error'}

local epoch = 1
local iter = 1
local lr = opt.lr
print('Experiment Configured')

local evaluation_count = eval_count(1,1)
-- Training
local function train()
	local time = sys.clock()
	model1:training()
	model2:training()
	local shuffle = torch.randperm(#train_files)	
	if epoch ~= 0 and epoch % ep_step == 0 and opt.optimizer ~= 'adagrad' and opt.optimizer ~= 'adam' then optimState.learningRate = optimState.learningRate*optimState.learningRateDecay end
	local train_losses = {}
	-- Create mini-batches on the fly
	local Xt_1 = torch.zeros(opt.bsize, disc_size, net_ip_size)
	local Xt_2 = torch.zeros(opt.bsize, disc_size, net_ip_size)
	local Yt = torch.zeros(opt.bsize, disc_size, ncat)
	for it = 1, #train_files, opt.bsize do
		local batch_id = ((it-1)/opt.bsize) + 1
		-- xlua.progress(batch_id, torch.floor(#train_files/opt.bsize))
		if (it + opt.bsize - 1) > #train_files then
			break
		end
		local idx = 1
		for i=it, it+opt.bsize-1 do
			local feat_path = train_path .. paths.basename(train_files[shuffle[i]], '.jpg') .. '.h5'
			local temp_h5 = hdf5.open(feat_path, 'r')
			Xt_1[idx], Xt_2[idx] = prepare_sequence(temp_h5:read('/data'):all(), 1, 0)
			Yt[idx] = prepare_sequence(temp_h5:read('/label'):all(), 0, 1)
			temp_h5:close()
			idx = idx + 1
		end
		if opt.gpuid + 1 > 0 then
			Xt_1 = Xt_1:cuda()
			Xt_2 = Xt_2:cuda()
			Yt = Yt:cuda()
		end
		local input, target = {}, {}
		table.insert(input, Xt_1)
		table.insert(input, Xt_2)
		target = nn.SplitTable(2):forward(Yt)
		-- Single feval function for container based param updates
		local feval = function(x)
		if x ~= param then param:copy(x) end
		model1:zeroGradParameters()
		model2:zeroGradParameters()
		local output1 = model1:forward(input)
		local op_table = {}
		for i=1,#output1 do
			local temp = output1[i]
			temp = temp:double()
			table.insert(op_table, nn.Unsqueeze(2):forward(temp))
		end
		local op = nn.JoinTable(2):forward(op_table)
		local op_1 = op:sub(1, op:size(1), 1, op:size(2)/2, 1, op:size(3))
		local op_2 = op:sub(1, op:size(1), 1 + op:size(2)/2, op:size(2), 1, op:size(3))
		op_1 = nn.MM():forward({inv_1, op_1})
		op_2 = nn.MM():forward({inv_2, op_2})
		op = torch.cat(op_1, op_2, 3)
		if opt.gpuid + 1 > 0 then
			op = op:cuda()
		end
		local inter_op = nn.SplitTable(2):forward(op)
		local output = model2:forward(inter_op)
		-- Get proper level loss
		local op_table_1 = {}
		for i=1, #output do
			local temp = output[i]
			temp = temp:double()
			temp = temp:cmax(0)
			table.insert(op_table_1, temp)
		end
		tr_op = nn.CAddTable():forward(op_table_1)
		local gt = torch.squeeze(Yt:sum(2))
		local mse_tr = evaluation_count:mse(0, tr_op, gt)
		local iter_loss = criterion:forward(output, target) 
		table.insert(train_losses, mse_tr)
		local gradOutputs_1 = criterion:backward(output, target)
		local gradOutputs_2 = model2:backward(inter_op, gradOutputs_1)
		local inter_g = {}
		for i=1, #gradOutputs_2 do
			local temp = gradOutputs_2[i]
			temp = temp:double()
			table.insert(inter_g, nn.Unsqueeze(2):forward(temp))
		end
		local grad = nn.JoinTable(2):forward(inter_g)
		local grad_1 = grad:sub(1, grad:size(1), 1, grad:size(2), 1, grad:size(3)/2)
		local grad_2 = grad:sub(1, grad:size(1), 1, grad:size(2), 1 + (grad:size(3)/2), grad:size(3))
		grad_1 = grad_1:double()
		grad_2 = grad_2:double()
		grad_1 = nn.MM():forward({torch.repeatTensor(perm1, grad_1:size(1), 1, 1), grad_1})
		grad_1 = nn.MM():forward({torch.repeatTensor(mul_mat, grad_1:size(1), 1, 1), grad_1})
		grad_2 = nn.MM():forward({torch.repeatTensor(perm2, grad_2:size(1), 1, 1), grad_2})
		grad_2 = nn.MM():forward({torch.repeatTensor(mul_mat, grad_2:size(1), 1, 1), grad_2})
		if opt.gpuid + 1 > 0 then
			grad_1 = grad_1:double()
			grad_2 = grad_2:double()
		end
		g_table1 = nn.SplitTable(2):forward(grad_1)
		g_table2 = nn.SplitTable(2):forward(grad_2)
		gradOutputs_3 = {g_table1, g_table2}
		gradOutputs_3 = nn.FlattenTable():forward(gradOutputs_3)
		model1:backward(input, gradOutputs_3)
		if gparam:norm() > 5 then
			gparam:mul(5/gparam:norm())
		end
		return iter_loss, gparam
		end
		-- Single optim call
		optim[opt.optimizer](feval, param, optimState)
		iter = iter + 1
	end
	time = sys.clock() - time
	print(color.red'Epoch: ' .. epoch .. color.blue' training loss: ' .. utils.table_mean(train_losses) .. color.blue' Learning Rate: ' .. optimState.learningRate .. color.blue' Time: ' .. time .. '    s')
	epoch = epoch + 1
	return utils.table_mean(train_losses)
end


local function test(model_test1, model_test2, split)
	eval_bsize = opt.bsize
	model_test1:evaluate()
	model_test2:evaluate()
	local all_counts = {}
	local Y_req = torch.Tensor()
	if split == 'val' then
		table_req = val_files
		req_path = val_path
		Y_req = Y_val:sub(1,torch.floor(#table_req/eval_bsize)*eval_bsize)
	else
		table_req = test_files
		req_path = test_path
		Y_req = Y_test_gt:sub(1,torch.floor(#table_req/eval_bsize)*eval_bsize)
	end
	local Xt_1 = torch.zeros(eval_bsize, disc_size, net_ip_size)
	local Xt_2 = torch.zeros(eval_bsize, disc_size, net_ip_size)
	local Y_pred = torch.zeros(torch.floor(#table_req/eval_bsize)*eval_bsize, ncat)
	for its = 1, #table_req, eval_bsize do
		local batch_id = ((its-1)/eval_bsize) + 1
		-- xlua.progress(batch_id, torch.floor(#table_req/eval_bsize))
		if (its + eval_bsize - 1) > #table_req then
			break
		end
		local idx = 1
		for i=its, its+eval_bsize-1 do
			local feat_path = req_path .. paths.basename(table_req[i], '.jpg') .. '.h5'
			local temp_h5 = hdf5.open(feat_path, 'r')
			Xt_1[idx], Xt_2[idx] = prepare_sequence(temp_h5:read('/data'):all(), 1, 0)
			temp_h5:close()
			idx = idx + 1
		end
		if opt.gpuid + 1 > 0 then
			Xt_1 = Xt_1:cuda()
			Xt_2 = Xt_2:cuda()
		end  
		local input, target = {}, {}
		table.insert(input, Xt_1)
		table.insert(input, Xt_2)
		local output1 = model_test1:forward(input)
		local op_table = {}
		for i=1,#output1 do
			local temp = output1[i]
			temp = temp:double()
			table.insert(op_table, nn.Unsqueeze(2):forward(temp))
		end
		local op = nn.JoinTable(2):forward(op_table)
		local op_1 = op:sub(1, op:size(1), 1, op:size(2)/2, 1, op:size(3))
		local op_2 = op:sub(1, op:size(1), 1 + op:size(2)/2, op:size(2), 1, op:size(3))
		op_1 = nn.MM():forward({inv_1, op_1})
		op_2 = nn.MM():forward({inv_2, op_2})
		op = torch.cat(op_1, op_2, 3)
		if opt.gpuid + 1 > 0 then
			op = op:cuda()
		end
		local inter_op = nn.SplitTable(2):forward(op)
		local output = model_test2:forward(inter_op)
		local op_table_1 = {}
		for i=1, #output do
			local temp = output[i]
			temp = temp:double()
			temp = temp:cmax(0)
			table.insert(op_table_1, temp)
		end
		ts_op = nn.CAddTable():forward(op_table_1)
		Y_pred[{{its,its+eval_bsize-1},{1,ncat}}] = ts_op
	end
	local mse = 0
	local mrmse = 0
	local mmae = 0
	local zero_mse = 0
	local one_mse = 0
	local Y_gt = Y_req
	if split == 'val' then
  		mse = evaluation_count:mse(0, Y_pred, Y_gt)
		mrmse = evaluation_count:mrmse(0, Y_pred, Y_gt)
  		mmae = evaluation_count:mmae(0, Y_pred, Y_gt)
  		print(string.format('Test Count Loss: split(%s)  mrmse: %f, mmae: %f, mse: %f', split, mrmse, mmae, mse))
	else
		mse, mse_std = evaluation_count:eval_for_paper('mse', 0, Y_pred, Y_gt)
    	mrmse, mrmse_std = evaluation_count:eval_for_paper('mrmse', 0, Y_pred, Y_gt)
    	mmae, mmae_std = evaluation_count:eval_for_paper('mmae', 0, Y_pred, Y_gt)
    	print(string.format('Test Count Loss: split(%s)  mrmse: %f std: %f, mmae: %f std: %f, mse: %f std: %f', split, mrmse, mrmse_std, mmae, mmae_std, mse, mse_std))
	end
  return mse, mrmse, mmae, Y_pred
end

local earlystop = opt.estop
local min_loss = 10000
local min_ep = 0
local lightmodel = {}

local i = 1
while(1) do
  if i % opt.checkpt == 0 or i == n_epochs then
	-- lightmodel1 = model1:clone()
	-- lightmodel2 = model2:clone()
	-- lightmodel1:clearState()
	-- lightmodel2:clearState()
	-- torch.save(exp_dir .. '/' .. 'all_dir_lstm_count_1_' .. 'ep_' .. tostring(i) .. '.t7', lightmodel1)
	-- torch.save(exp_dir .. '/' .. 'all_dir_lstm_count_2_' .. 'ep_' .. tostring(i) .. '.t7', lightmodel2)
	print('-----------')
	local mse, mrmse, mmae, pred = test(model1, model2, 'test')
	pred = nil
	print('-----------')
  end
  if i == n_epochs then
	break
  end
  local t_l = train()
  local Mse, Mrmse, Mmmae, c = test(model1, model2, 'val')
  c = nil
  local v_l = Mse
  if min_loss > v_l then
	min_loss = v_l
	min_ep = i
	-- lightmodel1 = model1:clone()
	-- lightmodel2 = model2:clone()
	-- lightmodel1:clearState()
	-- lightmodel2:clearState()
	torch.save(exp_dir .. '/counting_best_1.t7', model1)
	torch.save(exp_dir .. '/counting_best_2.t7', model2)
  end
  if i > 2 then
	logger:add{t_l, v_l}
	table.insert(tr_loss, t_l)
	table.insert(vl_loss, v_l)
	-- logger.showPlot = false
	-- logger.epsfile = '/home/prithv1/public_html/torch_logs/' .. exp_dir .. '.eps'
	-- logger:style{'-', '-'}
	-- logger:plot()
	-- if i~=1 then
	--   os.execute('sh /home/prithv1/public_html/torch_logs/convert.sh')
	-- end
  end
  if v_l > min_loss and (i-min_ep) == earlystop then
	break
  end
  i = i + 1
end

model_new_1 = torch.load(exp_dir .. '/counting_best_1.t7')
model_new_2 = torch.load(exp_dir .. '/counting_best_2.t7')
local MSE, MRMSE, MMAE, Predictions = test(model_new_1, model_new_2, 'test')
local best_losses = {}
best_losses['Test Count MSE'] = MSE
best_losses['Test Count MRMSE'] = MRMSE
best_losses['Test Count MMAE'] = MMAE
best_losses['Predictions'] = Predictions
torch.save(exp_dir .. '/loss_and_pred.t7', best_losses)
local loss_json = json.encode(best_losses)
local l_file = io.open(exp_dir .. '/test_results.json', 'w')
if l_file then
  l_file:write(loss_json)
  io.close(l_file)
end
print('---------')
print('---------')
print('---------')











