#! /usr/bin/env lua
--[[
Script to train different models
1. Import the models from model_utils.lua
2. Import the training functions from train_utils.lua
3. Import the fprop functions from fprop_utils.lua
]]

require 'eval'
require 'optim'
require 'fprop_utils'
require 'model_utils'
require 'train_utils'

local matio = require 'matio'
local json = require 'json'
local color = require 'trepl.colorize'
local weight_init = require 'weight_init'

-- Fill these in later
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options for training the model')
cmd:option('-seed', 123, 'Random seed for experiments')
cmd:option('-set', 'pascal', 'Dataset to use: pascal/coco')
cmd:option('-disc', 1, 'Discretization at which you want to operate: 1 (glance); 3 or more (aso_sub, seq_sub)')
cmd:option('-exp_dir', 'test', 'Directory in which to store the models/logs')
cmd:option('-feat_dir', 'data/features', 'Directory in which CNN features are stored')
cmd:option('-list_dir', 'imagelists/pascal', 'Directory containing the imagelists corresponding to each split')
cmd:option('-feature_dim', 2048, 'Dimensions of the input feature vectors')
cmd:option('-num_classes', 20, 'Number of output classes')
cmd:option('-num_epochs', 100, 'Maximum number of epochs to train the models for')
cmd:option('-optimizer', 'adam', 'Optimizer to use')
cmd:option('-learning_rate', 1e-3, 'Learning rate to use')
cmd:option('-weight_decay', 0.95, 'Weight decay to use')
cmd:option('-loss', 'MSECriterion', 'Loss function to use')
cmd:option('-train_gt', 'data/count_gt/pascal_train_gt.mat', 'Path to train split ground truth count')
cmd:option('-val_gt', 'data/count_gt/pascal_val_gt.mat', 'Path to val split ground truth count')
cmd:option('-test_gt', 'data/count_gt/pascal_test_gt.mat', 'Path to test split ground truth count')
cmd:option('-tr_load_models', 0, 'Load trained checkpoint')
cmd:option('-model_type', 'glance', 'Type of model to train')
cmd:option('-num_hidden', 2, 'Number of hidden layers to use')
cmd:option('-num_bilstms', 2, 'Number of bi-directional LSTMs to use in seq_sub')
cmd:option('-hidden_size', 500, 'Hidden layer size')
cmd:option('-include_relu', 0, 'Whether to include relu after the last layer or not')
cmd:option('-method', 'heuristic', 'How to initialize the weights of the network')
cmd:option('-estop', 20, 'Number of epochs to use for early stopping criterion')
cmd:option('-checkpt', 10, 'Number of epochs after which to regularly save checkpoints')
cmd:option('-gpu_flag', true, 'Whether to use GPU or not')
cmd:option('-gpu_id', 1, 'GPU-ID to use')
cmd:option('-cudnn_flag', false, 'Whether to use CuDNN or not')
cmd:option('-des_bsize', 64, 'Desired batch size. Batch size used will cover the entire split')


opt = cmd:parse(arg)

-- Load arguments
local disc = opt.disc
local exp_dir = opt.exp_dir
local data_dir = opt.feat_dir
local list_dir = opt.list_dir

-- Make directories
paths.mkdir(exp_dir)

-- Decide criterion to train for
if opt.model_type == 'seq_sub' then
	criterion = nn.SequencerCriterion(nn[opt.loss]())
else
	criterion = nn[opt.loss]()
end

-- Load utilities
local model_utils = model_utils(opt.feature_dim, opt.num_classes)
local fprop_utils = fprop_utils(data_dir, list_dir, disc, opt.feature_dim, opt.num_classes)
local train_utils = train_utils(data_dir, list_dir, disc, opt.feature_dim, opt.num_classes, opt.num_epochs, opt.optimizer, opt.learning_rate, opt.weight_decay, criterion)
local evaluation_count = eval_count(1, 1)

print('Arguments specified..')
print(opt)

print('Saving in directory.. ' .. opt.exp_dir)

print('-------')
print('-------')
print('-------')
local table_json = json.encode(opt)

-- Save experiment config
local file = io.open(exp_dir .. '/exp_config.json', 'w')
if file then
	file:write(table_json)
	io.close(file)
end

-- Set Defaults
print('Setting defaults..')
torch.setdefaulttensortype('torch.DoubleTensor')
torch.manualSeed(opt.seed)

-- Load the ground truth counts
train_gt = matio.load(opt.train_gt, 'category_count')
val_gt = matio.load(opt.val_gt, 'category_count')
test_gt = matio.load(opt.test_gt, 'category_count')

-- Load or define models
print('Preparing models..')
print('--------------------------------------------------------------')
if opt.tr_load_models == 0 then
	if opt.model_type == 'glance' or opt.model_type == 'aso_sub' then
		model = model_utils:glance(opt.num_hidden, opt.hidden_size, opt.include_relu)
		model = weight_init(model, opt.method)
		print(model)
	elseif opt.model_type == 'seq_sub' then
		model1, model2 = model_utils:seq_sub(opt.num_bilstms, opt.hidden_size, opt.include_relu, disc*disc)
		model1 = weight_init(model1, opt.method)
		model2 = weight_init(model2, opt.method)
		print(model1)
		print(model2)
	end
else
	print('Loading pretrained model')
	if opt.model_type == 'seq_sub' then
		model1 = torch.load(exp_dir .. '/counting_best_1.t7')
		model2 = torch.load(exp_dir .. '/counting_best_2.t7')
		print(model1)
		print(model2)
	else
		model = torch.load(exp_dir .. '/counting_best.t7')
		print(model)
	end
end
print('--------------------------------------------------------------')

-- Extract parameters
print('Extracting parameters')
if opt.model_type == 'seq_sub' then
	ct = nn.Container()
	ct:add(model1)
	ct:add(model2)
	param, gparam = ct:getParameters()
else
	param, gparam = model:getParameters()
end

-- Write training loops
local i = 1
local min_loss = 10000
local min_ep = 0
local tr_loss = {}
local vl_loss = {}
local logger = optim.Logger(exp_dir .. '/' .. opt.model_type .. '_loss_per_epoch.log')
logger:setNames{'Training Loss', 'Validation Loss'}
local early_stop = opt.estop

print('Starting training..')
while(1) do
	if i % opt.checkpt == 0 or i == opt.num_epochs then
		if opt.model_type == 'seq_sub' then
			count_pred = fprop_utils:seq_fprop('val.txt', model1, model2, opt.gpu_flag, opt.gpu_id, opt.cudnn_flag, opt.des_bsize)
			lightmodel1 = model1:clone()
			lightmodel2 = model2:clone()
			lightmodel1:clearState()
			lightmodel2:clearState()
			torch.save(exp_dir .. '/model1' .. '_ep_' .. tostring(i) .. '.t7', lightmodel1)
			torch.save(exp_dir .. '/model2' .. '_ep_' .. tostring(i) .. '.t7', lightmodel2)
		else
			if opt.model_type == 'glance' then
				count_pred = fprop_utils:glance_fprop('val.txt', model, opt.gpu_flag, opt.gpu_id, opt.cudnn_flag, opt.des_bsize) 
			elseif opt.model_type == 'aso_sub' then
				count_pred = fprop_utils:aso_fprop('val.txt', model, opt.gpu_flag, opt.gpu_id, opt.cudnn_flag, opt.des_bsize)
			end	
			lightmodel = model:clone()
			lightmodel:clearState()
			torch.save(exp_dir .. '/model' .. '_ep_' .. tostring(i) .. '.t7', lightmodel)
		end
		-- Run evaluation on predictions
		val_mrmse = evaluation_count:sampled_eval('mrmse', 0, count_pred, val_gt, 10)
		val_mrmse_nz = evaluation_count:sampled_eval('mrmse', 1, count_pred, val_gt, 10)
		val_rel_mrmse = evaluation_count:sampled_eval('rel_mrmse', 0, count_pred, val_gt, 10)
		val_rel_mrmse_nz = evaluation_count:sampled_eval('rel_mrmse', 1, count_pred, val_gt, 10)
		print('-----------------------')
		print('Model performance before saving..')
		print('-----------------------')
		print(string.format('Val Count Loss: mrmse: %f, rel_mrmse: %f', val_mrmse, val_rel_mrmse))
		print(string.format('Val Count Loss (Non-zero): mrmse: %f, rel_mrmse: %f', val_mrmse_nz, val_rel_mrmse_nz))
		print('-----------------------')
	end
	if i == opt.num_epochs then
		break
	end
	local time = sys.clock()
	if opt.model_type == 'glance' or opt.model_type == 'aso_sub' then
		train_loss = train_utils:glance_train('train.txt', model, opt.gpu_flag, opt.gpu_id, opt.cudnn_flag, opt.des_bsize, param, gparam)
		if opt.model_type == 'glance' then
			count_pred = fprop_utils:glance_fprop('val.txt', model, opt.gpu_flag, opt.gpu_id, opt.cudnn_flag, opt.des_bsize)
		elseif opt.model_type == 'aso_sub' then
			count_pred = fprop_utils:aso_fprop('val.txt', model, opt.gpu_flag, opt.gpu_id, opt.cudnn_flag, opt.des_bsize)
		end
	elseif opt.model_type == 'seq_sub' then
		train_loss = train_utils:seq_train('train.txt', model1, model2, opt.gpu_flag, opt.gpu_id, opt.cudnn_flag, opt.des_bsize, param, gparam)
		count_pred = fprop_utils:seq_fprop('val.txt', model1, model2, opt.gpu_flag, opt.gpu_id, opt.cudnn_flag, opt.des_bsize)
	end
	time = sys.clock() - time
	val_loss = evaluation_count:sampled_eval('mse', 0, count_pred, val_gt, 10)
	table.insert(tr_loss, train_loss)
	table.insert(vl_loss, val_loss)
	print(color.red'Epoch: ' .. i .. color.blue' Training Loss: ' .. train_loss .. color.blue' Validation Loss: ' .. val_loss .. color.blue' Time: ' .. time .. '    s')
	if min_loss > val_loss then
		min_loss = val_loss
		min_ep = i
		if opt.model_type == 'seq_sub' then
			lightmodel1 = model1:clone()
			lightmodel2 = model2:clone()
			lightmodel1:clearState()
			lightmodel2:clearState()
			torch.save(exp_dir .. '/model1' .. '_ep_' .. tostring(i) .. '.t7', lightmodel1)
			torch.save(exp_dir .. '/model2' .. '_ep_' .. tostring(i) .. '.t7', lightmodel2)
		else
			lightmodel = model:clone()
			lightmodel:clearState()
			torch.save(exp_dir .. '/model' .. '_ep_' .. tostring(i) .. '.t7', lightmodel)
		end
	end
	if val_loss > min_loss and (i-min_ep) == early_stop then
		break
	end
	i = i + 1
end

-- Show test set performance
if opt.model_type == 'glance' then
	test_pred = fprop_utils:glance_fprop(test_path, 'test.txt', model, opt.gpu_flag, opt.gpu_id, opt.cudnn_flag, opt.des_bsize)
elseif opt.model_type == 'aso_sub' then
	test_pred = fprop_utils:aso_fprop(test_path, 'test.txt', model, opt.gpu_flag, opt.gpu_id, opt.cudnn_flag, opt.des_bsize)
elseif opt.model_type == 'seq_sub' then
	test_pred = fprop_utils:seq_fprop(test_path, 'test.txt', model1, model2, opt.gpu_flag, opt.gpu_id, opt.cudnn_flag, opt.des_bsize)
end

-- Save predictions
torch.save(exp_dir .. '/model_predictions.t7', test_pred)

-- Run evaluation on predictions
test_mrmse = evaluation_count:sampled_eval('mrmse', 0, test_pred, test_gt, 10)
test_mrmse_nz = evaluation_count:sampled_eval('mrmse', 1, test_pred, test_gt, 10)
test_rel_mrmse = evaluation_count:sampled_eval('rel_mrmse', 0, test_pred, test_gt, 10)
test_rel_mrmse_nz = evaluation_count:sampled_eval('rel_mrmse', 1, test_pred, test_gt, 10)
print('-----------------------')
print('Test Set Performance')
print('-----------------------')
print(string.format('Test Count Loss: mrmse: %f, rel_mrmse: %f', test_mrmse, test_rel_mrmse))
print(string.format('Test Count Loss (Non-zero): mrmse: %f, rel_mrmse: %f', test_mrmse_nz, test_rel_mrmse_nz))
print('-----------------------')
