#! /usr/bin/env lua
--[[
Forward pass utilities
1. Contains functions that have be invoked while conducting a fprop for any of the models
2. Glance and Aso-sub differ only in terms of data-preprocessing
3. Seq-sub is completely different and involves sequential processing
]]

require 'nn'
require 'rnn'
require 'sys'
require 'xlua'
require 'paths'
require 'hdf5'
require 'math'

local fprop_utils = torch.class('fprop_utils')

function fprop_utils:__init(feature_directory, imagelist_dir, disc, feature_dimensions, num_classes)
	--[[
	(Initialize settings for forward passes)
	Arguments
	**********
	feature_directory: the directory containing the feature files
	imagelist_dir: directory containing the image lists
	disc: the discretization at which we're dealing - 1 for glance, 3 or more for aso-sub/seq-sub
	feature_dimensions: dimensions of the feature directory
	num_classes: the number of output classes

	Returns
	**********
	(Nothing, just initializes stuff)
	]]
	self.feat_dir = feature_directory
	self.imlist_dir = imagelist_dir
	self.feat_dim = feature_dimensions
	self.nout = num_classes
	self.disc = disc
	self.disc_size = disc*disc
	if self.disc > 1 then
		self.perm1 = torch.eye(self.disc_size)
		self.prev_ind = torch.linspace(1, self.disc_size, self.disc_size)
		self.new_ind = torch.zeros(self.disc_size)
		for i = 1,self.disc_size do
			if i % self.disc ~= 0 then
				self.new_ind[i] = 1 + self.disc*((i % self.disc) - 1) + ((i - (i % self.disc))/self.disc)
			else
				self.new_ind[i] = self.disc_size + self.disc*((i % self.disc) - 1) + ((i - (i % self.disc))/self.disc)
			end
		end
		self.perm2 = torch.zeros(self.disc_size, self.disc_size)
		for i = 1,self.disc_size do
			self.perm2[{{self.prev_ind[i]}, {self.new_ind[i]}}] = 1
		end
		self.mul_mat = torch.eye(self.disc_size)
		self.mul_ind = torch.linspace(1, self.disc_size, self.disc_size)
		local a = 1 + ((2*torch.linspace(1, (self.disc - 1)/2, (self.disc - 1)/2)) - 1)*self.disc
		local b = 2*torch.linspace(1, (self.disc - 1)/2, (self.disc - 1)/2)*self.disc
		for i= 1,(self.disc - 1)/2 do
			self.mul_ind[{{a[i], b[i]}}] = torch.linspace(b[i], a[i], self.disc)
		end
		self.mul_mat = self.mul_mat:fill(0)
		for i= 1,self.disc_size do
			self.mul_mat[{{i}, {self.mul_ind[i]}}] = 1
		end
	end
end

function get_batchsize(split_size, des_bsize)
	--[[
	(Get closest batchsize to cover the entire split)
	Arguments
	**********
	split_size: size of the entire split
	des_bsize: desired batch-size

	Returns
	**********
	exp_bsize: the batch-size closest to the one required
	]]
	local factors = {}
	for possible_factor = 1,math.sqrt(split_size),1 do
		local remainder = split_size % possible_factor
		if remainder == 0 then
			local factor, factor_pair = possible_factor, split_size/possible_factor
			table.insert(factors, factor)
			if factor ~= factor_pair then
				table.insert(factors, factor_pair)
			end
		end
	end
	table.sort(factors)
	factor_tensor = torch.Tensor(factors)
	_, ind = torch.min(torch.abs(factor_tensor - des_bsize), 1)
	exp_bsize = factor_tensor[ind[1]]
	return exp_bsize
end

function prepare_sequence(feat_mat, perm1, perm2, mul_mat)
	--[[
	(Create permuted ordering of features for sequential subitizing)
	Arguments
	**********
	feat_mat: feature matrix containing features for all the cells
	perm1: the first permutation matrix fir seq-sub
	perm2: the second permutation matrix fir seq-sub
	mul_mat: matrix to fix ordering interms of close-cells

	Returns
	**********
	perm_feat1, perm_feat2: permuted feature vectors
	]]
	local perm_feat1 = feat_mat:clone()
	local perm_feat2 = feat_mat:clone()
	perm_feat1 = nn.MM():forward({perm1, perm_feat1})
	perm_feat1 = nn.MM():forward({mul_mat, perm_feat1})
	perm_feat2 = nn.MM():forward({perm2, perm_feat2})
	perm_feat2 = nn.MM():forward({mul_mat, perm_feat2})
	return perm_feat1, perm_feat2
end

function fprop_utils:glance_fprop(split_list, model, gpu_flag, gpu_id, cudnn_flag, des_bsize)
	--[[
	(Forward pass for glancing)
	Arguments
	**********
	split_list: list of images for the corresponding split
	model: nn.Sequential() model
	gpu_flag: whether to use a GPU or not
	gpu_id: corresponding GPU IDs
	cudnn_flag: whether to CuDNN
	des_bsize: desired batchsize

	Returns
	**********
	Yo: return predictions for the whole split
	]]
	model:evaluate()
	local img_list = {}
	local img_file = io.open(self.imlist_dir .. '/' .. split_list)
	if img_file then for line in img_file:lines() do table.insert(img_list, line) end end
	-- Get batch size
	local bsize = get_batchsize(#img_list, des_bsize)
	-- Feature tensor per-batch
	Xt = torch.zeros(bsize, self.feat_dim)
	Yt = torch.zeros(bsize, self.nout)
	-- Predictions for the whole split
	Yo = torch.zeros(#img_list, self.nout)
	if gpu_flag then
		require 'cutorch'
		require 'cunn'
		cutorch.setDevice(gpu_id + 1)
		model:cuda()
		Yo = Yo:cuda()
		if cudnn_flag then
			require 'cudnn'
			cudnn.benchmark = true
			cudnn.fastest = true
			cudnn.verbose = true
			cudnn.convert(model, cudnn)	
		end
	end
	for it = 1,#img_list,bsize do
		local batch_id = ((it-1)/bsize) + 1
		xlua.progress(batch_id, torch.floor(#img_list/bsize))
		if (it + bsize - 1) > #img_list then
			break
		end
		local idx = 1
		for i = it,it+bsize-1 do
			local feat_path = self.feat_dir .. '/' .. paths.basename(img_list[i], '.jpg') .. '.h5'
			local feat_h5 = hdf5.open(feat_path, 'r')
			Xt[idx] = feat_h5:read('/data'):all()
			feat_h5:close()
			idx = idx + 1
		end
		if gpu_flag then
			Xt = Xt:cuda()
			Yt = Yt:cuda()
		end
		Yt = model:forward(Xt)
		Yo[{{it, it + bsize - 1}, {1, self.nout}}] = Yt
	end
	return Yo
end

function fprop_utils:aso_fprop(split_list, model, gpu_flag, gpu_id, cudnn_flag, des_bsize)
	--[[
	(Forward pass for associative subitizing)
	Arguments
	**********
	split_list: list of images for the corresponding split
	model: nn.Sequential() model
	gpu_flag: whether to use a GPU or not
	gpu_id: corresponding GPU IDs
	cudnn_flag: whether to CuDNN
	des_bsize: desired batchsize

	Returns
	**********
	Yo: return predictions for the whole split
	]]
	model:evaluate()
	-- Keep smaller batch size for associative subitizing
	local img_list = {}
	local img_file = io.open(self.imlist_dir .. '/' .. split_list)
	if img_file then for line in img_file:lines() do table.insert(img_list, line) end end
	-- Get batch size
	local bsize = get_batchsize(#img_list, des_bsize)
	-- Feature tensor per-batch
	Xt = torch.zeros(bsize*self.disc_size, self.feat_dim)
	Yt = torch.zeros(bsize*self.disc_size, self.nout)
	-- Predictions for the whole split
	Yo = torch.zeros(#img_list, self.nout)
	if gpu_flag then
		require 'cutorch'
		require 'cunn'
		cutorch.setDevice(gpu_id + 1)
		model:cuda()
		Yo = Yo:cuda()
		if cudnn_flag then
			require 'cudnn'
			cudnn.benchmark = true
			cudnn.fastest = true
			cudnn.verbose = true
			cudnn.convert(model, cudnn)	
		end
	end
	for it = 1,#img_list,bsize do
		local batch_id = ((it-1)/bsize) + 1
		xlua.progress(batch_id, torch.floor(#img_list/bsize))
		if (it + bsize - 1) > #img_list then
			break
		end
		local idx = 1
		for i = it,it+bsize-1 do
			local feat_path = self.feat_dir .. '/' .. paths.basename(img_list[i], '.jpg') .. '.h5'
			local feat_h5 = hdf5.open(feat_path, 'r')
			Xt[{{1+(idx-1)*self.disc_size, idx*self.disc_size}, {1, self.feat_dim}}] = feat_h5:read('/data'):all()
			feat_h5:close()
			idx = idx + 1
		end
		if gpu_flag then
			Xt = Xt:cuda()
			Yt = Yt:cuda()
		end
		Yt = model:forward(Xt)
		-- Compress count predictions from the model to get dataset predictions
		local comp_idx = 1
		for i = it, it+bsize-1 do
			local count_ten = Yt[{{1+(comp_idx-1)*self.disc_size, comp_idx*self.disc_size}, {1, self.nout}}]
			Yo[{{i}, {1, self.nout}}] = count_ten:double():sum(1):squeeze()
			comp_idx = comp_idx + 1
		end
	end
	return Yo
end

function fprop_utils:seq_fprop(split_list, model1, model2, gpu_flag, gpu_id, cudnn_flag, des_bsize)
	--[[
	(Forward pass for glancing)
	Arguments
	**********
	split_list: list of images for the corresponding split
	model1: nn.Sequential() model to get bi-LSTMs output states
	model2: nn.Sequential() model to get cell-level predictions
	gpu_flag: whether to use a GPU or not
	gpu_id: corresponding GPU IDs
	cudnn_flag: whether to CuDNN
	des_bsize: desired batchsize

	Returns
	**********
	Yo: return predictions for the whole split
	]]
	model1:evaluate()
	model2:evaluate()
	-- Keep smaller batch size for associative subitizing
	local img_list = {}
	local img_file = io.open(self.imlist_dir .. '/' .. split_list)
	if img_file then for line in img_file:lines() do table.insert(img_list, line) end end
	-- Get batch size
	local bsize = get_batchsize(#img_list, des_bsize)
	-- Feature tensors and tables per-batch
	local Xt1 = torch.zeros(bsize, self.disc_size, self.feat_dim)
	local Xt2 = torch.zeros(bsize, self.disc_size, self.feat_dim)
	local Yt = torch.zeros(bsize, self.disc_size, self.nout)
	-- Predictions for the whole split
	local Yo = torch.zeros(#img_list, self.nout)
	-- Inverse permutation matrices
	local inv1 = nn.MM():forward({self.perm1, self.mul_mat})
	inv1 = torch.repeatTensor(inv1, bsize, 1, 1)
	local inv2 = nn.MM():forward({self.perm2, self.mul_mat})
	inv2 = torch.repeatTensor(inv2, bsize, 1, 1)
	if gpu_flag then
		require 'cutorch'
		require 'cunn'
		cutorch.setDevice(gpu_id + 1)
		model1:cuda()
		model2:cuda()
		Yo = Yo:cuda()
		if cudnn_flag then
			require 'cudnn'
			cudnn.benchmark = true
			cudnn.fastest = true
			cudnn.verbose = true
			cudnn.convert(model, cudnn)	
		end
	end
	for it = 1,#img_list,bsize do
		local batch_id = ((it-1)/bsize) + 1
		xlua.progress(batch_id, torch.floor(#img_list/bsize))
		if (it + bsize - 1) > #img_list then
			break
		end
		local idx = 1
		for i = it,it+bsize-1 do
			local feat_path = self.feat_dir .. '/' .. paths.basename(img_list[i], '.jpg') .. '.h5'
			local feat_h5 = hdf5.open(feat_path, 'r')
			Xt1[idx], Xt2[idx] = prepare_sequence(feat_h5:read('/data'):all(), self.perm1, self.perm2, self.mul_mat)
			feat_h5:close()
			idx = idx + 1
		end
		if gpu_flag then
			Xt1 = Xt1:cuda()
			Xt2 = Xt2:cuda()
			Yt = Yt:cuda()
		end
		local input, target = {}, {}
		table.insert(input, Xt1)
		table.insert(input, Xt2)
		local op_state = model1:forward(input)
		local op_state_table = {}
		for i = 1,#op_state do
			local temp = op_state[i]
			temp = temp:double()
			table.insert(op_state_table, nn.Unsqueeze(2):forward(temp))
		end
		local op = nn.JoinTable(2):forward(op_state_table)
		local op_1 = op:sub(1, op:size(1), 1, op:size(2)/2, 1, op:size(3))
		local op_2 = op:sub(1, op:size(1), 1 + op:size(2)/2, op:size(2), 1, op:size(3))
		op_1 = nn.MM():forward({inv1, op_1})
		op_2 = nn.MM():forward({inv2, op_2})
		op_tensor = torch.cat(op_1, op_2, 3)
		if gpu_flag then
			op_tensor = op_tensor:cuda()
		end
		local inter_ip = nn.SplitTable(2):forward(op_tensor)
		local output = model2:forward(inter_ip)
		local cell_op = {}
		for i=1, #output do
			local temp = output[i]
			temp = temp:double()
			temp = temp:cmax(0)
			table.insert(cell_op, temp)
		end
		Yo[{{it, it+bsize-1}, {1, self.nout}}] = nn.CAddTable():forward(cell_op)
	end
	return Yo
end
