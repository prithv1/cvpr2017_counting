#! /usr/bin/env lua

--[[
Script contains model definitions
1. Glancing/Aso-sub model : Simple MLP that takes in fc7 features as input and regresses to the count vectors
2. Sequential-sub model: Stacked bi-LSTMs that takes in cell-level features and regresses to the cell count vectors
]]

require 'nn'
require 'rnn'
require 'dpnn'

local model_utils = torch.class('model_utils')

function model_utils:__init(input_dimensions, num_classes)
	--[[
	Initialize input-output dimensionalities
	]]
	self.input_dimensions = input_dimensions
	self.num_classes = num_classes
end

function model_utils:glance(num_hidden, hidden_size, include_relu)
	--[[
	(Aso-sub and glance share the same model structure)
	Arguments
	**********
	num_hidden: number of hidden layers in the MLP
	hidden_size: number of activations in the hidden layers in the MLP
	include_relu: whether to include relu after the final fully-connected layer

	Returns
	**********
	model: a nn.Sequential() model structure
	]]
	local model = nn.Sequential()
	model:add(nn.Linear(self.input_dimensions, hidden_size))
	model:add(nn.BatchNormalization(hidden_size))
	model:add(nn.ReLU())
	if num_hidden > 1 then
		for i = 1,num_hidden-1 do
			model:add(nn.Linear(hidden_size, hidden_size))
			model:add(nn.BatchNormalization(hidden_size))
			model:add(nn.ReLU())
		end
	end
	model:add(nn.Linear(hidden_size, self.num_classes))
	if include_relu ~= 0 then
		model:add(nn.ReLU())
	end
	return model
end

function model_utils:seq_sub(num_bilstms, hidden_size, include_relu, disc_size)
	--[[
	(Model structure for sequential subitizing)
	Arguments
	**********
	num_bilstms: number of bi-directional LSTMs to capture context
	hidden_size: number of activations in the hidden layers
	include_relu: whether to include relu after the final fully-connected layer
	disc_size: cell-discretization

	Returns
	**********
	model1: context capturing unit consisting of bi-LSTMs
	model2: cell-level MLPs to regress to cell-counts
	]]
	local model1 = nn.Sequential()
	local model2 = nn.Sequential()
	local c = nn.ConcatTable()
	for i = 1,2 do
		local bilstm = nn.Sequential()
		bilstm:add(nn.SelectTable(i))
		bilstm:add(nn.SplitTable(2))
		bilstm:add(nn.Sequencer(nn.ReLU()))
		bilstm:add(nn.Sequencer(nn.Linear(self.input_dimensions, hidden_size)))
		bilstm:add(nn.Sequencer(nn.ReLU()))
		for j = 1,num_bilstms do
			local lstm_size = hidden_size*torch.pow(2, j-1)
			bilstm:add(nn.BiSequencer(nn.LSTM(lstm_size, lstm_size, disc_size), nn.LSTM(lstm_size, lstm_size, disc_size), nn.JoinTable(1,1)))
		end
		c:add(bilstm)
	end
	model1:add(c)
	model1:add(nn.FlattenTable())
	model2:add(nn.Sequencer(nn.ReLU()))
	model2:add(nn.Sequencer(nn.Linear(hidden_size*torch.pow(2, num_bilstms+1), self.num_classes)))
	if include_relu ~= 0 then
		model2:add(nn.Sequencer(nn.ReLU()))
	end
	return model1, model2
end
