#! /usr/bin/env lua
--[[
Evaluation utilities for counting
1. The script takes in raw counts from a model, thresholds them to 0 and rounds them
2. Performance is computed according to different metrics
]]

local eval = torch.class('eval_count')

function eval:__init(relu, round)
	--[[
	Initialize raw count preprocessing flags
	Arguments
	**********
	relu: whether to threshold the counts at 0
	round: whether to round the counts
	]]
	self.relu_count = relu or 1
	self.round_count = round or 1
end

function eval:__prepro(count_pred)
	--[[
	(Preprocess the raw counts obtained from the models)
	Arguments
	**********
	count_pred: the raw count predictions from the model

	Returns
	**********
	counts: the preprocessed count tensor
	]]
	local counts = count_pred:clone()
	if self.relu_count == 1 then
		counts = counts:cmax(0)
	end
	if self.round_count == 1 then
		counts = torch.round(counts)
	end
	return counts
end

function eval:sampled_eval(metric, non_zero, count_pred, count_gt, trials)
	--[[
	(Sampled evaluation to reflect confidence intervals)
	Arguments
	**********
	metric: which metric to evaluate on (mRMSE/rel-mRMSE)
	non_zero: whether to evaluate on non-zero/all counts
	count_pred: the raw count predictions from the model
	count_gt: the ground truth counts from the dataset
	trials: #sampled_counts

	Returns
	**********
	res:mean(): mRMSE
	res:std(): mRMSE - standard deviation
	]]
	local trials = trials or 10
	local splits = 0.8	
	local res = torch.Tensor(trials)
	for i = 1,trials do
		pm = torch.multinomial(torch.range(1, count_pred:size(1)), splits * count_pred:size(1), true)
		pred_trial = count_pred:index(1, pm)
		gt_trial = count_gt:index(1, pm)
		if metric == 'mrmse' then
			res[i] = eval:mrmse(non_zero, pred_trial, gt_trial)
		elseif metric == 'rel_mrmse' then
			res[i] = eval:rel_mrmse(non_zero, pred_trial, gt_trial)
		elseif metric == 'mse' then
			res[i] = eval:mse(non_zero, pred_trial, gt_trial)
		end
	end
	return res:mean(), res:std()
end

function eval:mse(non_zero, count_pred, count_gt)
	--[[
	(Mean Squared Error)
	Arguments
	**********
	non_zero: whether to evaluate on non-zero/all_counts
	count_pred: the raw count predictions from the model
	count_gt: the ground truth counts from the dataset

	Returns
	**********
	mse: MSE
	]]
	count_pred = count_pred:double()
	count_gt = count_gt:double()
	local nzero_mask = count_gt:clone()
	nzero_mask = nzero_mask:fill(1)
	if non_zero == 1 then
		nzero_mask = nzero_mask:fill(0)
		nzero_mask[count_gt:ne(0)] = 1
	end
	local mse = torch.pow(targets - count_pred, 2)
	mse = torch.cmul(mse, nzero_mask)
	mse = mse:mean()
	nzero = nzero_mask:mean()
	mse = mse/nzero
	return mse
 end

function eval:mrmse(non_zero, count_pred, count_gt)
	--[[
	(Mean Root Mean Squared Error)
	Arguments
	**********
	non_zero: whether to evaluate on non-zero/all_counts
	count_pred: the raw count predictions from the model
	count_gt: the ground truth counts from the dataset

	Returns
	**********
	mrmse: mRMSE
	]]
	count_pred = count_pred:double()
	count_gt = count_gt:double()
	local nzero_mask = count_gt:clone()
	nzero_mask = nzero_mask:fill(1)
	if non_zero == 1 then
		nzero_mask = nzero_mask:fill(0)
		nzero_mask[count_gt:ne(0)] = 1
	end
	count_pred = self:_prepro(count_pred)
	local mrmse = torch.pow(count_pred - count_gt, 2)
	mrmse = torch.cmul(mrmse, nzero_mask)
	mrmse = torch.mean(mrmse, 1)
	nzero = torch.mean(nzero_mask, 1)
	mrmse = torch.cdiv(mrmse, nzero)
	mrmse = torch.sqrt(mrmse)
	mrmse = torch.mean(mrmse)
	return mrmse
end

function eval:rel_mrmse(non_zero, count_pred, count_gt)
	--[[
	(Relative Mean Root Mean Squared Error)
	Arguments
	**********
	non_zero: whether to evaluate on non-zero/all_counts
	count_pred: the raw count predictions from the model
	count_gt: the ground truth counts from the dataset

	Returns
	**********
	rel_mrmse: relative mRMSE
	]]
	count_pred = count_pred:double()
	count_gt = count_gt:double()
	count_pred = self:_prepro(count_pred)
	local nzero_mask = count_gt:clone()
	nzero_mask = nzero_mask:fill(1)
	if non_zero == 1 then
		nzero_mask = nzero_mask:fill(0)
		nzero_mask[targets:ne(0)] = 1
	end
	local num = torch.pow(count_pred - count_gt, 2)
	local denom = count_gt:clone()
	denom = denom:add(1)
	local rel_mrmse = torch.cdiv(num, denom)
	rel_mrmse = torch.cmul(rel_mrmse, nzero_mask)
	rel_mrmse = torch.mean(rel_mrmse, 1)
	nzero = torch.mean(nzero_mask, 1)
	rel_mrmse = torch.cdiv(rel_mrmse, nzero)
	rel_mrmse = torch.sqrt(rel_mrmse)
	rel_mrmse = torch.mean(rel_mrmse)
	return rel_mrmse
end