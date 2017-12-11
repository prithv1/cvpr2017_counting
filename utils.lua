require 'nn'

local cjson = require 'cjson'
local utils = {}

function utils.num_keys(t) 
	local num_keys = 0 

	for key, value in pairs(t) do
		num_keys = num_keys + 1
	end
	return num_keys
end

-- utility function to get associative subitizing predictions
function utils.get_aso_preds(frac_counts, disc)
	-- frac_counts: a tensor with 'disc' contiguous rows indicating
	-- image counts
	-- disc: discretization of the original image

	assert(frac_counts:size(1) % (disc * disc) == 0)
	
	frac_prds = frac_counts:clone()
	-- threshold at 0
	frac_prds:cmax(0)

	-- split
	frac_prds = frac_prds:split(disc*disc, 1)

	-- sum counts
	for it = 1, #frac_prds do
		frac_prds[it] = frac_prds[it]:sum(1)
	end

	frac_prds = utils.flatten_table(frac_prds, 1)

	return frac_prds
end

function utils.flatten_table(seq, dim)
	-- flatten a table of tensors into a single tensor 
	-- joined along dimension `dim`
	assert(type(seq) == 'table')
	assert(#seq > 0)

	join = nn.JoinTable(dim)

	if seq[1]:type() == 'torch.CudaTensor' then
		join:cuda()
	end

	joined_seq = join:forward(seq)
	return joined_seq
end
function utils.caffe_to_torch(model, location)
	local location = location or {1}
	local m = model
	for i =1 , #location do
		m = m:get(location[i])
	end

	local weight = m.weight:double()
	local weight_clone = weight:clone()
	local nchannels = weight:size(2)

	for i = 1, nchannels do
		weight:select(2, i):copy(weight_clone:select(2, nchannels+1-i))
	end
	weight:mul(255)
end

function utils.table_mean(tab)
	local sum = 0
	local size = 0

	for key, value in ipairs(tab) do
		sum = sum + value
		size = size + 1
	end
	return sum/size
end

-- Assume required if default_value is nil
function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end
  local v = opt[key]
  if v == nil then v = default_value end
  return v
end

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

-- dicts is a list of tables of k:v pairs, create a single
-- k:v table that has the mean of the v's for each k
-- assumes that all dicts have same keys always
function utils.dict_average(dicts)
  local dict = {}
  local n = 0
  for i,d in pairs(dicts) do
    for k,v in pairs(d) do
      if dict[k] == nil then dict[k] = 0 end
      dict[k] = dict[k] + v
    end
    n=n+1
  end
  for k,v in pairs(dict) do
    dict[k] = dict[k] / n -- produce the average
  end
  return dict
end

-- seriously this is kind of ridiculous
function utils.count_keys(t)
  local n = 0
  for k,v in pairs(t) do
    n = n + 1
  end
  return n
end

-- return average of all values in a table...
function utils.average_values(t)
  local n = 0
  local vsum = 0
  for k,v in pairs(t) do
    vsum = vsum + v
    n = n + 1
  end
  return vsum / n
end

return utils
