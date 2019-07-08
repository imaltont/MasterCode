require"rnn"
require"optim"
require "cutorch"
require "cunn"

torch.setdefaulttensortype('torch.CudaTensor')

math.randomseed(os.time())

--> Requires at least one hidden layer
function build_network(opts)
    local r = nn.Recurrent(opts.hidden_sizes[1], nn.Linear(opts.input_size, opts.hidden_sizes[1]), nn.Linear(opts.hidden_sizes[1], opts.hidden_sizes[1]), nn.ELU(), opts.num_recurrent)

    local rnn = nn.Sequential()
    rnn:add(r)
    opts.hidden_sizes[0] = opts.hidden_sizes[1]
    --rnn:add(nn.ELU())
    for i = 1, #opts.hidden_sizes do
        rnn:add(nn.Linear(opts.hidden_sizes[i-1], opts.hidden_sizes[i]))
        rnn:add(nn.ELU())
    end

    rnn:add(nn.Linear(opts.hidden_sizes[#opts.hidden_sizes], #opts.actions))

    rnn = nn.Sequencer(rnn)

    rnn:remember("eval")

    local criterion = nn.SequencerCriterion(nn.MSECriterion())
    
    return rnn:cuda(), criterion:cuda()
end

-- rho x batch x input
function get_q_update(state, action, reward, state2, terminal, network, target_network, discount)
    local delta, q, q2, q2_max, targets
    local s = state
    local a = action
    local r = reward
    local s2 = state2
    local network = network
    local target_network = target_network
    local discount = discount

    --Max a q for state 2
    q2_max = target_network:forward(s2):float():max(3)

    --q2
    q2 = q2_max:clone():mul(discount):cmul(terminal)

    delta = r:clone():add(q2)

    --local q_all = network:forward(s)--:clone()

    --targets = torch.zeros(q_all:size(1), q_all:size(2), q_all:size(3)):add(q_all)

    --for i = 1, state:size(1) do
    --    for j = 1, state:size(2) do
    --        targets[i][j][a[i][j]] = delta[i][j] 
    --    end
    --end

    --return targets
    return delta
end

function q_learn_batch(batch, state2, actions, rewards, terminal, learning_rate, discount, network, target_network, criterion, epoch, iteration)
    local targets = get_q_update(batch, actions, rewards, state2, terminal, network, target_network, discount)

    training_step(network, criterion, batch, targets, actions, learning_rate, iteration, epoch)

    return network
end

-- inputs consist of {input1, input2, action, reward}
function generate_batches(inputs, batch_size, num_recurrent)
    local rewards = {}
    local inp_tensor = {}
    local state2 = {}
    local actions = {}
    local terminal = {}
    local terminal_func = function(term) if term then return 1 else return 0 end end
    local last_batch = 1 --math.floor(batch_size / 2)
    for i = 1, num_recurrent do
        rewards[i] = {}
        inp_tensor[i] = {}
        actions[i] = {}
        state2[i] = {}
        terminal[i] = {}
    end
    for i = 1, last_batch do
            rewards[i] = inputs[i].rewards
            inp_tensor[i] = inputs[i].states
            actions[i] = inputs[i].actions
            state2[i] = inputs[i].state2s
            terminal[i] = inputs[i].terminals
    end
    for i = last_batch+1, batch_size do
        local ind = math.random(1, #inputs-num_recurrent)
        local terminated = false
        rewards[i] = inputs[ind].rewards
        inp_tensor[i] = inputs[ind].states
        actions[i] = inputs[ind].actions
        state2[i] = inputs[ind].state2s
        terminal[i] = inputs[ind].terminals
    end
    inp_tensor = torch.cat(inp_tensor, 2):cuda()
    state2 = torch.cat(state2, 2):cuda()
    actions = torch.cat(actions, 2)
    rewards = torch.cat(rewards, 2)
    terminal = torch.cat(terminal, 2)
    return inp_tensor, actions, rewards, state2, terminal
end

--Only considers N x 1 x M
function select_action(output, chance, legal_power)
    local indices = #output
    if chance > math.random() then
        local ind = math.random(1, indices[3])
        while not legal_power[ind] do
            ind = math.random(1, indices[3])
        end
        return ind
    else
        local max_rew = -math.huge
        local max_ind = 1
        local output = output:cuda()
        for i = 1, indices[3] do
            if output[indices[1]][indices[2]][i] > max_rew and (legal_power[i]) then
                max_rew = output[indices[1]][indices[2]][i]
                max_ind = i
            elseif output[indices[1]][indices[2]][i] == max_rew and (legal_power[i]) and 0.5 > math.random() then
                max_rew = output[indices[1]][indices[2]][i]
                max_ind = i
            end
        end
        return max_ind
    end
end

function training_step(network, criterion, inputs, delta, a, learning_rate, iteration, epoch)
    params, gradParams = network:getParameters()
    for i = 1, epoch do
        function feval(params)
            gradParams:zero()
            local outputs = network:forward(inputs)
            local targets = torch.zeros(outputs:size(1), outputs:size(2), outputs:size(3)):add(outputs)

            for i = 1, inputs:size(1) do
                for j = 1, inputs:size(2) do
                    targets[i][j][a[i][j]] = delta[i][j] 
                end
            end

            --outputs = nn.SoftMax()(outputs)
            --targets = nn.SoftMax()(targets)
            local err = criterion:forward(outputs, targets) --torch.sum(targets) 
            local grad_outputs = criterion:backward(outputs, targets)
            local grad_inputs = network:backward(inputs, grad_outputs)
            return err, gradParams
        end
        optim.sgd(feval, params, {learningRate = learning_rate})
    end
end
