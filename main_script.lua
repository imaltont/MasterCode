require"rnn-rl"
require"collect_info"
require"mycbr"
emu.pause()
emu.speedmode("maximum")
opts = {
    actions = {
        "joypad.set(1, {up=false, down = false, left = false, right = false, A = false, B = false, start = false, select = false})",
        "joypad.set(1, {up=false, down = false, left = true, right = false, A = false, B = false, start = false, select = false})",
        "joypad.set(1, {up=false, down = false, left = false, right = true, A = false, B = false, start = false, select = false})",
        "joypad.set(1, {up=false, down = false, left = true, right = false, A = true, B = false, start = false, select = false})",
        "joypad.set(1, {up=false, down = false, left = false, right = true, A = true, B = false, start = false, select = false})",
        "joypad.set(1, {up=false, down = false, left = true, right = false, A = false, B = true, start = false, select = false})",
        "joypad.set(1, {up=false, down = false, left = false, right = true, A = false, B = true, start = false, select = false})",
        "joypad.set(1, {up=false, down = false, left = true, right = false, A = true, B = true, start = false, select = false})",
        "joypad.set(1, {up=false, down = false, left = false, right = true, A = true, B = true, start = false, select = false})",
        "joypad.set(1, {up=false, down = false, left = false, right = false, A = true, B = false, start = false, select = false})",
        "joypad.set(1, {up=false, down = false, left = false, right = false, A = false, B = true, start = false, select = false})",
        "joypad.set(1, {up=false, down = false, left = false, right = false, A = true, B = true, start = false, select = false})",
        "joypad.set(1, {up=true, down = false, left = false, right = false, A = false, B = false, start = false, select = false})",
        "joypad.set(1, {up=false, down = true, left = false, right = false, A = false, B = false, start = false, select = false})" 
    },
    action_repeats = 5,
    learningrate = 0.01, 
    min_learningrate = 0.001, 
    discount = 0.99,
    learn_start = 1000,
    replay_memory = 1000,
    update_frequency = 30,
    n_replay = 1,
    input_size = 1 + 1 + 15*16,
    batch_size = 128,
    num_batch = 7,
    target_q = 20000,
    hidden_sizes = {1000, 1000, 1000},
    num_recurrent = 5,
    completions = 1,
    --steps = 100000,
    steps = 10000,
    random_action = 1.0,
    --min_random_action = 0.3,
    min_random_action = 0.1,
    --min_random_action = 0.0,
    training = false,
    cbr = false,
    save_freq = 1000
}
history_collection = {}
history_stats = {}
visited_cases = {}
global_rew = 0
level_savestate = {}
level_savestate["Bubble"] = 1
level_savestate["Air"] = 2
level_savestate["Quick"] = 3
level_savestate["Wood"] = 4
level_savestate["Crash"] = 5
level_savestate["Flash"] = 6
level_savestate["Metal"] = 7
level_savestate["Heat"] = 8

all_cases = get_all_cases()


local agent = {}

function load_state(level)
    local ss = savestate.object(level_savestate[level])
    savestate.load(ss)
    --set 99 lives
end

function create_agent(opts)
    local agent = {}
    agent.network, agent.criterion = build_network(opts)
    agent.target_network = agent.network:clone()
    agent.target_network:remember("neither")
    return agent
end
best_x_all = 255
since_moved = 0
since_improved = 0
is_stuck = false
function get_reward(prev_life, prev_alive, prev_x, prev_y, prev_boss, prev_screen)
    local reward = 0
    local alive, life, x, y, boss, screen, terminal, x_rew, y_rew, life_rew, screen_fix
    life = get_megaman_hp()
    alive = get_death()
    boss = get_megaman_boss_health()
    screen = get_screen()
    x = get_megaman_x() + (screen*255)
    y = get_megaman_y() 
    terminal = false
    screen_fix = 255 
    x_rew = (x - prev_x + math.max(x - best_x_all, 0)) * 10
    if x == prev_x then
        x_rew = x_rew - 50 * opts.action_repeats 
        since_moved = since_moved + 1
    end
    x_rew = x_rew + math.max(screen - prev_screen, 0) * 255
    y_rew = math.max(y - prev_y, prev_y - y) 
    if x > best_x_all then
        best_x_all = x
        since_moved = 0
        since_improved = 0
        is_stuck = false 
    else
        since_improved = since_improved + 1
    end

    if (prev_alive >= 128 and alive < 128) or life <= 0 then
        terminal = true
        since_moved = 0
        since_improved = 0
        is_stuck = false 
        best_x_all = 255
        life = 28
    end
    if since_moved > 200 or since_improved > 500 then
        is_stuck = true
    end
    if since_improved > 1000 then
        terminal = true
        since_moved = 0
        is_stuck = false 
        since_improved = 0
        best_x_all = 255
        life = 28
    end
    if terminal then
        x_rew = -510
        x = 255
    end
    life_rew = math.min(life - prev_life, 0)/28
    reward = (x_rew)/255

    if get_life() - 1 == 0 then
        best_x_all = 0
        since_moved = 0
        is_stuck = false 
        since_improved = 0
        terminal = true
    end
    if prev_boss > boss then
        reward =reward + math.min(1 * (prev_boss - boss), 50)
        since_moved = 0
        is_stuck = false 
        since_improved = 0
        if boss == 0 then
            terminal = true
            reward =reward + 1020
            best_x_all = 0
        end
    end
    return reward, life, alive, x, y, boss, screen, terminal
end

function get_input(prev_input)
    local inputs = {}
    local map = get_nn_map()
    for i = 1, #map do
        for j = 1, #map[i] do
            inputs[j+ (i-1)*#map[i]] = (map[i][j]) / 210
        end
    end
    inputs[#inputs+1] = 1 - get_megaman_hp() / 28
    inputs[#inputs+1] = get_megaman_boss_health() / 28
    return inputs
end

if not opts.cbr and opts.training then
    agent = create_agent(opts)
    agent.name = "Single_network"
end
if not opts.training then
    if opts.cbr then
        file = io.open("testing_stats_cbrann.csv", "w")
    else
        file = io.open("testing_stats_singleANN.csv", "w")
        agent = torch.load("./" .. "Single_network.th7")
        agent.network:forget()
    end
    io.output(file)
    io.write("step;level;global rew;level reward;game reward;deaths;\n")
    io.close(file)
    
    if opts.cbr then
        file = io.open("testing_stats_cbrann.csv", "a")
    else
        file = io.open("testing_stats_singleANN.csv", "a")
    end
    visited_cases = torch.load("./" .. "visited.th7")
else
    if opts.cbr then
        file = io.open("training_stats_cbrann.csv", "w")
    else
        file = io.open("training_stats_singleANN.csv", "w")
    end
    io.output(file)
    io.write("step;level;global rew;level reward;game reward;deaths;\n")
    io.close(file)
    
    if opts.cbr then
        file = io.open("training_stats_cbrann.csv", "a")
    else
        file = io.open("training_stats_singleANN.csv", "a")
    end
end
io.output(file)
emu.pause()
local cbr_file = io.open("cbr_file.txt", "w")
io.output(cbr_file)
io.write("None")
io.close(cbr_file)
for n_completed = 1, opts.completions do
    if opts.training then
        for i = 1, #all_cases do
            local ind = math.random(1, #all_cases)
            all_cases[i], all_cases[ind] = all_cases[ind], all_cases[i]
        end
    end
    for k, v in pairs(all_cases) do
        load_state(all_cases[k].Level)
        --Retrieve case from case base
        --Update agent according to new case
        local level_file = io.open("level.txt", "w")
        io.output(level_file)
        io.write(all_cases[k].Level)
        io.close(level_file)
        if opts.cbr then
            local has_cases = false
            for k2, v2 in pairs(visited_cases) do
                has_cases = true
                break
            end
            io.output(io.stdout)
            if not has_cases and opts.training then
                agent = create_agent(opts)
                io.write(string.format("Creating case: %s\n", all_cases[k].caseID))
            else
                local case = retrieve_case(all_cases[k], visited_cases)
                agent = torch.load("./" .. case.name .. ".th7")
                io.write(string.format("Loading case: %s\n", case.name))
                local cbr_file = io.open("cbr_file.txt", "w")
                io.output(cbr_file)
                io.write(case.name)
                io.close(cbr_file)
            end
            agent.name = all_cases[k].caseID
        end
        agent.legal_action = get_legal_actions(all_cases[k].Powerup)
        if not visited_cases[all_cases[k].caseID] then
            visited_cases[all_cases[k].caseID] = {name = all_cases[k].caseID, reward = 0}
        end
        local step = 1
        local terminal = false
        local reward = 0
        local local_rew = 0
        local best_x = 255
        local best_y = 0
        local best_screen = 1
        local life = 28 
        local boss_hp = 0
        local alive = 0
        local prev_input = 0
        best_x_all = 0
        since_moved = 0
        since_improved = 0
        is_stuck = false
        history_collection = {}
        while step <= opts.steps do
            local press_start = false
            --if agent is not playing currently
	        if memory.readbyte(GAME_STATE) ~= PLAYING and memory.readbyte(GAME_STATE) ~= BOSS_RUSH then
                emu.unpause()
                press_start = not press_start
                joypad.set(1, {up=false, down = false, left = false, right = false, A = false, B = false, start = press_start, select = false})      
                emu.frameadvance()
            else
                emu.pause()
                local state = get_input(prev_input)

                table.insert(history_collection, 1, {state = state})

                local inp = {}
                local inp_order = 1
                --for i = 1, math.min(opts.num_recurrent, #history_collection) do
                for i = 1, math.min(1, #history_collection) do
                    table.insert(inp, i, {})
                    table.insert(inp[i], 1, history_collection[i].state)
                end

                inp = torch.CudaTensor(inp):cuda()

                local action_chance = 0
                if opts.learn_start == step then
                    io.write("Learning for level started\n")
                end
                if not opts.training then
                    action_chance = opts.min_random_action
                elseif opts.learn_start < step then
                    if is_stuck then
                        action_chance = 1
                    else
                        action_chance = opts.random_action
                        action_chance = math.max(action_chance - (action_chance * (step/opts.steps)), opts.min_random_action)
                    end
                else
                    action_chance = 1
                end

                    
                local action = select_action(agent.network:forward(inp), action_chance, agent.legal_action)

                history_collection[1].action = action
                prev_input = action

                
                if #history_collection > opts.replay_memory then
                    table.remove(history_collection, #history_collection)
                end

                emu.unpause()
                if (action > 14 ) then
                    loadstring(opts.actions[action])()
                end
                for i = 1, opts.action_repeats do
                    if (action <= 14 ) then
                        loadstring(opts.actions[action])()
                    end
                    emu.frameadvance()
                end
                emu.pause()

                local state2 = get_input(prev_input)

                reward, life, alive, best_x, best_y, boss_hp, best_screen, terminal = get_reward(life, alive, best_x, best_y, boss_hp, best_screen)

                visited_cases[all_cases[k].caseID].reward = visited_cases[all_cases[k].caseID].reward + reward
                history_collection[1].reward = reward
                history_collection[1].state2 = state2
                history_collection [1].terminal = terminal

                --if #history_collection >= opts.num_recurrent then
                if #history_collection >= 1 then
                    local state = inp:float()
                    local actions = {}
                    local rewards = {}
                    local state2s = {}
                    local terminals = {}
                    local terminal_func = function(term) if term then return 1 else return 0 end end
                    --for i = 1, math.min(opts.num_recurrent, #history_collection) do
                    for i = 1, math.min(1, #history_collection) do
                        table.insert(actions, i, {})
                        table.insert(actions[i], 1, history_collection[i].action)

                        table.insert(rewards, i, {})
                        table.insert(rewards[i], 1, history_collection[i].reward)

                        table.insert(state2s, i, {})
                        table.insert(state2s[i], 1, history_collection[i].state2)

                        table.insert(terminals, i, {})
                        table.insert(terminals[i], 1, terminal_func(history_collection[i].terminal))
                    end
                    history_collection[1].rewards = torch.FloatTensor(rewards)
                    history_collection[1].state2s = torch.FloatTensor(state2s)
                    history_collection [1].terminals = torch.FloatTensor(terminals)
                    history_collection [1].actions = torch.FloatTensor(actions)
                    history_collection [1].states = inp:clone():float()

                end

                global_rew = global_rew + reward
                local_rew = local_rew + reward
                


                if step % opts.update_frequency == 0 and step > opts.learn_start and opts.training then
                    for i = 1, opts.num_batch do
                        --local batch, actions, rewards, state2, terminal = generate_batches(history_collection, opts.batch_size, opts.num_recurrent)
                        local batch, actions, rewards, state2, terminal = generate_batches(history_collection, opts.batch_size, 1)
                        local learningrate = math.max(opts.learningrate - (opts.learningrate * (step/opts.steps)), opts.min_learningrate)
                        agent.network = q_learn_batch(batch, state2, actions, rewards, terminal, learningrate, opts.discount, agent.network, agent.target_network, agent.criterion, opts.n_replay, step)
                    end
                    collectgarbage()
                end
                if step % 100 == 0 then
                    history_stats[#history_stats+1] = {step = step, local_rew = local_rew, reward = reward, global_rew = global_rew}
                    collectgarbage() 
                end

                if step % opts.target_q == 0 and step > opts.learn_start then
                    agent.target_network = agent.network:clone()
                    agent.target_network:forget()
                    agent.target_network:remember("neither")
                end

                step = step + 1
                if terminal then
                    local_rew = 0
                    load_state(all_cases[k].Level)
                    terminal = false
                end

                io.output(file)
                io.write(string.format("%d;%d;%f;%f;%s;%s\n", step,k, global_rew, local_rew, all_cases[k].caseID, all_cases[k].Level))
                if step % opts.save_freq == 0 and opts.training then
                    torch.save(agent.name .. ".th7", agent)
                    if opts.cbr then
                        torch.save("visited.th7", visited_cases)
                        torch.save("history.th7", history_stats)
                    else
                        torch.save("visited_single.th7", visited_cases)
                        torch.save("history_single.th7", history_stats)
                    end
                end
            end
        end
    end
end
