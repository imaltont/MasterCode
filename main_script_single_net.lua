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
    learningrate = 0.1, 
    min_learningrate = 0.01, 
    discount = 1.0,
    learn_start = 1000,
    replay_memory = 1000,
    update_frequency = 30,
    n_replay = 1,
    input_size = 1 + 1 + 15*16,
    batch_size = 128,
    num_batch = 2,
    target_q = 10000,
    hidden_sizes = {1000, 1000, 1000},
    num_recurrent = 5,
    completions = 1,
    steps = 100000,
    random_action = 0.7,
    min_random_action = 0.1,
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
end

function create_agent(opts)
    local agent = {}
    agent.network, agent.criterion = build_network(opts)
    agent.target_network = agent.network:clone()
    return agent
end
best_x_all = 0
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
    x_rew = x - prev_x + math.max(x - best_x_all, 0)
    if x == prev_x then
        x_rew = x_rew - 10
    end
    x_rew = x_rew + math.max(screen - prev_screen, 0) * 510
    y_rew = math.max(y - prev_y, prev_y - y) 
    if x > best_x_all then
        best_x_all = x
        since_moved = 0
        since_improved = 0
        is_stuck = false 
    else
        since_improved = since_improved + 1
    end

    if x == prev_x then
        since_moved = since_moved + 1
    end

    if (prev_alive >= 128 and alive < 128) or life <= 0 then
        terminal = true
        since_moved = 0
        since_improved = 0
        is_stuck = false 
        best_x_all = 0
        life = 0
    end
    if since_moved > 200 or since_improved > 500 then
        is_stuck = true
    end
    if since_improved > 1000 then
        terminal = true
        since_moved = 0
        is_stuck = false 
        since_improved = 0
        best_x_all = 0
        life = 0
    end
    if terminal then
        x_rew = -510
        x = 0
    end
    life_rew = math.min(life - prev_life, 0)/28
    reward = (x_rew)/100

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
            inputs[j+ (i-1)*#map[i]] = (map[i][j]) / (7+6)
        end
    end
    inputs[#inputs+1] = get_megaman_hp() / 28
    inputs[#inputs+1] = get_megaman_boss_health() / 28
    return inputs
end

agent = create_agent(opts)
agent.name = "Single_network"
local ordered_cases = {}
for i = 1, #all_cases do
    if all_cases[i].Level == "Wood" then
        ordered_cases[1] = all_cases[i]
    elseif all_cases[i].Level == "Flash" then
        ordered_cases[2] = all_cases[i]
    elseif all_cases[i].Level == "Quick" then
        ordered_cases[3] = all_cases[i]
    elseif all_cases[i].Level == "Metal" then
        ordered_cases[4] = all_cases[i]
    elseif all_cases[i].Level == "Air" then
        ordered_cases[5] = all_cases[i]
    end
end
all_cases = ordered_cases
for n_completed = 1, opts.completions do
    for k, v in pairs(all_cases) do
        load_state(all_cases[k].Level)
        agent.legal_action = get_legal_actions(all_cases[k].Powerup)
        if not visited_cases[all_cases[k].caseID] then
            visited_cases[all_cases[k].caseID] = {name = all_cases[k].caseID, reward = 0}
        end
        local step = 1
        local terminal = false
        local reward = 0
        local local_rew = 0
        local best_x = 0
        local best_y = 0
        local best_screen = 0
        local life = 0 
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
            --if agent is not playing
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
                for i = 1, math.min(opts.num_recurrent, #history_collection) do
                    table.insert(inp, i, {})
                    table.insert(inp[i], 1, history_collection[i].state)
                end

                inp = torch.CudaTensor(inp):cuda()

                local action_chance = 0
                if opts.learn_start == step then
                    io.write("Learning for level started\n")
                end
                if opts.learn_start < step then
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

                global_rew = global_rew + reward
                local_rew = local_rew + reward
                
                if step % opts.update_frequency == 0 and step > opts.learn_start then
                    for i = 1, opts.num_batch do
                        local batch, actions, rewards, state2, terminal = generate_batches(history_collection, opts.batch_size, opts.num_recurrent)
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
                end

                step = step + 1
                if terminal then
                    local_rew = 0
                    load_state(all_cases[k].Level)
                    terminal = false
                end

                if step % opts.save_freq == 0 then
                    torch.save(agent.name .. ".th7", agent)
                    torch.save("visited_single.th7", visited_cases)
                    torch.save("history_single.th7", history_stats)

                    io.flush()
                    io.write(string.format("%d;%d;%f;%f;%s;%s\n", step,k, global_rew, local_rew, all_cases[k].caseID, all_cases[k].Level))
                end
            end
        end
    end
end
