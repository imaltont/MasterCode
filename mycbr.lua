http = require"socket.http"
json = require"json"
url = "http://localhost:8080"

concept = "Megaman"
casebase = "MegamanCB_Test"
amal = "MegamanSim"

function retrieve_case(case_name, visited_cases)
    if visited_cases[case_name.caseID] then
        return visited_cases[case_name.caseID]
    end
    local retrievals = retrieve_cases(case_name.caseID)
    local best_sim = -math.huge
    local best_reward  = -math.huge
    local best_case = {}
    for k, v in pairs(retrievals) do
        if visited_cases[retrievals[k].caseID] then
            if tonumber(retrievals[k].similarity) > best_sim or (tonumber(retrievals[k].similarity) == best_sim and visited_cases[retrievals[k].caseID].reward) then
                best_sim = tonumber(retrievals[k].similarity)
                best_reward = visited_cases[retrievals[k].caseID].reward
                best_case.name = retrievals[k].caseID
            end
        end
    end
    best_case.reward = 0
    return best_case
end

function get_legal_actions(powers)
    local pow = {"Air", "Bubble", "Quick", "Flash", "Wood", "Metal", "Heat", "Crash"}
    local legal_powers = {true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false,false, false, false, false, false, false, false}
    if string.find(powers, pow[1]) then
        legal_powers[15+2] = true
        legal_powers[15+10] = true
    end
    if string.find(powers, pow[2]) then
        legal_powers[15+4] = true
    end
    if string.find(powers, pow[3]) then
        legal_powers[15+5] = true
    end
    if string.find(powers, pow[4]) then
        legal_powers[15+6] = true
        legal_powers[15+11] = true
    end
    if string.find(powers, pow[5]) then
        legal_powers[15+3] = true
    end
    if string.find(powers, pow[6]) then
        legal_powers[15+7] = true
    end
    if string.find(powers, pow[7]) then
        legal_powers[15+1] = true
        legal_powers[15+9] = true
    end
    if string.find(powers, pow[8]) then
        legal_powers[15+8] = true
    end
    return legal_powers
end

function get_all_cases()
    local url = url .. "/concepts/" .. concept .. "/casebases/" .. casebase .. "/instances"
    local retrievals = {}
    http.request{url = url, method = "GET", headers={["content-type"] = "application/json"}, sink=ltn12.sink.table(retrievals)}
    local rets = ""
    for i = 1, #retrievals do
        rets = rets .. retrievals[i]
    end
    rets = json.decode(rets)
    return rets
end
function retrieve_cases(case_name)
    local url = url .. "/concepts/" .. concept .. "/casebases/" .. casebase .. "/retrievalByIDWithContent?amalgamation%20function=" .. amal .. "&caseID=" .. case_name  .. "&k=-1"
    local retrievals = {}
    http.request{url = url, method = "GET", headers={["content-type"] = "application/json"}, sink=ltn12.sink.table(retrievals)}
    local rets = ""
    for i = 1, #retrievals do
        rets = rets .. retrievals[i]
    end
    rets = json.decode(rets)
    return rets
end
