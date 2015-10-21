local normalNLL, parent = torch.class('normalNLL', 'nn.Criterion')

function normalNLL:__init(n)
    parent.__init(self)
    n = n or 1
    self.n = n
end

-- Compute pdf at target for normal dist w/ mean mu and std dev s
local function normal(target, mu, s)
    local arg = torch.cdiv(-torch.pow(target - mu, 2), torch.pow(s, 2)*2)
    local exparg = torch.exp(arg)
    return torch.div(torch.cdiv(exparg, s), math.sqrt(2*math.pi))
end

-- Gradient wrt mu for normal dist
local function grad_mu(target, mu, s)
    return torch.cdiv(mu - target, torch.pow(s, 2))
end

-- Gradient wrt sigma for normal dist
local function grad_s(target, mu, s)
    local g1 = -torch.cdiv(torch.pow(target - mu, 2), torch.pow(s, 3))
    local g2 = torch.cdiv(torch.ones(s:size()), s)
    return g1 + g2
end

-- Returns loss
function normalNLL:updateOutput(input, target)
    -- Single normal distribution
    if self.n == 1 and input:size(2) == 2 then
        local mu = input[{{}, 1}] -- mean
        local s = input[{{}, 2}] -- std dev
        self.output = -torch.log(normal(target, mu, s))

    elseif self.n > 1 and input:size(2) == 3*self.n then
        local sum = torch.zeros(input:size(1))
        for i = 1, n do
            local w = input[{{}, 3*(i-1) + 1}]
            local mu = input[{{}, 3*(i-1) + 2}]
            local s = input[{{}, 3*i}]
            sum = sum + w * normal(target, mu, s)
        end
        self.output = -torch.log(sum)
    else
        error('Invalid number of inputs')
    end
    return self.output
end

-- Returns gradients
function normalNLL:updateGradInput(input, target)
    self.gradInput:resizeAs(input)
    self.gradInput:zero()

    if self.n == 1 then
        local mu = input[{{}, 1}] -- mean
        local s = input[{{}, 2}] -- std dev

        -- Gradient wrt mu
        self.gradInput[{{}, 1}] = grad_mu(target, mu, s)

        -- Gradient wrt sigma
        self.gradInput[{{}, 2}] = grad_s(target, mu, s)

    else 
        for i = 1, n do
            local w = input[{{}, 3*(i-1) + 1}]
            local mu = input[{{}, 3*(i-1) + 2}]
            local s = input[{{}, 3*i}]

            -- Calculate gradients
            self.gradInput[{{}, 3*(i-1) + 1}] = -cdiv(normal(target, mu, s), self.output)
            self.gradInput[{{}, 3*(i-1) + 2}] = cdiv(w * normal(target, mu, s) * grad_mu(target, mu, s), self.output)
            self.gradInput[{{}, 3*i}] = cdiv(w * normal(target, mu, s) * grad_s(target, mu, s), self.output)
        end
    end
    return self.gradInput
end
