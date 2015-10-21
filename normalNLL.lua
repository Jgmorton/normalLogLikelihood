require 'nn'
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

-- Computes log of sum of Gaussian components
local function logSum(input, target, n)
    local sum = torch.zeros(input:size(1))
        for i = 1, n do
            local w = input[{{}, i}]
            local mu = input[{{}, n + i}]
            local s = input[{{}, 2*n + i}]
            sum = sum + w * normal(target, mu, s)
        end
    return -torch.log(sum)
end

-- Returns loss
function normalNLL:updateOutput(input, target)
    -- Single normal distribution
    if self.n == 1 and input:size(2) == 2 then
        local mu = input[{{}, 1}] -- mean
        local s = input[{{}, 2}] -- std dev
        self.output = -torch.log(normal(target, mu, s))

    elseif self.n > 1 and input:size(2) == 3*self.n then
        self.output = logSum(input, target, self.n)
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
        -- Store constant value
        local a = logSum(input, target, self.n)
        for i = 1, self.n do
            local w = input[{{}, i}]
            local mu = input[{{}, self.n + i}]
            local s = input[{{}, 2*self.n + i}]

            -- Calculate gradientsno
            self.gradInput[{{}, i}] = -torch.cdiv(normal(target, mu, s), a)
            self.gradInput[{{}, self.n + i}] = torch.cdiv(w * (normal(target, mu, s) * grad_mu(target, mu, s)), a)
            self.gradInput[{{}, 2*self.n + i}] = torch.cdiv(w * (normal(target, mu, s) * grad_s(target, mu, s)), a)
        end
    end
    return self.gradInput
end
