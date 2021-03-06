local BinActiveZ , parent= torch.class('nn.BinActiveZ', 'nn.Module')
local TriActiveZ, parent= torch.class('nn.TriActiveZ', 'nn.Module')


function BinActiveZ:updateOutput(input)
	local s = input:size()
   self.output:resizeAs(input):copy(input)
   self.output=self.output:sign();
   return self.output
end

function BinActiveZ:updateGradInput(input, gradOutput)
   local s = input:size()
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput[input:ge(1)]=0
   self.gradInput[input:le(-1)]=0
   return self.gradInput
end

function TriActiveZ:updateOutput(input)
	local s = input:size()
		self.output:resizeAs(input):copy(input)
		local smallOutput = torch.CudaTensor(s):resizeAs(input):copy(input)
		smallOutput[input:ge(0.1)]=1
		smallOutput[input:lt(0.1)]=0
		self.output[input:le(-0.1)]=-1
		self.output[input:gt(-0.1)]=0
		self.output:long():cbitor(smallOutput:long())
		return self.output:cuda()
end

function TriActiveZ:updateGradInput(input, gradOutput)
   local s = input:size()
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput[input:ge(1)]=0
   self.gradInput[input:le(-1)]=0
   return self.gradInput
end
