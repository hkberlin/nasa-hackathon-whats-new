import torch

def R2_score(outputs, y):
	y_bar = torch.sum(torch.mean(y))
	SS_tot = torch.sum((y - y_bar)**2)
	SS_res = torch.sum((y - outputs)**2)
	R2 = 1 - SS_res / SS_tot
	return R2

def cls_accuracy(outputs, y):
	outputs = torch.where(outputs>0.5, 1, 0)
	accuracy = torch.count_nonzero(outputs==y) / torch.numel(y)
	return accuracy

def cls_metrics(outputs, y):
	outputs = torch.where(outputs>0.5, 1, 0)
	TP_mask = torch.logical_and(outputs==y, outputs==1)
	FP_mask = torch.logical_and(outputs!=y, outputs==1)
	TN_mask = torch.logical_and(outputs==y, outputs==0)
	FN_mask = torch.logical_and(outputs!=y, outputs==0)
	TP = torch.count_nonzero(outputs[TP_mask])
	FP = torch.count_nonzero(outputs[FP_mask])
	TN = torch.count_nonzero(outputs[TN_mask])
	FN = torch.count_nonzero(outputs[FN_mask])
	precision = TP/(TP+FP)
	recall = TP/(TP+FN)
	return precision, recall

def loss_balancing(outputs, y):
	mask_storm = torch.nonzero(y==1)
	num_to_sample = torch.count_nonzero(mask_storm)

	outputs_storm = outputs[mask_storm]
	outputs_peace = outputs[~mask_storm]
	y_storm = y[mask_storm]
	y_peace = y[~mask_storm]

	idx_peace = torch.randperm(outputs.size(0))[:num_to_sample]
	# idx_peace = torch.randint(0, outputs_peace.size(1), size=(num_to_sample,))
	outputs_peace = outputs_peace[idx_peace]
	y_peace = y_peace[idx_peace]

	outputs = torch.cat([outputs_storm, outputs_peace], dim=0)
	y = torch.cat([y_storm, y_peace], dim=0)

	return outputs, y

