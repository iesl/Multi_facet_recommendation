import torch
import random

def predict_basis(model_further, n_basis, output_emb, predict_coeff_sum = False):
    #print( output_emb.size() )
    new_bsz = output_emb.size(1)*output_emb.size(0)

    #hidden_init = model_further.init_hidden( new_bsz )

    input_squeezed = output_emb.view( new_bsz , output_emb.size(2))
    input_2nd_RNN = input_squeezed.expand(n_basis, input_squeezed.size(0), input_squeezed.size(1) ) # input_2nd_RNN = input_squeezed.unsqueeze(0).expand(n_basis, input_squeezed.size(0), input_squeezed.size(1) )

    #basis_pred, hidden_no_use =  model_further(input_2nd_RNN, hidden_init)
    #basis_pred, hidden_no_use =  model_further(input_2nd_RNN, input_squeezed)
    if predict_coeff_sum:
        basis_pred, hidden_no_use, coeff_pred =  model_further(input_2nd_RNN, input_squeezed, predict_coeff_sum = True)
    
        #basis_pred should have dimension ( n_basis, n_seq_len*n_batch, n_emb_size)
        #coeff_pred should have dimension ( n_basis, n_seq_len*n_batch, n_emb_size)
        basis_pred = basis_pred.permute(1,0,2)
        coeff_pred = coeff_pred.permute(1,0,2)
        #basis_pred should have dimension ( n_seq_len*n_batch, n_basis, n_emb_size)
        return basis_pred, coeff_pred
    else:
        basis_pred, hidden_no_use =  model_further(input_2nd_RNN, input_squeezed, predict_coeff_sum = False)
        basis_pred = basis_pred.permute(1,0,2)
        return basis_pred

def estimate_coeff_mat_batch_max_iter(target_embeddings, basis_pred, device):
    batch_size = target_embeddings.size(0)
    #A = basis_pred.permute(0,2,1)
    C = target_embeddings.permute(0,2,1)
    #basis_pred_norm = basis_pred / (0.000000000001 + basis_pred.norm(dim = 2, keepdim=True) )
    
    basis_pred_norm = basis_pred.norm(dim = 2, keepdim=True)
    #basis_pred_norm_sq = basis_pred_norm * basis_pred_norm
    XX = basis_pred_norm * basis_pred_norm
    n_not_sparse = 2
    coeff_mat_trans = torch.zeros(batch_size, basis_pred.size(1), target_embeddings.size(1), requires_grad= False, device=device )
    for i in range(n_not_sparse):
        XY = torch.bmm(basis_pred, C)
        coeff = XY / XX
        #coeff should have dimension ( n_seq_len*n_batch, n_basis, n_further)
        max_v, max_i = torch.max(coeff, dim = 1, keepdim=True)
        max_v[max_v<0] = 0
    
        coeff_mat_trans_temp = torch.zeros(batch_size, basis_pred.size(1), target_embeddings.size(1), requires_grad= False, device=device )
        coeff_mat_trans_temp.scatter_(dim=1, index = max_i, src = max_v)
        coeff_mat_trans.scatter_add_(dim=1, index = max_i, src = max_v)
        #pred_emb = torch.bmm(coeff_mat_trans_temp.permute(0,2,1),basis_pred)
        #C = C - pred_emb
        pred_emb = torch.bmm(coeff_mat_trans.permute(0,2,1),basis_pred)
        C = (target_embeddings - pred_emb).permute(0,2,1)
        
    #pred_emb = max_v * torch.gather(basis_pred,  max_i
    
    return coeff_mat_trans.permute(0,2,1)
    #torch.gather(coeff_mat_trans , dim=1, index = max_i)


def estimate_coeff_mat_batch_max(target_embeddings, basis_pred, device):
    batch_size = target_embeddings.size(0)
    #A = basis_pred.permute(0,2,1)
    C = target_embeddings.permute(0,2,1)
    #basis_pred_norm = basis_pred / (0.000000000001 + basis_pred.norm(dim = 2, keepdim=True) )
    
    basis_pred_norm = basis_pred.norm(dim = 2, keepdim=True)
    #basis_pred_norm_sq = basis_pred_norm * basis_pred_norm
    XX = basis_pred_norm * basis_pred_norm
    XY = torch.bmm(basis_pred, C)
    coeff = XY / XX
    #coeff should have dimension ( n_seq_len*n_batch, n_basis, n_further)
    max_v, max_i = torch.max(coeff, dim = 1, keepdim=True)
    max_v[max_v<0] = 0
    
    coeff_mat_trans = torch.zeros(batch_size, basis_pred.size(1), target_embeddings.size(1), requires_grad= False, device=device )
    coeff_mat_trans.scatter_(dim=1, index = max_i, src = max_v)
    return coeff_mat_trans.permute(0,2,1)
    #torch.gather(coeff_mat_trans , dim=1, index = max_i)



def estimate_coeff_mat_batch(target_embeddings, basis_pred, L1_losss_B, device):
    def compute_matrix_magnitude(M_diff):
        return torch.mean( torch.abs(M_diff) )

    #B_raw = B_in - lr * ( np.dot(AT,np.dot(A,B_in)-C) + reg
    #np.maximum( B_raw, np.zeros( (num_cluster,num_sent) ) )

    def update_B_from_AC(AT,BT,CT,A,lr):
        #lr = 0.05
        BT_grad = torch.bmm( torch.bmm(BT, AT) - CT, A )
        #BT = BT - lr * (BT_grad)
        #return BT, BT_grad
        BT = BT - lr * (BT_grad + L1_losss_B)

        #BT_nonneg = torch.max( torch.tensor([0.0], device=device), BT )
        #BT_nonneg = torch.min( torch.tensor([2.0], device=device), BT_nonneg )
        #BT_nonneg = torch.min( torch.tensor([1.0], device=device), BT_nonneg )
        BT_nonneg = BT.clamp(0,1)
        return BT_nonneg, BT_grad

    batch_size = target_embeddings.size(0)
    #converge_threshold = 0.01

    A = basis_pred.permute(0,2,1)
    #coeff_mat_prev = torch.abs(torch.randn(batch_size, target_embeddings.size(1), basis_pred.size(1), requires_grad= False, device=device ))
    coeff_mat_prev = torch.randn(batch_size, target_embeddings.size(1), basis_pred.size(1), requires_grad= False, device=device )
    #coeff_mat = update_B_from_AC(basis_pred, coeff_mat_prev, target_embeddings, A)
    #coeff_mat_prev = torch.zeros_like(coeff_mat)
    #max_iter = 50
    #max_iter = 150
    max_iter = 100
    #diff_prev = 10
    lr = 0.05
    #lr = 0.02
    #lr = 0.1
    for i in range(max_iter):
        coeff_mat, coeff_mat_grad = update_B_from_AC(basis_pred, coeff_mat_prev, target_embeddings, A, lr)
        ##diff = compute_matrix_magnitude(coeff_mat - coeff_mat_prev)

        #diff = compute_matrix_magnitude(coeff_mat_grad)

        #print(diff)
        #if diff > diff_prev:
        #    lr *= 0.9
        #if diff < converge_threshold:
        #    break
        #diff_prev = diff
        coeff_mat_prev = coeff_mat
    #print(diff,lr,i)
    #print(i)
    #coeff_mat should have dimension (n_seq_len*n_batch,n_further,n_basis)
    return coeff_mat

def target_emb_preparation(target_index, w_embeddings, n_batch, n_further, rotate_shift):
    target_embeddings = w_embeddings[target_index,:]
    #print( target_embeddings.size() )
    #target_embeddings should have dimension (n_seq_len*n_batch, n_further, n_emb_size)
    #should be the same as w_embeddings.select(0,target_further) and select should not copy the data
    #print w_embeddings[0,:]
    target_embeddings = target_embeddings / (0.000000000001 + target_embeddings.norm(dim = 2, keepdim=True) ) # If this step is really slow, consider to do normalization before doing unfold
    
    target_embeddings_4d = target_embeddings.view(-1,n_batch, n_further, target_embeddings.size(2))
    #half_batch_size = int(n_batch/2)
    target_embeddings_rotate = torch.cat( (target_embeddings_4d[:,rotate_shift:,:,:], target_embeddings_4d[:,:rotate_shift,:,:]), dim = 1)
    target_emb_neg = target_embeddings_rotate.view(-1,n_further, target_embeddings.size(2))

    return target_embeddings, target_emb_neg

def compute_loss_further(output_emb, model_further, w_embeddings, target_further, n_basis, n_further, L1_losss_B, device, w_freq, coeff_opt, compute_target_grad):

    basis_pred, coeff_pred = predict_basis(model_further, n_basis, output_emb, predict_coeff_sum = True)
    #basis_pred should have dimension ( n_seq_len*n_batch, n_basis, n_emb_size)
    #basis_pred = basis_pred.cpu()
    #print( basis_pred.size() )

    #print( target_further.size() )
    #target_further should have dimension (n_seq_len + n_further - 1, n_batch)
    target_unfold = target_further.unfold(0,n_further,1)#.permute(2,1,0)
    #print( target_unfold.size() )
    #target_unfoled should have dimension (n_further, n_batch, n_seq_len) -> (n_seq_len, n_batch, n_further)
    target_index = target_unfold.contiguous().view(-1,target_unfold.size(2))
    #target_unfoled should have dimension (n_seq_len*n_batch, n_further)
    #print( target_index.size() )

    n_batch = target_further.size(1)
    rotate_shift = random.randint(1,n_batch-1)
    if compute_target_grad:
        target_embeddings, target_emb_neg = target_emb_preparation(target_index, w_embeddings, n_batch, n_further, rotate_shift)
    else:
        with torch.no_grad():
            target_embeddings, target_emb_neg = target_emb_preparation(target_index, w_embeddings, n_batch, n_further, rotate_shift)

    with torch.no_grad():
        target_freq = w_freq[target_index]
        #target_freq = torch.masked_select( target_freq, target_freq.gt(0))
        target_freq_inv = 1 / target_freq
        target_freq_inv[target_freq_inv<0] = 0
        inv_mean = torch.sum(target_freq_inv) / torch.sum(target_freq_inv>0).float()
        if inv_mean > 0:
            target_freq_inv_norm =  target_freq_inv / inv_mean
        else:
            target_freq_inv_norm =  target_freq_inv
        
        #half_batch_size = int(target_further.size(1)/2)
        target_freq_inv_norm_3d = target_freq_inv_norm.view(-1, target_further.size(1), target_freq_inv_norm.size(1))
        target_freq_inv_norm_rotate = torch.cat( (target_freq_inv_norm_3d[:,rotate_shift:,:], target_freq_inv_norm_3d[:,:rotate_shift,:]), dim = 1)
        target_freq_inv_norm_neg = target_freq_inv_norm_rotate.view(-1,target_freq_inv_norm.size(1))
        
        #target_embeddings = target_embeddings * target_freq_inv_norm.unsqueeze(dim = 2)
        
        #coeff_mat = estimate_coeff_mat_batch(target_embeddings.cpu(), basis_pred.detach(), L1_losss_B)
        if coeff_opt == 'lc':
            coeff_mat = estimate_coeff_mat_batch(target_embeddings, basis_pred.detach(), L1_losss_B, device)
            coeff_mat_neg = estimate_coeff_mat_batch(target_emb_neg, basis_pred.detach(), L1_losss_B, device)
        else:
            coeff_mat = estimate_coeff_mat_batch_max(target_embeddings, basis_pred.detach(), device)
            coeff_mat_neg = estimate_coeff_mat_batch_max(target_emb_neg, basis_pred.detach(), device)
        #coeff_mat = estimate_coeff_mat_batch_max_iter(target_embeddings, basis_pred.detach(), device)
        coeff_sum_basis = coeff_mat.sum(dim = 1)
        coeff_sum_basis_neg = coeff_mat_neg.sum(dim = 1)
        coeff_mean = (coeff_sum_basis.mean() + coeff_sum_basis_neg.mean()) / 2
        #coeff_sum_basis should have dimension (n_seq_len*n_batch,n_basis)
    
    pred_embeddings = torch.bmm(coeff_mat, basis_pred)
    pred_embeddings_neg = torch.bmm(coeff_mat_neg, basis_pred)
    #print target_index
    #print target_embeddings
    #print pred_embeddings
    #pred_embeddings = torch.bmm(coeff_mat.cuda(), basis_pred.cuda())
    #pred_embeddings should have dimension (n_seq_len*n_batch, n_further, n_emb_size)
    #loss_further = torch.mean( target_freq_inv_norm * torch.norm( pred_embeddings.cuda() - target_embeddings, dim = 2 ) )
    loss_further = torch.mean( target_freq_inv_norm * torch.pow( torch.norm( pred_embeddings - target_embeddings, dim = 2 ), 2) )
    loss_further_neg = - torch.mean( target_freq_inv_norm_neg * torch.pow( torch.norm( pred_embeddings_neg - target_emb_neg, dim = 2 ), 2) )
    loss_coeff_pred = torch.mean( torch.pow( coeff_sum_basis/coeff_mean - coeff_pred[:,:,0].view_as(coeff_sum_basis), 2 ) )
    loss_coeff_pred += torch.mean( torch.pow( coeff_sum_basis_neg/coeff_mean - coeff_pred[:,:,1].view_as(coeff_sum_basis_neg), 2 ) )
    
    with torch.no_grad():
        basis_pred_norm = basis_pred / basis_pred.norm(dim = 2, keepdim=True)
        pred_mean = basis_pred_norm.mean(dim = 0, keepdim = True)
        loss_further_reg = - torch.mean( (basis_pred_norm - pred_mean).norm(dim = 2) )
    
        pred_mean = basis_pred_norm.mean(dim = 1, keepdim = True)
        loss_further_div = - torch.mean( (basis_pred_norm - pred_mean).norm(dim = 2) )



    return loss_further, loss_further_reg, loss_further_div, loss_further_neg, loss_coeff_pred
