import torch
import torch.nn as nn


class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=None, task_prompt_size=None,pruning=False, num_classes=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.pruning = pruning
        self.task_prompt_size = task_prompt_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        self.num_classes = num_classes

        # todo Dynamic Prompt Pool (task share:2, task specific:3)
        # todo make prompt pool every task
        # todo self.prompt --> task share
        # todo self.task_prompt --> task specific(Dynamic expansion)
        # todo self.total_prompt_pool --> list of each task prompts

        if self.prompt_pool:  # * shared
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
                
        if self.prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
                

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(
            square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def update_task_prompt_pool(self, task_id, prompt_init='uniform'):
        # * make New Task prompt
        task_prompt_pool_shape = (
            self.task_prompt_size, self.length, self.embed_dim)
        if self.prompt_init == 'zero':
            self.assist_prompt = nn.Parameter(
                torch.zeros(task_prompt_pool_shape)).cuda()
        elif self.prompt_init == 'uniform':
            self.assist_prompt = nn.Parameter(
                torch.randn(task_prompt_pool_shape)).cuda()
            nn.init.uniform_(self.assist_prompt, -1, 1)

    def update_assist_prompt(self, task_id):
        if task_id == 0:
            self.prompt_key_layer = nn.Linear(
                self.embed_dim, task_id+1,).cuda()    # dim,numTask
            print('before_layer',self.prompt_key_layer.weight.shape)
            wts = self.assist_prompt.mean(dim=1).mean(dim=0)
            print('wts',wts.shape)
            self.prompt_key_layer.weight.data[task_id] = wts
            print('after_layer',self.prompt_key_layer.weight.shape)
            
        else:
            old_wts = self.prompt_key_layer.weight.data
            old_bias = self.prompt_key_layer.bias.data
            self.prompt_key_layer = nn.Linear(
                self.embed_dim, task_id+1,).cuda()    # dim,numTask
            self.prompt_key_layer.weight.data[:task_id] = old_wts
            self.prompt_key_layer.bias.data[:task_id] = old_bias
            
            cur_wts = self.assist_prompt.mean(dim=1).mean(dim=0)
            self.prompt_key_layer.weight.data[-1] = cur_wts

        print('update Key Weight:', self.prompt_key_layer.weight.shape)
        print('update Key Bias:', self.prompt_key_layer.bias.shape)

    def get_ent_(self, out):
        prob = torch.softmax(out, dim=1)
        ent = -torch.sum(prob*torch.log(prob+1e-5), 1)
        return ent

    def forward(self, x_embed, prompt_mask=None, cls_features=None, test=-1, task_id=None,threshold=0):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[
                    0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError(
                    "Not supported way of calculating embedding keys!")

            if test>=0: #when test
                x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)  # B, C

            else:
                x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)  # B, C
                # Tprompt_logit = self.task_prompt_key[-1](x_embed_norm)
                # print('x_embed_norm',x_embed_norm.shape)
                #* original
                # key_pred = self.prompt_key_layer(x_embed_norm)
                key_pred = self.prompt_key_layer(x_embed_mean)
                out['key_logits'] = key_pred
                out['key_targets'] = torch.full(
                    (key_pred.shape[0],), task_id).long().cuda()

            prompt_norm = self.l2_normalize(self.prompt_key,dim=1)
            similarity = torch.matmul(x_embed_norm,prompt_norm.t())
            _, idx = torch.topk(similarity,k=self.top_k,dim=1)
            #? self.prompt에서 Selection 필요 기존 topk argparser 사용하면 될듯 
            
            if self.pruning:
                new_idx = []
                for i,instance in enumerate(idx): # instance: [1,5,6,7,9]
                    
                    major_prompt_id = [prompt.item() for prompt in instance if similarity[i][prompt]>threshold]
                    while len(major_prompt_id)<self.top_k:
                        major_prompt_id.append(-1)
                    new_idx.append(major_prompt_id)
                idx = torch.Tensor(new_idx).long() # [16,5]
                
                out['iw_pruned_prompts'] = idx #[16,5] -1 포함한값
                batched_prompt_raw = self.prompt[idx]
                batch_size, top_k, length, c = batched_prompt_raw.shape
                for ii,prompts in enumerate(idx): #각 instance 마다
                    dummy_num=0
                    for p in prompts: #prompts: [4,6,7,-1,-1]
                        if p==-1:
                            dummy_num+=1
                    non_dummy=self.top_k - dummy_num
                    if dummy_num>0:
                        dummy_prompts = torch.zeros(dummy_num, self.length, self.embed_dim,requires_grad=False, device="cuda")
                        batched_prompt_raw_i = batched_prompt_raw[ii][:non_dummy]
                        batched_prompt_raw[ii] = torch.cat([batched_prompt_raw_i, dummy_prompts], dim=0)
                
                #! reshape --> assist cat 해준 다음에 하자
                #todo assist expand it self from 1,5,768 to B,1,5,768 (B,1,length,dim)
                
                
                # batched_masked_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C
                
            
            N,L,Dim = self.assist_prompt.shape
            assist_prompt = self.assist_prompt.expand(batch_size,N,L,Dim)
            selected_prompt = torch.cat(
                    [assist_prompt, batched_prompt_raw], dim=1)  # 5,length,C
            B,numP,LengP,p_dim = selected_prompt.shape
            batched_masked_prompt = selected_prompt.reshape(B, numP * LengP, p_dim) # B, top_k * length, C
            

            # Debugging, return sim as well
            out['x_embed_norm'] = x_embed_norm
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(
                    torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(
                    torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(
                0).expand(x_embed.shape[0], -1, -1)

        # key_norm = prompt_norm[idx]
        sim_flag_idx = idx.ne(-1)
        sim_idx = idx[sim_flag_idx]
        batched_key_norm = prompt_norm[sim_idx]
        x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
        
        
        #* original
        sim = batched_key_norm * x_embed_norm # B, top_k, C
        
        #* similarity = torch.matmul(x_embed_norm,prompt_norm.t())
        # sim = torch.matmul(x_embed_norm,batched_key_norm.t())
        reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar
        
        out['reduce_sim'] = reduce_sim
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_masked_prompt.shape[1]
        out['token_per_p'] = batched_masked_prompt.shape[1]/self.length
        out['num_prompt'] = self.pool_size+self.task_prompt_size
        out['num_S_tokens'] = self.task_prompt_size*self.length
        out['tokens'] = batched_masked_prompt
        out['task_prompts']= self.assist_prompt
        out['prompted_embedding'] = torch.cat([batched_masked_prompt, x_embed], dim=1)

        return out
