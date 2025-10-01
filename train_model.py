from utils import *
import torch.nn as nn
CUDA = torch.cuda.is_available()
import torch
from pathlib import Path
from models.modules import ProbabilisticPromptPool as PromptPool
import torch.utils.data
import os
from models.modules import Prompter

def train_one_epoch_train_model(data_loader, net, optimizer):
    net.train()
    
    tl = Averager()
    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        for domain in range(int(net.num_domains)):
            x_input = x_batch[domain * 167:(domain + 1) * 167]
            y = y_batch[domain * 167:(domain + 1) * 167]
            rec_x, mu, logvar = net.featurizer(x_input)            
            

            y_hat = net.classifier_list[domain](mu)
            y_hat_all = net.classifier_list[-1](mu)
            if domain == 0:
                loss = (loss_function(y_hat, y) + loss_function(y_hat_all, y)) / 2
                loss_re = vae_loss(rec_x, mu, logvar, x_input)
            else:
                loss += (loss_function(y_hat, y) + loss_function(y_hat_all, y)) / 2
                loss_re += vae_loss(rec_x, mu, logvar, x_input) 

            _, pred = torch.max(y_hat, 1)
            _, pred_all = torch.max(y_hat_all, 1)
            pred_train.extend(pred.data.tolist())
            pred_train.extend(pred_all.data.tolist())
            act_train.extend(y.data.tolist())
            act_train.extend(y.data.tolist())
        loss /= net.num_domains
        loss_re /= net.num_domains
        t_loss = loss_re + loss
        optimizer.zero_grad()
        t_loss.backward()
        optimizer.step()
        tl.add(t_loss.item())

    return tl.item(), pred_train, act_train
def train_one_epoch_train_experts(data_loader, net, optimizer, prompters):
    net.train()
    net.state_dict()
    for prompter in prompters:
        prompter.train()
    tl = Averager()
    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        loss=0.0
        loss_re=0.0
        for domain in range(int(net.num_domains)):
            x_input = x_batch[domain * 167:(domain + 1) * 167]
            y = y_batch[domain * 167:(domain + 1) * 167]
            x_prompt_input = prompters[domain](x_input)
            rec_x, mu, logvar = net.featurizer(x_prompt_input)
            y_hat = net.classifier_list[domain](mu)
            y_hat_all = net.classifier_list[-1](mu)
            
            loss +=  (loss_function(y_hat, y) + loss_function(y_hat_all, y)) / 2
            loss_re += vae_loss(rec_x, mu, logvar, x_input) 

            _, pred = torch.max(y_hat, 1)
            _, pred_all = torch.max(y_hat_all, 1)
            pred_train.extend(pred.data.tolist())
            pred_train.extend(pred_all.data.tolist())
            act_train.extend(y.data.tolist())
            act_train.extend(y.data.tolist())
        loss /= net.num_domains
        loss_re /= net.num_domains
        t_loss = loss_re + loss
        optimizer.zero_grad()
        t_loss.backward()
        optimizer.step()
        tl.add(t_loss.item())

    return tl.item(), pred_train, act_train

def train_one_epoch_attention_generalization(data_loader, net, optimizer, prompters,attention_model):
    net.train()
    attention_model.train()
    tl = Averager()
    pred_train = []
    act_train = []
    for prompter in prompters:
        prompter.train()
    
    for i, (x_batch, y_batch) in enumerate(data_loader):  # 40次
        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        prompts = [prompter.prompt for prompter in prompters]
        prompts = torch.cat(prompts)
        x = x_batch
        y = y_batch
        adjusted_target_data = attention_model(x, prompts)

        rec_x, mu, logvar = net.featurizer(adjusted_target_data)
        y_hat_all = net.classifier_list[-1](mu)

        loss = loss_function(y_hat_all, y)
        loss_re = vae_loss(rec_x, mu, logvar, x)
        _, pred = torch.max(y_hat_all, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y.data.tolist())

        t_loss = loss_re + loss
        optimizer.zero_grad()
        t_loss.backward()
        optimizer.step()
        tl.add(t_loss.item())

    return tl.item(), pred_train, act_train

def predict(data_loader, model, prompters,attention_model):
    model.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    attention_model.eval()
    for prompter in prompters:
        prompter.eval()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda().squeeze(1)
            prompts = [prompter.prompt for prompter in prompters]
            prompts = torch.cat(prompts)
            adjusted_target_data = attention_model(x_batch, prompts)
            rec_x, mu, logvar = model.featurizer(adjusted_target_data)
            y_hat = model.classifier_list[-1](mu)  # y_hat  60  2
            loss = loss_function(y_hat, y_batch)
            _, pred = torch.max(y_hat, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val
def predict_train_model(data_loader, model):
    model.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda().squeeze(1)

            rec_x, mu, logvar = model.featurizer(x_batch)
            entropy, y_hats, y_hats_pre = torch.zeros((model.num_domains , mu.shape[0])).cuda(
            ), torch.zeros(model.num_domains , mu.shape[0], model.num_class).cuda(), []
            loss = torch.zeros(1)
            for domain in range(model.num_domains ):

                rec_x, mu, logvar = model.featurizer(x_batch)
                y_hat = model.classifier_list[domain](mu)  # y_hat  60  2
                y_hats[domain] = torch.nn.functional.softmax(y_hat, dim=1)
                y_hats_pre.append(y_hat)
                entropy[domain] = model.softmax_entropy(y_hat)
            com_result = torch.zeros(y_hat.shape[0], y_hat.shape[1]).cuda()  # 60* 2
            if model.gamma >= 0:
                weight = 1.0 / (entropy ** model.gamma)
                weight = torch.nn.functional.normalize(weight, p=1, dim=0)
                for i in range(model.num_domains ):
                    com_result += torch.mul(y_hats[i].T, weight[i]).T
            else:

                for i in range(y_hats.shape[1]):
                    idx = entropy[:, i].argmin()
                    com_result[i] = y_hats[idx, i]
            confidences, predict = com_result.softmax(1).max(1)
            loss = loss_function(com_result, y_batch)
            vl.add(loss.item())
            pred_val.extend(predict.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val

def predict_train_experts(data_loader, model, prompters):
    model.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    for prompter in prompters:
        prompter.eval()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):

            x_batch, y_batch = x_batch.cuda(), y_batch.cuda().squeeze(1)
            x_input = x_batch

            x_prompt_input = prompters[0](x_input)
            rec_x, mu, logvar = model.featurizer(x_prompt_input)

            entropy, y_hats, y_hats_pre = torch.zeros((model.num_domains , mu.shape[0])).cuda(
            ), torch.zeros(model.num_domains , mu.shape[0], model.num_class).cuda(), []
            for domain in range(model.num_domains ):
                x_prompt_input = prompters[domain](x_input)
                rec_x, mu, logvar = model.featurizer(x_prompt_input)
                y_hat = model.classifier_list[domain](mu)  # y_hat  60  2
                y_hats[domain] = torch.nn.functional.softmax(y_hat, dim=1)
                y_hats_pre.append(y_hat)
                entropy[domain] = model.softmax_entropy(y_hat)
            com_result = torch.zeros(y_hat.shape[0], y_hat.shape[1]).cuda()  # 60* 2
            if model.gamma >= 0:
                weight = 1.0 / (entropy ** model.gamma)
                weight = torch.nn.functional.normalize(weight, p=1, dim=0)
                for i in range(model.num_domains ):
                    com_result += torch.mul(y_hats[i].T, weight[i]).T
            else:

                for i in range(y_hats.shape[1]):
                    idx = entropy[:, i].argmin()
                    com_result[i] = y_hats[idx, i]
            confidences, predict = com_result.softmax(1).max(1)
            loss = loss_function(com_result, y_batch)
            vl.add(loss.item())
            pred_val.extend(predict.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val


def set_up(args):
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


def loss_function(output, target):
    # 计算基本损失
    base_loss = nn.CrossEntropyLoss()(output, target)

    return base_loss


def train(args, data_train, label_train, data_val, label_val, subject, fold, domain_num):
    seed_all(args.random_seed)
    save_name = '_sub' + str(subject) + '_trial' + str(fold)
    set_up(args)

    train_loader = get_dataloader(data_train, label_train, 128)
    val_loader = get_dataloader(data_val, label_val, 60)
    # MODEL & OPTIMIZER
    expert_dir = Path('./log').joinpath('experts')
    if args.use_meta:
        expert_filenames = [
            f'sub_{subject}_meta_domain_{domain}_prompt_size_{args.prompt_size}_{args.bands}.pt' for domain in range(domain_num)]
    if args.use_zero:
        expert_filenames = [
            f'sub_{subject}_zero_domain_{domain}_prompt_size_{args.prompt_size}_{args.bands}.pt' for domain in range(domain_num)]
    elif args.use_unif:
        expert_filenames = [
            f'sub_{subject}_unif_domain_{domain}_prompt_size_{args.prompt_size}_{args.bands}.pt' for domain in range(domain_num)]
    elif args.use_rand:
        expert_filenames = None
    else:
        expert_filenames = [
            f'{args.model}_{args.dataset.lower()}_{src.lower()}_{args.prompt_size}_{args.bands}.pt' for src in src_domains]

    if expert_filenames is None:
        expert_state_dicts = [Prompter(9, args.prompt_size).state_dict() for _ in src_domains]
    else:
        expert_state_dicts = [torch.load(expert_dir.joinpath(fn))['best_state_dict'] for fn in expert_filenames]
    experts = []
    jitter = args.aug_pt is not None
    with torch.no_grad():
        for state_dict in expert_state_dicts:
            expert = Prompter(9, args.prompt_size).cuda()
            expert.load_state_dict(state_dict)
            if jitter:
                expert.jitter_normal(std=0.03)
            experts.append(expert)

    if args.norm_experts=='yes':
        with torch.no_grad():
            for prompter in experts:
                prompter: Prompter
                prompt_norm = torch.norm(torch.cat([
                    prompter.prompt_t,
                    prompter.prompt_b,
                    prompter.prompt_l.transpose(2, 3),
                    prompter.prompt_r.transpose(2, 3),
                ], dim=3), p=2)
                prompter.prompt_t /= prompt_norm
                prompter.prompt_b /= prompt_norm
                prompter.prompt_l /= prompt_norm
                prompter.prompt_r /= prompt_norm
    

    model = PGDG(domain_num).cuda()
    model.load_state_dict(torch.load('./save/' + 'cross_sub_train_model' + '_' +args.bands + '_' + str(subject) + '_' + 'max-acc.pth',weights_only=True))
    model_attention = MultiHeadAttentionPrompt(in_dim=5,use_softmax=args.use_softmax, use_tanh=args.use_tanh, num_heads=5).cuda()
    para = get_trainable_parameter_num(model)
    print('Model {} size:{}'.format(args.model, para))
    optimizer_parameters = []     
    if args.tuning_prompt=='yes':           
        for prompter in experts:
            optimizer_parameters += list(prompter.parameters())
    
    
    if args.tuning=='no':
        optimizer_parameters+=list(model_attention.parameters())
        
    if args.tuning=='classifier':      
        optimizer_parameters+=list(model_attention.parameters())
        optimizer_parameters+=list(model.classifier_list[-1].parameters())
        
    if args.tuning=='all':     
        optimizer_parameters+=list(model_attention.parameters())
        optimizer_parameters+=list(model.parameters())
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.learning_rate, betas=(0.5, 0.999))    
    def save_model(name):
        previous_model = osp.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.save_path, '{}.pth'.format(name)))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['f1_val'] = []
    #best_state_dict = [[] for _ in range(domain_num)]
    timer = Timer()
    for epoch in range(1, args.max_epoch + 1):
        loss_train, pred_train, act_train = train_one_epoch_attention_generalization(
            data_loader=train_loader, net=model, optimizer=optimizer, prompters=experts,attention_model=model_attention)

        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('subject:',subject)
        print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss_train, acc_train, f1_train))

        loss_val, pred_val, act_val = predict(
            data_loader=val_loader, model=model, prompters=experts,attention_model=model_attention)
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.
              format(epoch, loss_val, acc_val, f1_val))

        if acc_train > trlog['max_acc']:
            trlog['max_acc'] = acc_train

            #save_model(args.sub_dependent_or_not + '_' + str(subject) + '_' + 'max-acc')

        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)
        trlog['f1_val'].append(f1_val)

        print('ETA:{}/{} SUB:{} FOLD:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                subject, fold))
    save_name_ = 'trlog' + save_name + '.txt'
    ensure_path(osp.join(args.save_path, 'log_train'))
    torch.save(trlog, osp.join(args.save_path, 'log_train', save_name_))

    return max(trlog['val_acc']), max(trlog['f1_val'])

def train_models(args, data_train, label_train, data_val, label_val, subject, fold, domain_num):
    seed_all(args.random_seed)
    save_name = '_sub' + str(subject) + '_trial' + str(fold)
    set_up(args)
    train_loader = get_dataloader(data_train, label_train, args.batch_size)
    val_loader = get_dataloader(data_val, label_val, 60)
    model = PGDG(domain_num).cuda()
    para = get_trainable_parameter_num(model)
    print('Model {} size:{}'.format(args.model, para))
    def save_model(name):
        previous_model = osp.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.save_path, '{}.pth'.format(name)))
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['f1_val'] = []

    timer = Timer()
    lr = args.learning_rate
    optimizer = torch.optim.AdamW(
            (list(model.parameters()) ), lr=lr, betas=(0.5, 0.999))
    for epoch in range(1, args.max_epoch + 1):
        loss_train, pred_train, act_train = train_one_epoch_train_model(
            data_loader=train_loader, net=model, optimizer=optimizer)
        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('subject:',subject)
        print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss_train, acc_train, f1_train))

        loss_val, pred_val, act_val = predict_train_model(
            data_loader=val_loader, model=model
        )
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.
              format(epoch, loss_val, acc_val, f1_val))

        if acc_train > trlog['max_acc']:
            trlog['max_acc'] = acc_train
            save_model(args.sub_dependent_or_not + '_'+args.bands + '_' + str(subject) + '_' + 'max-acc')
            

        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)
        trlog['f1_val'].append(f1_val)
        

        print('ETA:{}/{} SUB:{} FOLD:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                subject, fold))
    save_name_ = 'trlog' + save_name + '.txt'
    ensure_path(osp.join(args.save_path, 'log_train'))
    torch.save(trlog, osp.join(args.save_path, 'log_train', save_name_))

    return max(trlog['val_acc']), max(trlog['f1_val'])


def train_experts(args, data_train, label_train, data_val, label_val, subject, fold, domain_num):
    seed_all(args.random_seed)
    save_name = '_sub' + str(subject) + '_trial' + str(fold)
    set_up(args)
    train_loader = get_dataloader(data_train, label_train, args.batch_size)
    val_loader = get_dataloader(data_val, label_val, 60)
    model = PGDG(domain_num).cuda()
    model.load_state_dict(torch.load('./save/' + 'cross_sub_train_model' + '_' +args.bands + '_'+ str(subject) + '_' + 'max-acc.pth'))
    prompters = [Prompter(9, args.prompt_size, init=args.init_param).cuda()
                 for _ in range(domain_num)]  # 是否需要第N+1domian
    with torch.no_grad():
        if args.zero_init:
            for prompter in prompters:
                prompter.prompt_t = nn.Parameter(torch.zeros_like(prompter.prompt_t))
                prompter.prompt_b = nn.Parameter(torch.zeros_like(prompter.prompt_b))
                prompter.prompt_l = nn.Parameter(torch.zeros_like(prompter.prompt_l))
                prompter.prompt_r = nn.Parameter(torch.zeros_like(prompter.prompt_r))
        elif args.unif_init:
            for prompter in prompters:
                prompter.prompt_t = nn.Parameter((torch.rand_like(prompter.prompt_t) * 2 - 1) * args.init_param)
                prompter.prompt_b = nn.Parameter((torch.rand_like(prompter.prompt_b) * 2 - 1) * args.init_param)
                prompter.prompt_l = nn.Parameter((torch.rand_like(prompter.prompt_l) * 2 - 1) * args.init_param)
                prompter.prompt_r = nn.Parameter((torch.rand_like(prompter.prompt_r) * 2 - 1) * args.init_param)

        else:
            for prompter in prompters:
                prompter.prompt_t = nn.Parameter(torch.randn_like(prompter.prompt_t) * args.init_param)
                prompter.prompt_b = nn.Parameter(torch.randn_like(prompter.prompt_b) * args.init_param)
                prompter.prompt_l = nn.Parameter(torch.randn_like(prompter.prompt_l) * args.init_param)
                prompter.prompt_r = nn.Parameter(torch.randn_like(prompter.prompt_r) * args.init_param)
    para = get_trainable_parameter_num(model)
    print('Model {} size:{}'.format(args.model, para))
    prompters_optimizer_parameters = []
    for prompter in prompters:
        prompters_optimizer_parameters += list(prompter.parameters())
    def save_model(name):
        previous_model = osp.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.save_path, '{}.pth'.format(name)))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['f1_val'] = []
    best_state_dict = [[] for _ in range(domain_num)]
    timer = Timer()
    lr = args.learning_rate
    optimizer = torch.optim.AdamW(
            prompters_optimizer_parameters, lr=lr, betas=(0.5, 0.999))
    for epoch in range(1, args.max_epoch + 1):
        loss_train, pred_train, act_train = train_one_epoch_train_experts(
            data_loader=train_loader, net=model, optimizer=optimizer, prompters=prompters)

        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('subject:',subject)
        print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss_train, acc_train, f1_train))

        loss_val, pred_val, act_val = predict_train_experts(
            data_loader=val_loader, model=model, prompters=prompters
        )
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.
              format(epoch, loss_val, acc_val, f1_val))

        if acc_train > trlog['max_acc']:
            trlog['max_acc'] = acc_train
            best_epoch = epoch
            #save_model(args.sub_dependent_or_not + '_'+args.bands + '_' + str(subject) + '_' + 'max-acc')
            

            for domain in range(domain_num):
                best_state_dict[domain] = {key: val.clone().cpu()
                                           for key, val in prompters[domain].state_dict().items()}            

        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)
        trlog['f1_val'].append(f1_val)
        expert_path = Path('./log').joinpath('experts')
        expert_path.mkdir(parents=True, exist_ok=True)
        for domain in range(domain_num):
            if args.meta_init is not None:
                expert_path_name = expert_path.joinpath(
                    f'sub_{subject}_meta_domain_{domain}_prompt_size_{args.prompt_size}_{args.bands}.pt')
            elif args.zero_init:
                expert_path_name = expert_path.joinpath(
                    f'sub_{subject}_zero_domain_{domain}_prompt_size_{args.prompt_size}_{args.bands}.pt')
            elif args.unif_init:
                expert_path_name = expert_path.joinpath(
                    f'sub_{subject}_unif_domain_{domain}_prompt_size_{args.prompt_size}_{args.bands}.pt')
            else:
                expert_path_name = expert_path.joinpath(
                    f'{args.model}_{args.dataset.lower()}_{args.domain.lower()}_{args.prompt_size}_{args.bands}.pt')

            torch.save({
                'epoch': best_epoch,
                'best_state_dict': best_state_dict[domain],
                'last_state_dict': {key: val.clone().cpu() for key, val in prompter.state_dict().items()},
            }, expert_path_name)
        print('ETA:{}/{} SUB:{} FOLD:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                subject, fold))
    save_name_ = 'trlog' + save_name + '.txt'
    ensure_path(osp.join(args.save_path, 'log_train'))
    torch.save(trlog, osp.join(args.save_path, 'log_train', save_name_))

    return max(trlog['val_acc']), max(trlog['f1_val'])

def vae_loss(rec_x, mu, logvar, x):
    # Reconstruction loss
    loss_re_x_t = reconstruction_loss(x, rec_x)

    # KL divergence
    kl_loss_x = kl_divergence(mu, logvar)

    return loss_re_x_t + kl_loss_x


def reconstruction_loss(x, x_recon):
    # Compute mean squared error (MSE) or cross-entropy loss
    mse_loss = torch.mean((x - x_recon)**2)  # Example: MSE loss
    # cross_entropy_loss = F.binary_cross_entropy(x_recon, x)  # Example: Cross-entropy loss
    return mse_loss


def kl_divergence(mu, log_var):
    # Compute KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kl_loss