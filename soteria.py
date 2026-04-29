import torch
import torch.nn as nn

def apply_soteria_defense(model, data, defended_layer_name, prune_rate=0.8):
    """
    Soteria 방어 기법: 
    선택된 특정 레이어의 표현(Representation)을 분석하여, 
    데이터 유출에 치명적인 노드의 기울기만 선택적으로 0으로 만듭니다.
    """

    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            # FC 레이어의 가중치를 방어하기 위해서는 입력값(Representation)이 필요합니다.
            activations[name] = input[0].detach() # 특징값 저장
        return hook

    layer_to_defend = dict(model.named_modules())[defended_layer_name]
    handle = layer_to_defend.register_forward_hook(get_activation(defended_layer_name))

    # Hook을 발동시켜 활성화값을 얻기 위해 순전파(Forward)를 한 번 실행합니다.
    with torch.no_grad():
        model(data)

    r = activations[defended_layer_name] 
    
    importance_score = torch.abs(r).mean(dim=0) 
    
    num_features = importance_score.numel()
    num_prune = int(num_features * prune_rate)
    
    sorted_scores, _ = torch.sort(importance_score)
    threshold = sorted_scores[-num_prune] 

    mask = (importance_score < threshold).float()
    
    for name, param in model.named_parameters():
        if defended_layer_name in name and param.grad is not None:
            
            if len(param.grad.shape) == 2 and param.grad.shape[1] == mask.shape[0]:
                param.grad *= mask.unsqueeze(0) 

            elif len(param.grad.shape) == 1 and param.grad.shape[0] == mask.shape[0]:
                param.grad *= mask

    handle.remove()
