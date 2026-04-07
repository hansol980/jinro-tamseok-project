import torch

class FLServer:
    def __init__(self, model):
        self.global_model = model
        self.received_gradients_list = [] # 여러 클라이언트의 그래디언트를 저장

    def receive_gradients(self, gradients):
        """클라이언트로부터 압축된 그래디언트를 수신하여 리스트에 추가합니다."""
        self.received_gradients_list.append(gradients)
        
    def get_target_client_gradients(self, client_index=0):
        """특정 클라이언트의 데이터를 탈취하기 위해 해당 그래디언트를 반환합니다."""
        return self.received_gradients_list[client_index]

    def aggregate_and_update(self, lr=0.01):
        """수신된 모든 클라이언트의 그래디언트를 평균내어 글로벌 모델을 업데이트합니다."""
        if not self.received_gradients_list:
            print("No gradients received.")
            return
            
        num_clients = len(self.received_gradients_list)
        print(f"\n--- Server Aggregation Phase ---")
        print(f"Aggregating gradients from {num_clients} clients...")
        
        # 글로벌 모델의 파라미터 형태와 동일한 0 텐서 리스트 생성
        aggregated_grads = [torch.zeros_like(p.data) if p.requires_grad else None 
                            for p in self.global_model.parameters()]
        
        # 1. 수신한 모든 그래디언트 합산
        for client_grads in self.received_gradients_list:
            for i, g in enumerate(client_grads):
                if g is not None:
                    # 압축(Sparsification) 과정에서 0으로 가지치기 된 값이 있어도 그대로 합산 가능합니다.
                    aggregated_grads[i] += g / num_clients
                    
        # 2. 파라미터 업데이트 (SGD 방식 적용)
        with torch.no_grad():
            for param, agg_grad in zip(self.global_model.parameters(), aggregated_grads):
                if agg_grad is not None:
                    param.data -= lr * agg_grad
                    
        print("Global model weights have been successfully updated.")
        
        # 다음 연합학습 라운드를 위해 리스트 초기화
        self.received_gradients_list = []