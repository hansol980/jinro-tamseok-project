class FLServer:
    def __init__(self, model):
        self.global_model = model
        self.received_gradients = None

    def receive_gradients(self, gradients):
        """클라이언트로부터 압축된 그래디언트를 수신합니다."""
        self.received_gradients = gradients
        # 실제 연합학습에서는 여기서 여러 클라이언트의 그래디언트를 평균내어 global_model을 업데이트합니다.
        
    def get_received_gradients(self):
        return self.received_gradients